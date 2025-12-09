<#
run_finetune_pipeline.ps1

Full pipeline (single script) — venv-aware PowerShell script that runs:
  1) Self-play data generation using a pretrained model (iot12345_az via selfplay_worker_win.py)
  2) Merge per-worker example files into one examples.pkl
  3) Fine-tune (train) the network on the merged examples (train_gpu.py, checkpoint resume)
  4) Pick the final checkpoint saved by training
  5) Evaluate the fine-tuned model vs the alpha-beta baseline (evaluate.py)

Usage (example):
  .\run_finetune_pipeline.ps1 -VenvPath ".\venv" -ModelPath "checkpoints/supervised_pretrain.pt" -Workers 2 -Gpus "0","1" -GamesPerWorker 500 -NSim 400 -TrainEpochs 3 -EvalGames 200

Notes:
- The script expects these helper scripts to exist in the repo:
  selfplay_worker_win.py, merge_examples.py, train_gpu.py, evaluate.py
- Ensure your venv has required packages (torch, numpy, tqdm).
- On Windows, PowerShell ExecutionPolicy may need to allow script execution.
#>

param(
    [string] $VenvPath = ".\venv",
    [string] $ModelPath = "",

    [int] $Workers = 1,
    [string[]] $Gpus = @("0"),                # e.g. "0","1"
    [int] $GamesPerWorker = 60,
    [int] $NSim = 400,

    [string] $ExamplesDir = "examples_pretrain",
    [string] $MergedExamples = "examples_pretrain_merged.pkl",

    [string] $CheckpointsDir = "checkpoints_finetune",
    [int] $TrainEpochs = 3,
    [int] $TrainBatchSize = 256,
    [int] $NumFilters = 128,
    [switch] $UseAmp,

    [int] $EvalGames = 200,
    [int] $EvalNSim = 400,
    [string] $EvalDevice = "cuda:0",

    [switch] $ShuffleMergedExamples,
    [switch] $Verbose
)

function Resolve-PythonExe {
    param([string]$venv)
    if ($venv -and (Test-Path $venv)) {
        $p = Join-Path $venv "Scripts\python.exe"
        if (Test-Path $p) { return (Resolve-Path $p).Path }
    }
    # fallback local ./venv
    $pLocal = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
    if (Test-Path $pLocal) { return (Resolve-Path $pLocal).Path }
    return "python"
}

function Write-Log {
    param([string]$msg)
    $t = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$t] $msg"
}

# Start
$pythonExe = Resolve-PythonExe -venv $VenvPath
Write-Log "Using Python executable: $pythonExe"

# Validate helper scripts presence
$requiredScripts = @("selfplay_worker_win.py","merge_examples.py","train_gpu.py","evaluate.py")
foreach ($s in $requiredScripts) {
    if (-not (Test-Path (Join-Path $PSScriptRoot $s))) {
        Write-Error "Required script '$s' not found in script directory ($PSScriptRoot). Please ensure it exists."
        exit 1
    }
}

# Prepare directories
if (-not (Test-Path $ExamplesDir)) { New-Item -ItemType Directory -Path $ExamplesDir | Out-Null }
if (-not (Test-Path $CheckpointsDir)) { New-Item -ItemType Directory -Path $CheckpointsDir | Out-Null }

# 1) Self-play generation (possibly multiple workers)
Write-Log "STEP 1: Self-play generation starting (Workers=$Workers, Games/worker=$GamesPerWorker, n_sim=$NSim)"
$workerProcs = @()
for ($i = 0; $i -lt $Workers; $i++) {
    $gpuIndex = $Gpus[$i % $Gpus.Length]
    $device = "cuda:$gpuIndex"
    $outFile = Join-Path $ExamplesDir ("selfplay_worker_{0}_{1}.log" -f $i, (Get-Random))
    $errFile = Join-Path $ExamplesDir ("selfplay_worker_{0}_{1}.err" -f $i, (Get-Random))
    $modelArg = ""
    if ($ModelPath -ne "") { $modelArg = "--model `"$ModelPath`"" }
    $args = "--out_dir `"$ExamplesDir`" --games $GamesPerWorker --n_sim $NSim --device $device $modelArg"

    Write-Log "Starting worker $i on $device. Logs: $outFile"
    # Start in background and capture process object
    $proc = Start-Process -FilePath $pythonExe -ArgumentList "selfplay_worker_win.py $args" -RedirectStandardOutput $outFile -RedirectStandardError $errFile -WindowStyle Hidden -PassThru
    $workerProcs += $proc
}

# Wait for all workers to finish
if ($workerProcs.Count -gt 0) {
    Write-Log "Waiting for $($workerProcs.Count) worker(s) to finish..."
    Wait-Process -InputObject $workerProcs
    Write-Log "All self-play workers finished."
} else {
    Write-Log "No workers started. Skipping self-play generation step."
}

# 2) Merge examples
Write-Log "STEP 2: Merging examples from $ExamplesDir -> $MergedExamples"
$mergeArgs = "--input_dir `"$ExamplesDir`" --out_file `"$MergedExamples`""
if ($ShuffleMergedExamples) { $mergeArgs += " --shuffle" }
$mergeCmd = "$pythonExe merge_examples.py $mergeArgs"
Write-Log "Running: $mergeCmd"
& $pythonExe merge_examples.py --input_dir $ExamplesDir --out_file $MergedExamples $(if ($ShuffleMergedExamples) { "--shuffle" } )
if ($LASTEXITCODE -ne 0) {
    Write-Error "merge_examples.py failed (exit code $LASTEXITCODE). Aborting."
    exit 1
}
Write-Log "Merge completed: $MergedExamples"

# 3) Fine-tune (train)
Write-Log "STEP 3: Fine-tune/train on merged examples"

$useAmpArg = ""
if ($UseAmp) { $useAmpArg = "--use_amp" }

$checkpointArg = ""
if ($ModelPath -ne "") { $checkpointArg = "--checkpoint `"$ModelPath`"" }

$trainArgs = @(
    "--examples", "`"$MergedExamples`"",
    "--out_dir", "`"$CheckpointsDir`"",
    "--epochs", $TrainEpochs,
    "--batch_size", $TrainBatchSize,
    "--num_filters", $NumFilters,
    $useAmpArg,
    $checkpointArg
) -join " "

Write-Log "Running training: $pythonExe train_gpu.py $trainArgs"
# Use call operator (&) to run synchronously so we can capture exit code.
& $pythonExe train_gpu.py --examples $MergedExamples --out_dir $CheckpointsDir --epochs $TrainEpochs --batch_size $TrainBatchSize --num_filters $NumFilters $useAmpArg $checkpointArg
if ($LASTEXITCODE -ne 0) {
    Write-Error "train_gpu.py failed (exit code $LASTEXITCODE). Aborting."
    exit 1
}
Write-Log "Training finished."

# 4) Determine final checkpoint
$finalCkpt = Join-Path $CheckpointsDir "checkpoint_final.pt"
if (-not (Test-Path $finalCkpt)) {
    # fallback: pick last checkpoint_epoch*.pt by highest epoch number
    $files = Get-ChildItem -Path $CheckpointsDir -Filter "checkpoint_epoch*.pt" | Sort-Object Name
    if ($files.Count -gt 0) {
        $finalCkpt = $files[-1].FullName
        Write-Log "checkpoint_final.pt not found. Using latest epoch checkpoint: $finalCkpt"
    } else {
        # maybe training saved without .pt names; find any .pt
        $any = Get-ChildItem -Path $CheckpointsDir -Filter "*.pt" | Sort-Object LastWriteTime
        if ($any.Count -gt 0) {
            $finalCkpt = $any[-1].FullName
            Write-Log "Using latest checkpoint file: $finalCkpt"
        } else {
            Write-Error "No checkpoint found in $CheckpointsDir. Aborting."
            exit 1
        }
    }
} else {
    Write-Log "Found final checkpoint: $finalCkpt"
}

# 5) Evaluation
Write-Log "STEP 5: Evaluation: model=$finalCkpt, games=$EvalGames, n_sim=$EvalNSim"
$evalArgs = @(
    "--model", "`"$finalCkpt`"",
    "--games", $EvalGames,
    "--n_sim", $EvalNSim,
    "--device", "`"$EvalDevice`"",
    "--first", "alternate"
) -join " "

Write-Log "Running evaluation: $pythonExe evaluate.py $evalArgs"
& $pythonExe evaluate.py --model $finalCkpt --games $EvalGames --n_sim $EvalNSim --device $EvalDevice --first alternate
if ($LASTEXITCODE -ne 0) {
    Write-Error "evaluate.py failed (exit code $LASTEXITCODE)."
    exit 1
}

Write-Log "Pipeline completed successfully."
# end of script