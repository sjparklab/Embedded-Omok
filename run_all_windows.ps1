<#
run_all_windows.ps1 (venv-aware)

PowerShell 전체 파이프라인(venv friendly):
  - 여러 self-play worker 시작 (각 worker는 selfplay_worker_win.py 실행)
  - 모든 worker 완료 대기
  - examples 병합
  - train_gpu.py로 학습 실행 (venv python 사용)

Usage:
  .\run_all_windows.ps1 -Gpus "0","1" -WorkersPerGpu 1 -GamesPerWorker 50 -NSim 400 -ModelPath "az_net.pt" -VenvPath ".\venv"
#>

param(
    [string[]] $Gpus = @("0"),
    [int] $WorkersPerGpu = 1,
    [int] $GamesPerWorker = 50,
    [int] $NSim = 200,
    [string] $ModelPath = "",
    [string] $ExamplesDir = "examples",
    [string] $LogDir = "logs",
    [string] $TrainExamples = "examples.pkl",
    [int] $TrainEpochs = 5,
    [int] $TrainBatchSize = 256,
    [int] $NumFilters = 128,
    [string] $VenvPath = ""   # <-- optional, e.g. ".\venv"
)

# Resolve python executable (venv-aware)
function Get-PythonExe {
    param([string]$venv)
    if ($venv -and (Test-Path $venv)) {
        $p = Join-Path $venv "Scripts\python.exe"
        if (Test-Path $p) { return (Resolve-Path $p).Path }
    }
    # fallback: check ./venv
    $pLocal = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
    if (Test-Path $pLocal) { return (Resolve-Path $pLocal).Path }
    # final fallback: 'python' in PATH
    return "python"
}

$pythonExe = Get-PythonExe -venv $VenvPath
Write-Host "Using python executable:" $pythonExe

# Ensure directories
if (!(Test-Path -Path $ExamplesDir)) { New-Item -ItemType Directory -Path $ExamplesDir | Out-Null }
if (!(Test-Path -Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

# Start worker processes
$workerProcs = @()
foreach ($gpu in $Gpus) {
    for ($i = 0; $i -lt $WorkersPerGpu; $i++) {
        $device = "cuda:$gpu"
        $outFile = Join-Path $LogDir ("selfplay_worker_gpu{0}_pid{1}.log" -f $gpu, (Get-Random))
        $errFile = Join-Path $LogDir ("selfplay_worker_gpu{0}_pid{1}.err" -f $gpu, (Get-Random))
        $modelArg = if ($ModelPath -ne "") { "--model `"$ModelPath`"" } else { "" }
        $args = "--out_dir `"$ExamplesDir`" --games $GamesPerWorker --n_sim $NSim --device $device $modelArg"
        Write-Host "Starting worker on device $device (games=$GamesPerWorker, nsim=$NSim). Logs: $outFile"

        # Start with venv python (no activation needed because we call the venv python directly)
        $proc = Start-Process -FilePath $pythonExe -ArgumentList "selfplay_worker_win.py $args" -RedirectStandardOutput $outFile -RedirectStandardError $errFile -WindowStyle Hidden -PassThru
        $workerProcs += $proc
    }
}

# Wait for all workers to finish
Write-Host "Waiting for workers to complete..."
Wait-Process -InputObject $workerProcs

Write-Host "All workers finished. Merging example files..."

# Merge examples (use python from venv)
& $pythonExe merge_examples.py --input_dir $ExamplesDir --out_file $TrainExamples

Write-Host "Training using $TrainExamples ..."
# Train on GPU (use venv python)
$trainArgs = "--examples `"$TrainExamples`" --out_dir checkpoints --epochs $TrainEpochs --batch_size $TrainBatchSize --num_filters $NumFilters --use_amp"
& $pythonExe train_gpu.py $trainArgs

Write-Host "Training completed. Check checkpoints directory for saved models."