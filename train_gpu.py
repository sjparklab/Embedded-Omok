# train_gpu.py
# GPU용 학습 스크립트 (venv 환경에서 --examples 등 인자 사용 가능)
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import argparse
import pickle
from tqdm import tqdm
from alphazero_model import AlphaZeroNet

class AZDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        s, pi, z = self.examples[idx]
        return s, pi, z

def load_examples(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_checkpoint(state, path):
    torch.save(state, path)

def train_loop(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    examples = load_examples(args.examples)
    print(f"Loaded {len(examples)} examples from {args.examples}")
    ds = AZDataset(examples)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True, drop_last=True)

    net = AlphaZeroNet(board_size=args.board_size, num_filters=args.num_filters).to(device)
    start_epoch = 0
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        if 'model' in ckpt:
            net.load_state_dict(ckpt['model'])
        else:
            net.load_state_dict(ckpt)
        start_epoch = ckpt.get('epoch', 0) if isinstance(ckpt, dict) else 0
        print("Loaded checkpoint", args.checkpoint, "start_epoch", start_epoch)

    opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    if args.torch_compile and hasattr(torch, 'compile'):
        net = torch.compile(net)

    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        net.train()
        running_loss = 0.0
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            s_batch, pi_batch, z_batch = batch
            s_batch = s_batch.to(device, non_blocking=True)
            pi_batch = pi_batch.to(device, non_blocking=True)
            z_batch = z_batch.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.use_amp):
                logits, v = net(s_batch)
                logp = F.log_softmax(logits, dim=1)
                loss_p = - (logp * pi_batch).sum(dim=1).mean()
                loss_v = (v - z_batch).pow(2).mean()
                loss = loss_p + args.value_weight * loss_v

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss/ (pbar.n + 1))

        # checkpoint
        ckpt_state = {'model': net.state_dict(), 'opt': opt.state_dict(), 'epoch': epoch+1}
        save_checkpoint(ckpt_state, os.path.join(args.out_dir, f"checkpoint_epoch{epoch+1}.pt"))
        print(f"[Epoch {epoch+1}] saved checkpoint to {args.out_dir}")

    # final save
    save_checkpoint({'model': net.state_dict(), 'opt': opt.state_dict(), 'epoch': args.epochs},
                    os.path.join(args.out_dir, "checkpoint_final.pt"))
    print(f"Training finished. Final model saved to {args.out_dir}/checkpoint_final.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=str, default="examples.pkl", help="Path to examples pickle")
    parser.add_argument("--out_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--value_weight", type=float, default=1.0)
    parser.add_argument("--board_size", type=int, default=19)
    parser.add_argument("--num_filters", type=int, default=128)
    parser.add_argument("--use_amp", action="store_true", help="Enable automatic mixed precision")
    parser.add_argument("--torch_compile", action="store_true", help="Enable torch.compile if available")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint to resume from")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    args = parser.parse_args()
    train_loop(args)