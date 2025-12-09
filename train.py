# train.py
# 수집된 examples.pkl 로 네트워크 학습 (간단한 학습 루프)
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from alphazero_model import AlphaZeroNet

class AZDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        s, pi, z = self.examples[idx]
        return torch.from_numpy(s).float(), torch.from_numpy(pi).float(), torch.tensor(z).float()

def train(examples_path="examples.pkl", model_out="az_net.pt", epochs=5, batch_size=64, lr=1e-3, device='cpu', board_size=19):
    with open(examples_path, "rb") as f:
        examples = pickle.load(f)
    if len(examples) == 0:
        print("No examples found.")
        return
    ds = AZDataset(examples)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    net = AlphaZeroNet(board_size=board_size).to(device)
    opt = optim.Adam(net.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        net.train()
        total_loss = 0.0
        for s, pi, z in dl:
            s = s.to(device)
            pi = pi.to(device)
            z = z.to(device)
            logits, v = net(s)
            # policy loss: cross-entropy between pi (target probs) and logits
            logp = F.log_softmax(logits, dim=1)
            loss_p = -torch.sum(logp * pi, dim=1).mean()
            # value loss: MSE
            loss_v = (v - z).pow(2).mean()
            loss = loss_p + loss_v
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
        print(f"Epoch {epoch} loss {total_loss/len(dl):.4f}")
        torch.save(net.state_dict(), f"{model_out}.epoch{epoch}")
    # final save
    torch.save(net.state_dict(), model_out)
    print(f"Model saved to {model_out}")

if __name__ == "__main__":
    train(examples_path="examples.pkl", model_out="az_net.pt", epochs=3, batch_size=64, lr=1e-3, device='cpu')