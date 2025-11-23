import torch
import torch.nn as nn
import torch.nn.functional as F

class AlphaZeroNet(nn.Module):
    def __init__(self, board_size=19, in_channels=2, num_filters=64):
        super().__init__()
        self.board_size = board_size
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        # 간단한 residual-like block (1개)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)

        # policy head
        self.p_conv = nn.Conv2d(num_filters, 2, kernel_size=1)
        self.p_bn = nn.BatchNorm2d(2)
        self.p_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)

        # value head
        self.v_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.v_bn = nn.BatchNorm2d(1)
        self.v_fc1 = nn.Linear(board_size * board_size, 64)
        self.v_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x: (B, C=2, H, W)
        z = F.relu(self.bn1(self.conv1(x)))
        z = F.relu(self.bn2(self.conv2(z)))

        # policy
        p = F.relu(self.p_bn(self.p_conv(z)))
        p = p.view(p.size(0), -1)
        p = self.p_fc(p)  # (B, H*W)

        # value
        v = F.relu(self.v_bn(self.v_conv(z)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.v_fc1(v))
        v = torch.tanh(self.v_fc2(v)).squeeze(-1)  # in [-1,1]

        return p, v