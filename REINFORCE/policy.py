import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyRemove(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.rnn = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.output_layer = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        # x: (B, D, Q, hidden_dim)
        x = x.permute(0, 3, 1, 2)  # (B, hidden_dim, D, Q) → CNN 용
        x = self.cnn(x)  # (B, hidden_dim, D, Q)
        x = x.permute(0, 3, 2, 1)  # (B, Q, D, hidden_dim)

        outputs = []
        for q in range(x.size(1)):
            q_seq = x[:, q, :, :]  # (B, D, H)
            rnn_out, _ = self.rnn(q_seq)  # (B, D, 2H)
            outputs.append(rnn_out)

        rnn_out = torch.stack(outputs, dim=2)  # (B, D, Q, 2H)
        logits = self.output_layer(rnn_out).squeeze(-1)  # (B, D, Q)
        probs = torch.sigmoid(logits)
        return probs


class PolicyInsertBasic(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.rnn = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.output_layer = nn.Linear(hidden_dim * 2, len(insert_gate_map))

    def forward(self, x, qubit_idx, depth_idx):
        # x: (1, D, Q, input_dim)
        x = x.permute(0, 3, 1, 2)  # (1, input_dim, D, Q)
        x = self.cnn(x)  # (1, hidden_dim, D, Q)
        x = x.permute(0, 3, 2, 1)  # (1, Q, D, hidden_dim)

        outputs = []
        for q in range(x.size(1)):
            q_seq = x[:, q, :, :]  # (1, D, H)
            rnn_out, _ = self.rnn(q_seq)  # (1, D, 2H)
            outputs.append(rnn_out)

        rnn_out = torch.stack(outputs, dim=2)  # (1, D, Q, 2H)
        target_feat = rnn_out[0, depth_idx, qubit_idx, :]  # (2H,)
        logits = self.output_layer(target_feat)  # (26,)
        probs = F.softmax(logits, dim=-1)
        return probs  # shape: (26,)


class PolicyInsertWithMask(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=input_dim + 1, out_channels=hidden_dim, kernel_size=3, padding=1),  # +1 for mask
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.rnn = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.output_layer = nn.Linear(hidden_dim * 2, len(insert_gate_map))

    def forward(self, x, qubit_idx, depth_idx):
        # x: (1, D, Q, input_dim)
        B, D, Q, C = x.shape

        # 1. Create position mask: shape (1, D, Q, 1)
        mask = torch.zeros((1, D, Q, 1), dtype=x.dtype, device=x.device)
        mask[0, depth_idx, qubit_idx, 0] = 1.0

        # 2. Concatenate mask to input
        x_aug = torch.cat([x, mask], dim=-1)  # (1, D, Q, input_dim + 1)

        # 3. Pass through CNN + RNN
        x_aug = x_aug.permute(0, 3, 1, 2)  # (1, input_dim+1, D, Q)
        x_aug = self.cnn(x_aug)  # (1, hidden_dim, D, Q)
        x_aug = x_aug.permute(0, 3, 2, 1)  # (1, Q, D, hidden_dim)

        outputs = []
        for q in range(x_aug.size(1)):
            q_seq = x_aug[:, q, :, :]  # (1, D, H)
            rnn_out, _ = self.rnn(q_seq)  # (1, D, 2H)
            outputs.append(rnn_out)

        rnn_out = torch.stack(outputs, dim=2)  # (1, D, Q, 2H)
        target_feat = rnn_out[0, depth_idx, qubit_idx, :]  # (2H,)
        logits = self.output_layer(target_feat)  # (26,)
        probs = F.softmax(logits, dim=-1)
        return probs


insert_gate_map = {
    'RX_0': 0,
    'RX_1': 1,
    'RX_2': 2,
    'RX_3': 3,
    'RY_0': 4,
    'RY_1': 5,
    'RY_2': 6,
    'RY_3': 7,
    'RZ_0': 8,
    'RZ_1': 9,
    'RZ_2': 10,
    'RZ_3': 11,
    'CNOT_1': 12,
    'CNOT_2': 13,
    'CNOT_3': 14,
    'H': 15,
    'I': 16,
}
