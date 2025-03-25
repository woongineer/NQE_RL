import torch
import torch.nn as nn


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
        x = self.cnn(x)            # (B, hidden_dim, D, Q)
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


class PolicyInsert(nn.Module):
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
        x = self.cnn(x)            # (B, hidden_dim, D, Q)
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
