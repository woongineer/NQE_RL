import pennylane as qml
import torch
import torch.nn as nn

from utils import quantum_embedding

num_qubit = 4
dev = qml.device('default.qubit', wires=num_qubit)


class CNNExtract(nn.Module):
    def __init__(self, in_channels=1, out_channels=16):
        super(CNNExtract, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        feat = self.conv(x)
        feat = feat.mean(dim=[2, 3])
        return feat


class CNNLSTM(nn.Module):
    def __init__(self, feature_dim=16, hidden_dim=32, output_dim=100, num_layers=1):
        super(CNNLSTM, self).__init__()

        self.cnn_extractor = CNNExtract(in_channels=1, out_channels=feature_dim)

        self.lstm = nn.LSTM(input_size=feature_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size, seq_len, h, w = x.shape
        x = x.view(batch_size * seq_len, 1, h, w)

        feat = self.cnn_extractor(x)
        feat = feat.view(batch_size, seq_len, -1)

        out, (h_n, c_n) = self.lstm(feat)
        last_output = h_n[-1]
        output = self.fc(last_output)
        return output


class NQEModel(nn.Module):
    def __init__(self, gate_list):
        super().__init__()

        @qml.qnode(dev, interface='torch')
        def circuit(inputs):
            quantum_embedding(inputs[0:4], gate_list)
            qml.adjoint(quantum_embedding)(inputs[4:8], gate_list)

            return qml.probs(wires=range(4))

        self.qlayer1 = qml.qnn.TorchLayer(circuit, weight_shapes={})
        self.linear_relu_stack1 = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )

    def forward(self, x1, x2):
        x1 = self.linear_relu_stack1(x1)
        x2 = self.linear_relu_stack1(x2)
        x = torch.concat([x1, x2], 1)
        x = self.qlayer1(x)
        return x[:, 0]


class ValueNetwork(nn.Module):
    """
    Optional for REINFORCE w Baseline
    """
    def __init__(self, feature_dim=16, hidden_dim=32):
        super(ValueNetwork, self).__init__()

        # 여기서는 간단히 CNNExtract -> LSTM -> FC(1)
        self.cnn_extractor = CNNExtract(in_channels=1, out_channels=feature_dim)

        self.lstm = nn.LSTM(input_size=feature_dim,
                            hidden_size=hidden_dim,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)  # scalar output

    def forward(self, x):
        batch_size, seq_len, h, w = x.shape
        x = x.view(batch_size * seq_len, 1, h, w)

        feat = self.cnn_extractor(x)
        feat = feat.view(batch_size, seq_len, -1)

        out, (h_n, c_n) = self.lstm(feat)
        last_output = h_n[-1]
        value = self.fc(last_output)  # shape: [batch_size, 1]
        return value

