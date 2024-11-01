import pennylane as qml
import torch
from torch import nn

from data import new_data
from embedding import quantum_embedding_rl, quantum_embedding_zz

dev = qml.device('default.qubit', wires=4)


class NQEModel(torch.nn.Module):
    def __init__(self, action_seq=None):
        super().__init__()
        self.action_seq = action_seq
        if action_seq is None:
            # Use quantum_embedding_zz
            @qml.qnode(dev, interface="torch")
            def circuit(inputs):
                quantum_embedding_zz(inputs[0:4])
                qml.adjoint(quantum_embedding_zz)(inputs[4:8])
                return qml.probs(wires=range(4))
        else:
            # Use quantum_embedding_rl
            @qml.qnode(dev, interface="torch")
            def circuit(inputs):
                quantum_embedding_rl(inputs[0:4], self.action_seq)
                qml.adjoint(quantum_embedding_rl)(inputs[4:8],
                                                  self.action_seq)
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


# Function to train NQE
def train_NQE(X_train, Y_train, NQE_iter, batch_sz,
              action_seq=None):
    NQE_model = NQEModel(action_seq)
    NQE_model.train()
    NQE_loss_fn = torch.nn.MSELoss()
    NQE_opt = torch.optim.SGD(NQE_model.parameters(), lr=0.01)
    NQE_losses = []
    for it in range(NQE_iter):
        X1_batch, X2_batch, Y_batch = new_data(batch_sz, X_train, Y_train)
        pred = NQE_model(X1_batch, X2_batch)
        loss = NQE_loss_fn(pred, Y_batch)

        NQE_opt.zero_grad()
        loss.backward()
        NQE_opt.step()

        if it % 3 == 0:
            print(f"Iterations: {it} Loss: {loss.item()}")
        NQE_losses.append(loss.item())
    return NQE_model, NQE_losses


def transform_data(NQE_model, X_data):
    NQE_model.eval()
    transformed_data = []
    with torch.no_grad():
        for x in X_data:
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            x_transformed = NQE_model.linear_relu_stack1(x_tensor)
            x_transformed = x_transformed.squeeze(0).detach().numpy()
            transformed_data.append(x_transformed)
    return transformed_data
