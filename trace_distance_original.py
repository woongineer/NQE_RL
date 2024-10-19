from pennylane import numpy as np
import torch
from torch import nn
import tensorflow as tf
from sklearn.decomposition import PCA
import pennylane as qml
import embedding


dev = qml.device('default.qubit', wires=4)


def data_load_and_process(dataset='mnist', reduction_size: int = 4):
    if dataset == 'mnist':
        (x_train, y_train), (
            x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif dataset == 'kmnist':
        # Path to training images and corresponding labels provided as numpy arrays
        kmnist_train_images_path = "/RL/kmnist/kmnist-train-imgs.npz"
        kmnist_train_labels_path = "/RL/kmnist/kmnist-train-labels.npz"

        # Path to the test images and corresponding labels
        kmnist_test_images_path = "/RL/kmnist/kmnist-test-imgs.npz"
        kmnist_test_labels_path = "/RL/kmnist/kmnist-test-labels.npz"

        x_train = np.load(kmnist_train_images_path)['arr_0']
        y_train = np.load(kmnist_train_labels_path)['arr_0']

        # Load the test data from the corresponding npz files
        x_test = np.load(kmnist_test_images_path)['arr_0']
        y_test = np.load(kmnist_test_labels_path)['arr_0']

    x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[
        ..., np.newaxis] / 255.0
    train_filter_tf = np.where((y_train == 0) | (y_train == 1))
    test_filter_tf = np.where((y_test == 0) | (y_test == 1))

    x_train, y_train = x_train[train_filter_tf], y_train[train_filter_tf]
    x_test, y_test = x_test[test_filter_tf], y_test[test_filter_tf]

    x_train = tf.image.resize(x_train[:], (256, 1)).numpy()
    x_test = tf.image.resize(x_test[:], (256, 1)).numpy()
    x_train, x_test = tf.squeeze(x_train).numpy(), tf.squeeze(x_test).numpy()

    X_train = PCA(reduction_size).fit_transform(x_train)
    X_test = PCA(reduction_size).fit_transform(x_test)
    x_train, x_test = [], []
    for x in X_train:
        x = (x - x.min()) * (np.pi / (x.max() - x.min()))
        x_train.append(x)
    for x in X_test:
        x = (x - x.min()) * (np.pi / (x.max() - x.min()))
        x_test.append(x)
    return x_train[:400], x_test[:100], y_train[:400], y_test[:100]

def new_data(batch_size, X, Y):
    X1_new, X2_new, Y_new = [], [], []
    for i in range(batch_size):
        n, m = np.random.randint(len(X)), np.random.randint(len(X))
        X1_new.append(X[n])
        X2_new.append(X[m])
        if Y[n] == Y[m]:
            Y_new.append(1)
        else:
            Y_new.append(0)
    return torch.tensor(X1_new).to(device), torch.tensor(X2_new).to(
        device), torch.tensor(Y_new).to(device)



@qml.qnode(dev, interface="torch")
def Four_circuit(inputs):
    embedding.Four_QuantumEmbedding1(inputs[0:4])
    embedding.Four_QuantumEmbedding1_inverse(inputs[4:8])
    return qml.probs(wires=range(4))


class Four_Model1_Fidelity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qlayer1 = qml.qnn.TorchLayer(Four_circuit, weight_shapes={})
        self.linear_relu_stack1 = nn.Sequential(
            nn.Linear(4, 6),
            nn.ReLU(),
            nn.Linear(6, 6),
            nn.ReLU(),
            nn.Linear(6, 4)
        )

    def forward(self, x1, x2):
        x1 = self.linear_relu_stack1(x1)
        x2 = self.linear_relu_stack1(x2)
        x = torch.concat([x1, x2], 1)
        x = self.qlayer1(x)
        return x[:, 0]


def train_models():
    train_loss = []
    model = Four_Model1_Fidelity().to(device)
    PATH = '/Users/tak/Github/QEmbedding/Results/QCNN_demonstration/Noiseless/Model 1/Four_Model1_Fidelity.pt'
    model.train()

    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    for it in range(1000):
        X1_batch, X2_batch, Y_batch = new_data(25, X_train, Y_train)
        X1_batch, X2_batch, Y_batch = X1_batch.to(device), X2_batch.to(
            device), Y_batch.to(device)

        pred = model(X1_batch, X2_batch)
        pred, Y_batch = pred.to(torch.float32), Y_batch.to(torch.float32)
        loss = loss_fn(pred, Y_batch)
        train_loss.append(loss.item())

        opt.zero_grad()
        loss.backward()
        opt.step()

        if it % 200 == 0:
            print(f"Iterations: {it} Loss: {loss.item()}")

    torch.save(model.state_dict(), PATH)

@qml.qnode(dev, interface="torch")
def Four_Distance(inputs):
    embedding.Four_QuantumEmbedding1(inputs[0:4])
    return qml.density_matrix(wires=range(4))


class Distances(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qlayer1_distance = qml.qnn.TorchLayer(Four_Distance,
                                                   weight_shapes={})
        self.linear_relu_stack1 = nn.Sequential(
            nn.Linear(4, 6),
            nn.ReLU(),
            nn.Linear(6, 6),
            nn.ReLU(),
            nn.Linear(6, 4)
        )

    def forward(self, x1, x0, Distance, Trained):
        if Trained:
            x1 = self.linear_relu_stack1(x1)
            x0 = self.linear_relu_stack1(x0)
        rhos1 = self.qlayer1_distance(x1)
        rhos0 = self.qlayer1_distance(x0)
        rho1 = torch.sum(rhos1, dim=0) / len(x1)
        rho0 = torch.sum(rhos0, dim=0) / len(x0)
        rho_diff = rho1 - rho0
        if Distance == 'Trace':
            eigvals = torch.linalg.eigvals(rho_diff)
            return 0.5 * torch.real(torch.sum(torch.abs(eigvals)))
        elif Distance == 'Hilbert-Schmidt':
            return 0.5 * torch.trace(rho_diff @ rho_diff)

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, X_test, Y_train, Y_test = data_load_and_process(dataset='kmnist',
                                                             reduction_size=4)


    N_valid, N_test = 500, 10000
    X1_new_valid, X2_new_valid, Y_new_valid = new_data(N_valid, X_test, Y_test)
    X1_new_test, X2_new_test, Y_new_test = new_data(N_test, X_test, Y_test)


    X1_test, X0_test = [], []
    for i in range(len(X_test)):
        if Y_test[i] == 1:
            X1_test.append(X_test[i])
        else:
            X0_test.append(X_test[i])
    X1_test, X0_test = torch.tensor(X1_test), torch.tensor(X0_test)

    X1_train, X0_train = [], []
    for i in range(len(X_train)):
        if Y_train[i] == 1:
            X1_train.append(X_train[i])
        else:
            X0_train.append(X_train[i])
    X1_train, X0_train = torch.tensor(X1_train), torch.tensor(X0_train)


    PATH_Model1_Fidelity = '/Users/tak/Github/QEmbedding/Results/QCNN_demonstration/Noiseless/Model 1/Four_Model1_Fidelity.pt'
    Model1_Fidelity_Distance = Distances().to(device)
    Model1_Fidelity_Distance.load_state_dict(
        torch.load(PATH_Model1_Fidelity, map_location=device))

    # Distances Before Training
    Trace_before_traindata = Model1_Fidelity_Distance(X1_train, X0_train, 'Trace',
                                                      False)
    Trace_before_testdata = Model1_Fidelity_Distance(X1_test, X0_test, 'Trace',
                                                     False)
    print(f"Trace Distance (Training Data) Before: {Trace_before_traindata}")
    print(f"Trace Distance (Test Data) Before: {Trace_before_testdata}")

    # Distances After training with Model1_Fidelity
    Trace_Fidelity_traindata = Model1_Fidelity_Distance(X1_train, X0_train, 'Trace',
                                                        True)
    Trace_Fidelity_testdata = Model1_Fidelity_Distance(X1_test, X0_test, 'Trace',
                                                       True)
    print(
        f"Trace Distance (Training Data) After Model1 Fidelity: {Trace_Fidelity_traindata}")
    print(
        f"Trace Distance (Test Data) After Model1 Fidelity: {Trace_Fidelity_testdata}")


    # Lower Bounds
    LB_before_traindata = 0.5 * (1 - Trace_before_traindata.detach().numpy())
    LB_Fidelity_traindata = 0.5 * (1 - Trace_Fidelity_traindata.detach().numpy())
