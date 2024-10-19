import pennylane as qml
import torch
from pennylane import numpy as np

from embedding import quantum_embedding_zz, quantum_embedding_rl

dev = qml.device('default.qubit', wires=4)


def circuit_training(QCNN_learning_rate, QCNN_steps, batch_size, X_train,
                     Y_train, scheme, NQE_model=None, action_sequence=None):
    weights = np.random.random(30, requires_grad=True)
    opt = qml.NesterovMomentumOptimizer(stepsize=QCNN_learning_rate)
    loss_history = []
    for it in range(QCNN_steps):
        batch_index = np.random.randint(0, len(X_train), (batch_size,))
        X_batch = [X_train[i] for i in batch_index]
        Y_batch = [Y_train[i] for i in batch_index]
        weights, cost_new = opt.step_and_cost(
            lambda v: cost(v, X_batch, Y_batch, scheme, NQE_model,
                           action_sequence),
            weights)
        loss_history.append(cost_new)
        if it % 3 == 0:
            print("iteration: ", it, " cost: ", cost_new)
    return loss_history, weights


def Linear_Loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss += 0.5 * (1 - l * p)
    return loss / len(labels)


def cost(weights, X_batch, Y_batch, scheme, NQE_model=None,
         action_sequence=None):
    preds = []
    for x in X_batch:
        if scheme == 'NQE_RL':
            pred = QCNN_classifier(weights, x, scheme, NQE_model,
                                   action_sequence)
        elif scheme == 'NQE':
            pred = QCNN_classifier(weights, x, scheme, NQE_model)
        else:
            pred = QCNN_classifier(weights, x, scheme)
        preds.append(pred)
    return Linear_Loss(Y_batch, preds)


@qml.qnode(dev)
def QCNN_classifier(params, x, scheme, NQE_model=None, action_sequence=None):
    if scheme == 'NQE_RL':
        statepreparation(x, scheme, NQE_model, action_sequence)
    elif scheme == 'NQE':
        statepreparation(x, scheme, NQE_model)
    else:
        statepreparation(x, scheme)
    QCNN(params)
    return qml.expval(qml.PauliZ(2))


def statepreparation(x, scheme, NQE_model=None, action_sequence=None):
    if scheme is None:
        quantum_embedding_zz(x)
    elif scheme == 'NQE':
        x = NQE_model.linear_relu_stack1(torch.tensor(x, dtype=torch.float32))
        x = x.detach().numpy()
        quantum_embedding_zz(x)
    elif scheme == 'NQE_RL':
        x = NQE_model.linear_relu_stack1(torch.tensor(x, dtype=torch.float32))
        x = x.detach().numpy()
        quantum_embedding_rl(x, action_sequence)


def QCNN(params):
    param1 = params[0:15]
    param2 = params[15:30]

    U_SU4(param1, wires=[0, 1])
    U_SU4(param1, wires=[2, 3])
    U_SU4(param1, wires=[1, 2])
    U_SU4(param1, wires=[3, 0])
    U_SU4(param2, wires=[0, 2])


def U_SU4(params, wires):  # 15 params
    qml.U3(params[0], params[1], params[2], wires=wires[0])
    qml.U3(params[3], params[4], params[5], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[6], wires=wires[0])
    qml.RZ(params[7], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RY(params[8], wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.U3(params[9], params[10], params[11], wires=wires[0])
    qml.U3(params[12], params[13], params[14], wires=wires[1])


def accuracy_test(predictions, labels):
    acc = 0
    for l, p in zip(labels, predictions):
        if np.abs(l - p) < 1:
            acc = acc + 1
    return acc / len(labels)
