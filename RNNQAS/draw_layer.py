from utils import generate_layers, quantum_embedding
import numpy as np
import pennylane as qml

dev = qml.device('default.qubit', wires=4)

@qml.qnode(dev)
def fig_circ(action_sequence):
    quantum_embedding(np.array([1, 1, 1, 1]), action_sequence)
    return qml.probs(wires=range(4))

if __name__ == "__main__":
    layer_set = generate_layers(4, 30)

    for i in range(10):
        fig, ax = qml.draw_mpl(fig_circ)(layer_set[i])

        fig.savefig(f'layer_{i}.png')

    print('dd')