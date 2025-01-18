from data import data_load_and_process as dataprep
from data import new_data
import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
import random
from utils import generate_layers
import torch
import seaborn as sns

dev = qml.device("default.qubit", wires=4)


def fitness_function(structure, batch_size, X_batch, y_batch):
    structure = [item for i in structure for item in layer_set[int(i)]]
    def quantum_embedding(x):
        for gate, idx in structure:
            if gate == 'RX':
                qml.RX(x[idx], wires=idx)
            elif gate == 'RY':
                qml.RY(x[idx], wires=idx)
            elif gate == 'RZ':
                qml.RZ(x[idx], wires=idx)
            elif gate == 'CNOT':
                qml.CNOT(wires=[idx[0], idx[1]])

    @qml.qnode(dev, interface='torch')
    def circuit(inputs):
        quantum_embedding(inputs[0:4])
        qml.adjoint(quantum_embedding)(inputs[4:8])

        return qml.probs(wires=range(4))

    loss_fn = torch.nn.MSELoss()
    X1_batch, X2_batch, Y_batch = new_data(batch_size, X_batch, y_batch)
    qlayer1 = qml.qnn.TorchLayer(circuit, weight_shapes={})
    x = torch.concat([X1_batch, X2_batch], 1)
    x = qlayer1(x)
    pred = x[:, 0]
    loss = loss_fn(pred, Y_batch)
    return loss


if __name__ == "__main__":
    num_qubit = 4
    X_train, X_test, y_train, y_test = dataprep(dataset='kmnist', reduction_sz=num_qubit)

    X_train = [np.array(item, dtype=np.float32) for item in X_train]
    X_train = np.vstack(X_train)

    X_test = [np.array(item, dtype=np.float32) for item in X_test]
    X_test = np.vstack(X_test)

    population_size = 64
    num_generations = 1000
    num_layer_kind = 100
    num_layer_list = 16
    num_qubits = 4
    batch_size = 100
    feature_dimension = X_train.shape[1]

    layer_set = generate_layers(num_qubit, num_layer_kind)

    population = [
        [torch.tensor(torch.randint(0, num_layer_kind, (1,)).item()) for _ in range(num_layer_list)]
        for _ in range(population_size)]
    training_loss = []
    evaluation_loss = []
    fitness_history = []

    for generation in range(num_generations):
        ## fitness evaluation
        fitness = [fitness_function(structure, batch_size, X_train, y_train) for structure in population]
        mean_fitness = np.mean(fitness)
        min_fitness = np.min(fitness)
        print(f"Generation {generation + 1}: Mean Fitness = {mean_fitness:.4f}, Min Fitness = {min_fitness:.4f}")
        training_loss.append([mean_fitness, min_fitness])

        ## criteria for early stopping
        if min_fitness < 0.01:
            print("Solution found!")
            break
        ## early stopping for overfitting
        if generation + 1 > 20:
            early_stop = np.sum(np.array(training_loss)[-20:, 1] == np.array(training_loss)[-1, 1])
            if early_stop == 20:
                print("Early stopping")
                break

        ## evaluation
        ev_fitness = [fitness_function(structure, batch_size, X_test, y_test) for structure in population]
        evaluation_loss.append([np.mean(ev_fitness), np.min(ev_fitness)])

        if (generation == 0) or ((generation + 1) % 10 == 0):
            fitness_history.append(fitness)

        ## training is over
        if generation + 1 == num_generations:
            print("Generation is over")
            break
        if mean_fitness == min_fitness:
            break

        ## selection
        sorted_population = [x for _, x in sorted(zip(fitness, population), key=lambda pair: pair[0], reverse=False)]
        population = sorted_population[:population_size // 2]

        ## crossover
        offspring = []
        for _ in range(population_size // 2):
            parent1, parent2 = random.sample(population, 2)
            crossover_point = random.randint(1, num_layer_list - 1)
            child = parent1[:crossover_point] + parent2[crossover_point:]
            offspring.append(child)

        ## mutation
        for child in offspring:
            if random.random() < 0.1:  # Mutation probability
                gate_idx = random.randint(0, num_layer_list - 1)
                child[gate_idx] = torch.tensor(random.randint(0, num_layer_kind - 1))

        ## update population
        population += offspring

    plt.title('Training min and mean Loss')
    plt.plot(np.array(training_loss)[:, 0])
    plt.plot(np.array(training_loss)[:, 1], "--")
    plt.legend(['Population mean loss', 'Population min loss'])
    plt.show()

    plt.title('Evaluation min and mean Loss')
    plt.plot(np.array(evaluation_loss)[:, 0])
    plt.plot(np.array(evaluation_loss)[:, 1], "--")
    plt.legend(['Evaluation mean loss', 'Evaluation min loss'])
    plt.show()

    plt.figure(figsize=(10, 6))

    # 색상 팔레트 생성
    colors = plt.cm.rainbow(np.linspace(0, 1, len(fitness_history)))

    for i, (generation_fitness, color) in enumerate(zip(fitness_history, colors)):
        # tensor를 numpy array로 변환
        generation_fitness_np = [tensor.detach().cpu().item() for tensor in generation_fitness]

        if i == 0:
            sns.kdeplot(x=generation_fitness_np, bw_adjust=0.7, fill=True, label='first generation', color=color,
                        alpha=0.5)
        elif i == len(fitness_history) - 1:
            sns.kdeplot(x=generation_fitness_np, bw_adjust=0.7, fill=True, label='last generation', color=color,
                        alpha=0.5)
        else:
            sns.kdeplot(x=generation_fitness_np, bw_adjust=0.7, fill=True, label=f'{i * 10}th generation',
                        color=color, alpha=0.5)

    plt.title('Generation-wise fidelity loss distribution')
    plt.xlabel('loss value')
    plt.ylabel('Frequency')
    plt.ylim(0, 100)
    plt.legend()
    plt.show()