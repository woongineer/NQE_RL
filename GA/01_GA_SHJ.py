from datetime import datetime
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from data import data_load_and_process as dataprep
import pennylane as qml
import seaborn as sns
from utils import generate_layers


dev = qml.device("default.qubit", wires=4)


def quantum_circuit(structure, data):
    for gate, qubit_idx in structure:
        if gate == 'R_x':
            qml.RX(data[qubit_idx], wires=qubit_idx)
        elif gate == 'R_y':
            qml.RY(data[qubit_idx], wires=qubit_idx)
        elif gate == 'R_z':
            qml.RZ(data[qubit_idx], wires=qubit_idx)
        elif gate == 'CNOT':
            qml.CNOT(wires=[qubit_idx[0], qubit_idx[1]])

@qml.qnode(dev)
def embedding_circuit(structure, data):
    quantum_circuit(structure, data)
    return qml.state()


def fitness_function(structure, X_batch, y_batch):
    gate_list = [item for i in structure for item in layer_set[int(i)]]
    states = []
    for data in X_batch:
        embedding = embedding_circuit(gate_list, data)
        embedding_tensor = torch.tensor(embedding)
        states.append(embedding_tensor)
    states = torch.stack(states)
    states = states / torch.norm(states, dim=1, keepdim=True)
    states_conj = torch.conj(states)
    inner_products = torch.matmul(states_conj, states.T)
    fidelity_matrix = torch.abs(inner_products) ** 2
    labels = torch.tensor(y_batch).view(-1)
    label_products = torch.outer(labels, labels)
    loss_matrix = (fidelity_matrix - 0.5 * (1 + label_products)) ** 2
    batch_size = X_batch.shape[0]
    indices = torch.triu_indices(batch_size, batch_size, offset=1)
    loss_values = loss_matrix[indices[0], indices[1]]
    loss = torch.mean(loss_values)
    return loss


if __name__ == "__main__":
    print(datetime.now())
    # 파라미터
    num_qubit = 4

    X_train, X_test, y_train, y_test = dataprep(dataset='kmnist', reduction_sz=num_qubit)

    X_train = [np.array(item, dtype=np.float32) for item in X_train]
    X_train = np.vstack(X_train)

    X_test = [np.array(item, dtype=np.float32) for item in X_test]
    X_test = np.vstack(X_test)

    X_train, X_test, y_train, y_test = X_train[:100], X_test[:25], y_train[:100], y_test[:25]

    population_size = 64
    num_gen = 1000
    num_layer_kind = 100
    num_layer_list = 16
    num_qubit = 4
    feature_dim = X_train.shape[1]

    layer_set = generate_layers(num_qubit, num_layer_kind)

    population = [
        [torch.tensor(torch.randint(0, num_layer_kind, (1,)).item()) for _ in range(num_layer_list)]
        for _ in range(population_size)]
    training_loss = []
    evaluation_loss = []
    fitness_history = []

    for gen in range(num_gen):
        fitness = [fitness_function(structure, X_train, y_train) for structure in population]
        mean_fitness = np.mean(fitness)
        min_fitness = np.min(fitness)
        print(f"Generation {gen + 1}: Mean Fitness = {mean_fitness:.4f}, Min Fitness = {min_fitness:.4f}")
        training_loss.append([mean_fitness, min_fitness])

        ## criteria for early stopping
        if min_fitness < 0.01:
            print("Solution found!")
            break
        ## early stopping for overfitting
        if gen + 1 > 20:
            early_stop = np.sum(np.array(training_loss)[-20:, 1] == np.array(training_loss)[-1, 1])
            if early_stop == 20:
                print("Early stopping")
                break

        ## evaluation
        ev_fitness = [fitness_function(structure, X_test, y_test) for structure in population]
        evaluation_loss.append([np.mean(ev_fitness), np.min(ev_fitness)])

        if (gen == 0) or ((gen + 1) % 10 == 0):
            fitness_history.append(fitness)

        ## training is over
        if gen + 1 == num_gen:
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
                # gate = random.choice(["RX", "RY", "RZ", "CNOT"])
                # if gate == "CNOT":
                #     qubits = random.sample(range(num_qubit), 2)
                # else:
                #     qubits = [random.choice(range(num_qubit))]
                # feature_idx = random.randint(0, feature_dim - 1)
                # child[gate_idx] = (gate, qubits, feature_idx)
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
