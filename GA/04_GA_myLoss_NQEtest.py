import random

import numpy as np
import pennylane as qml
import torch

from data import data_load_and_process as dataprep
from data import new_data
from utils_for_analysis import draw_GA_training, draw_GA_evaluation, draw_GA_fitness_distribution

dev = qml.device("default.qubit", wires=4)


def generate_random_structure(num_qubits, num_gates, feature_dim):
    structure = []
    for _ in range(num_gates):
        gate = random.choice(["RX", "RY", "RZ", "CNOT"])
        qubits = random.sample(range(num_qubits), 2 if gate == "CNOT" else 1)
        feature_idx = random.randint(0, feature_dim - 1)
        structure.append((gate, qubits, feature_idx))
    return structure


def fitness_function(structure, batch_size, X_batch, y_batch):
    def quantum_embedding(x):
        for gate, qubit_idx, data_index in structure:
            if gate == 'RX':
                qml.RX(x[data_index], wires=qubit_idx)
            elif gate == 'RY':
                qml.RY(x[data_index], wires=qubit_idx)
            elif gate == 'RZ':
                qml.RZ(x[data_index], wires=qubit_idx)
            elif gate == 'CNOT':
                qml.CNOT(wires=[qubit_idx[0], qubit_idx[1]])

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
    num_gates = 16
    num_qubits = 4
    batch_size = 100
    feature_dimension = X_train.shape[1]

    population = [generate_random_structure(num_qubits, num_gates, feature_dimension) for _ in range(population_size)]
    training_loss = []
    evaluation_loss = []
    fitness_history = []

    for_NQE_test = {}
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
            crossover_point = random.randint(1, num_gates - 1)
            child = parent1[:crossover_point] + parent2[crossover_point:]
            offspring.append(child)

        ## mutation
        for child in offspring:
            if random.random() < 0.1:  # Mutation probability
                gate_idx = random.randint(0, num_gates - 1)
                gate = random.choice(["RX", "RY", "RZ", "CNOT"])
                if gate == "CNOT":
                    qubits = random.sample(range(num_qubits), 2)
                else:
                    qubits = [random.choice(range(num_qubits))]
                feature_idx = random.randint(0, feature_dimension - 1)
                child[gate_idx] = (gate, qubits, feature_idx)

        ## update population
        population += offspring

        if generation + 1 == 2:
            for_NQE_test = {generation + 1: population}
        if generation + 1 == 20:
            for_NQE_test = {generation + 1: population}
        if generation + 1 == 200:
            for_NQE_test = {generation + 1: population}


    draw_GA_training(training_loss, 'training_NQEtest.png')
    draw_GA_evaluation(evaluation_loss, 'eval_NQEtest.png')
    draw_GA_fitness_distribution(fitness_history, 'dist_NQEtest.png')