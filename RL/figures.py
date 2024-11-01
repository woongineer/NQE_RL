import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as np

from embedding import quantum_embedding_rl

dev = qml.device('default.qubit', wires=4)

plot_path = "/Users/jwheo/Desktop/Y/NQE/Neural-Quantum-Embedding/RL/result_plot"

def plot_nqe_loss(RL_model, NQE_losses, iter):
    plt.figure()
    plt.plot(NQE_losses, label='NQE Loss')
    step = max(1, len(NQE_losses) // 10)
    plt.xticks(range(0, len(NQE_losses), step))
    plt.title(f'NQE Loop {iter}')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{plot_path}/{RL_model}_NQE_{iter}th.png')


def plot_policy_loss(RL_model, policy_losses, iter):
    plt.figure()
    if RL_model == 'DQN':
        plt.plot(policy_losses, color='orange', label='Total Reward')
        step = max(1, len(policy_losses) // 10)
        plt.xticks(range(0, len(policy_losses), step))
        plt.title(f'Policy Loop {iter}')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
    else:
        plt.plot(policy_losses, color='orange', label='Policy Loss')
        step = max(1, len(policy_losses) // 10)
        plt.xticks(range(0, len(policy_losses), step))
        plt.title(f'Policy Loop {iter}')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{plot_path}/{RL_model}_Policy_{iter}th.png')


def draw_circuit(RL_model, action_seq, iter):
    @qml.qnode(dev)
    def fig_circ(action_seq):
        quantum_embedding_rl(np.array([1, 1, 1, 1]), action_seq)

        return qml.probs(wires=range(4))

    if action_seq is not None:
        fig, ax = qml.draw_mpl(fig_circ)(action_seq)

        action_text = "\n".join(
            [str(action_seq[i:i + 5]) for i in
             range(0, len(action_seq), 5)]
        )
        fig.text(0.1, -0.1, f'Action Sequence: {action_text}', fontsize=8,
                 wrap=True)

        fig.savefig(
            f'{plot_path}/{RL_model}_circuit_{iter}th.png', bbox_inches='tight')


def plot_comparison(RL_model, loss_none, loss_NQE, loss_NQE_RL,
                    accuracy_none, accuracy_NQE, accuracy_NQE_RL):
    plt.figure()
    plt.plot(loss_none, label=f'None {accuracy_none:.3f}', color='blue')
    plt.plot(loss_NQE, label=f'NQE {accuracy_NQE:.3f}', color='green')
    plt.plot(loss_NQE_RL, label=f'NQE & RL_legacy {accuracy_NQE_RL:.3f}',
             color='red')
    step = max(1, len(loss_none) // 10)
    plt.xticks(range(0, len(loss_none), step))
    plt.title('QCNN')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{plot_path}/{RL_model}_QCNN.png')
