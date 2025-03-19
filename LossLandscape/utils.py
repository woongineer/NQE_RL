import itertools
import random

import pennylane as qml
import torch
import pandas as pd
from torch import nn
from data import new_data
import matplotlib.pyplot as plt
import numpy as np

num_qubit = 4
dev = qml.device("default.qubit", wires=num_qubit)

def check_lazy_regime_from_params(initial_params, trained_params, threshold=0.1):
    param_diff_norm = (trained_params - initial_params).norm().item()
    initial_norm = initial_params.norm().item()
    change_ratio = param_diff_norm / (initial_norm + 1e-12)
    is_lazy = change_ratio < threshold
    return is_lazy, change_ratio


def compute_QNTK(nn_model, random_circuit, input_data):
    """
    QNTK를 (다차원 출력) 샘플별로 자코비안을 구해,
    각 출력 차원마다 gradient를 누적하는 방식으로 계산합니다.
    """
    nn_model.eval()

    grad_vecs = []  # 자코비안(gradient) 벡터들을 담을 리스트

    # input_data: [batch_size, feature_dim]
    for i in range(input_data.size(0)):
        x_single = input_data[i : i + 1]  # (1, feature_dim)

        # forward
        output = nn_model(x_single)
        # 예: output.shape == [1, 4]  (배치=1, 출력차원=4)

        # (1,4)를 (4,)로 바꿔주기
        output = output.squeeze(0)
        # 이제 output.shape == [4]

        # 출력 차원별로 backward를 따로 호출 -> 자코비안 획득
        for dim_idx in range(output.size(0)):
            scalar_out = output[dim_idx]      # 이 텐서는 스칼라 크기([1])가 됨
            nn_model.zero_grad()             # 이전 gradient 초기화

            # 마지막 차원만 retain_graph=False
            #   (그 외에는 그래프를 유지해야 다음 backward 가능)
            retain = True if dim_idx < (output.size(0) - 1) else False
            scalar_out.backward(retain_graph=retain)

            # 현재 dim_idx 차원에 대한 gradient를 모아서 1차원 벡터로 만듦
            grad_params = []
            for param in nn_model.parameters():
                if param.grad is not None:
                    grad_params.append(param.grad.view(-1))
                else:
                    grad_params.append(torch.zeros(0))

            grad_vec = torch.cat(grad_params)
            grad_vecs.append(grad_vec.detach().clone())
            # 이렇게 하면 (batch_size * output_dim) 번의 backward 호출이 이뤄짐

    # grad_vecs는 총 (batch_size * 출력차원) 개의 벡터
    # 각 벡터 크기는 [total_params]
    grad_matrix = torch.stack(grad_vecs)  # shape: [(batch_size * output_dim), total_params]

    # NTK 계산: (grad_matrix) × (grad_matrix)^T
    QNTK_matrix = grad_matrix @ grad_matrix.T

    # 이제 QNTK_matrix는 [(batch_size*output_dim) × (batch_size*output_dim)] 크기
    # 고유값 분석
    eigenvalues = torch.linalg.eigvalsh(QNTK_matrix).real
    eig_max = eigenvalues.max().item()

    # 작은 고윳값 필터링
    filtered_eig = eigenvalues[eigenvalues > 1e-10]
    if filtered_eig.numel() == 0:
        eig_min = 1e-12
    else:
        eig_min = filtered_eig.min().item()

    condition_number = eig_max / eig_min
    eig_entropy = -torch.sum(eigenvalues * torch.log(eigenvalues + 1e-10)).item()

    return {
        "qntk_matrix": QNTK_matrix.detach().cpu(),
        "eig_max": eig_max,
        "eig_min": eig_min,
        "condition_number": condition_number,
        "eig_entropy": eig_entropy,
        "eigenvalues": eigenvalues.detach().cpu().numpy()
    }



def compute_local_ED(fisher_matrix, num_data, gamma=0.5):
    d = fisher_matrix.shape[0]
    kappa = gamma * num_data / (2 * np.pi * np.log(num_data))
    local_ed_matrix = torch.eye(d) + kappa * fisher_matrix
    eigenvalues = torch.linalg.eigvalsh(local_ed_matrix).real

    log_det = torch.sum(torch.log(eigenvalues + 1e-10)).item()
    local_ed = (2 / np.log(kappa)) * log_det
    print('dadasd')

    return {
        "local_ed": local_ed,
        "eigenvalues": eigenvalues.detach().cpu().numpy()
    }


def plot_eigen_spectrum(eigenvalues, title='Eigenvalue Spectrum'):
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(eigenvalues_sorted)), eigenvalues_sorted)
    plt.xlabel('Eigenvalue Index (sorted)')
    plt.ylabel('Eigenvalue Magnitude')
    plt.title(title)
    plt.yscale('log')
    plt.grid(True)
    plt.show()


def quantum_embedding_NQE(x, gate_structure):
    for gate, control, target, feature_dim in gate_structure:
        if gate == 'RX':
            qml.RX(x[feature_dim], wires=control)
        elif gate == 'RY':
            qml.RY(x[feature_dim], wires=control)
        elif gate == 'RZ':
            qml.RZ(x[feature_dim], wires=control)
        elif gate == 'CNOT':
            qml.CNOT(wires=[control, target])
        elif gate == 'H':
            qml.Hadamard(wires=control)
        elif gate == 'RX_arctan':
            qml.RX(torch.arctan(x[feature_dim]), wires=control)
        elif gate == 'RY_arctan':
            qml.RY(torch.arctan(x[feature_dim]), wires=control)
        elif gate == 'RZ_arctan':
            qml.RZ(torch.arctan(x[feature_dim]), wires=control)


class NQEModel(nn.Module):
    def __init__(self, dev, gate_structure):
        super().__init__()

        @qml.qnode(dev, interface='torch')
        def circuit(inputs):
            quantum_embedding_NQE(inputs[0:4], gate_structure)
            qml.adjoint(quantum_embedding_NQE)(inputs[4:8], gate_structure)

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


def NQE(circuit, batch_size_for_NQE, iter_for_NQE, X_train, X_test, Y_train, Y_test):
    NQE_model = NQEModel(dev, circuit)
    NQE_model.train()
    NQE_opt = torch.optim.SGD(NQE_model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    initial_params = torch.nn.utils.parameters_to_vector(NQE_model.parameters()).detach().clone()

    for nqe_epoch in range(iter_for_NQE):
        nqe_X1_batch, nqe_X2_batch, nqe_Y_batch = new_data(batch_size_for_NQE, X_train, Y_train)
        pred = NQE_model(nqe_X1_batch, nqe_X2_batch)
        loss = loss_fn(pred, nqe_Y_batch)
        NQE_opt.zero_grad()
        loss.backward()
        NQE_opt.step()

    valid_loss_list = []
    NQE_model.eval()
    for _ in range(10):
        nqe_X1_batch, nqe_X2_batch, nqe_Y_batch = new_data(batch_size_for_NQE, X_test, Y_test)
        with torch.no_grad():
            pred = NQE_model(nqe_X1_batch, nqe_X2_batch)
        valid_loss_list.append(loss_fn(pred, nqe_Y_batch).item())

    trained_params = torch.nn.utils.parameters_to_vector(NQE_model.parameters()).detach().clone()

    return valid_loss_list, initial_params, trained_params


def rerun_good_bad(landscape_resuls_list, good_idx, bad_idx,
                   batch_size_for_NQE, iter_for_NQE, X_train, X_test, Y_train, Y_test, num_of_trial):
    # GOOD Part
    print('GOOD start...')
    gidx_NQE = {}
    for gidx in good_idx:
        print(f'G {gidx}')
        random_circuit = landscape_resuls_list[gidx]['random_circuit']
        NQE_results_g = [NQE(random_circuit, batch_size_for_NQE, iter_for_NQE, X_train, X_test, Y_train, Y_test) for _
                         in
                         range(num_of_trial)]
        gidx_NQE[f'{gidx}'] = NQE_results_g

    # BAD Part
    print('BAD start...')
    bidx_NQE = {}
    for bidx in bad_idx:
        print(f'B {bidx}')
        random_circuit = landscape_resuls_list[bidx]['random_circuit']
        NQE_results_b = [NQE(random_circuit, batch_size_for_NQE, iter_for_NQE, X_train, X_test, Y_train, Y_test) for _
                         in
                         range(num_of_trial)]
        bidx_NQE[f'{bidx}'] = NQE_results_b

    return gidx_NQE, bidx_NQE


def gate_checker(random_circuit):
    required_gates = {"RX", "RY", "RZ", "RX_arctan", "RY_arctan", "RZ_arctan"}
    used_gates = {gate[0] for gate in random_circuit}
    result = any(gate in used_gates for gate in required_gates)
    return result


def generate_random_circuit(num_qubit, num_gates_range, gate_set):
    qubit_indices = list(itertools.permutations(range(num_qubit), 2))
    num_gates = random.randrange(num_gates_range[0], num_gates_range[1])

    while True:
        structure = []
        for _ in range(num_gates):
            gate = random.choice(gate_set)
            control, target = random.choice(qubit_indices)
            feature_dim = random.choice(range(num_qubit))
            structure.append([gate, control, target, feature_dim])

        if gate_checker(structure):
            return structure


class NN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.NN = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )

    def forward(self, x):
        return self.NN(x)


def compute_hessian(nn_model, random_circuit, x1_data, x2_data, y_data, loss_fn):
    """
    현재 NN 파라미터에서 loss의 Hessian을 계산 (2차 미분).
    """
    # baseline params
    baseline = torch.nn.utils.parameters_to_vector(nn_model.parameters())

    # loss_wrapper에서 create_graph=True를 해야 2차 미분 계산 가능
    def loss_wrapper(params):
        torch.nn.utils.vector_to_parameters(params, nn_model.parameters())
        output_loss = loss_fn(random_circuit, nn_model(x1_data), nn_model(x2_data), y_data)
        return output_loss

    # hessian 계산 함수
    # 기존의 torch.autograd.functional.hessian을 써도 되지만, 사용자님 코드 스타일 유지위해 아래처럼 구현
    # 1) 먼저 1차 미분(gradient) 구하고
    grad1 = torch.autograd.grad(loss_wrapper(baseline), (nn_model.parameters()), create_graph=True)
    # grad1은 tuple(param.shape), 각 param마다 gradient가 있음

    # 2) 각 gradient 성분에 대해 다시 미분
    #    여기서는 grad1에 대한 파라미터별 루프를 돌면서 full Hessian을 모은다.
    #    (메모리 많이 쓰므로, 실제 large scale에서는 비효율적일 수 있음)
    grad1_flat = torch.cat([g.reshape(-1) for g in grad1 if g is not None])
    n_params = grad1_flat.numel()
    # Hessian shape: [n_params, n_params]
    hessian = torch.zeros(n_params, n_params, dtype=grad1_flat.dtype, device=grad1_flat.device)

    for i, g_el in enumerate(grad1_flat):
        # retain_graph=True로 그래프 보존
        grad2 = torch.autograd.grad(g_el, nn_model.parameters(), retain_graph=True)
        grad2_flat = torch.cat(
            [g2.reshape(-1) if g2 is not None else torch.zeros(1, device=grad1_flat.device) for g2 in grad2])
        hessian[i, :] = grad2_flat

    return hessian


def compute_fisher_information(nn_model, random_circuit, x1_data, x2_data, y_data, loss_fn):
    """
    배치 내 각 샘플에 대해 gradient outer product를 계산하여 평균을 냄 -> Fisher Information
    (full matrix 버전)
    """
    # x1_data, x2_data, y_data가 [batch, ...]라 가정
    n_samples = x1_data.shape[0]
    # total param 개수
    baseline = torch.nn.utils.parameters_to_vector(nn_model.parameters())
    total_params = baseline.numel()

    fisher_matrix = torch.zeros(total_params, total_params, device=baseline.device)

    for i in range(n_samples):
        # 샘플별 데이터
        xi1 = x1_data[i:i + 1]
        xi2 = x2_data[i:i + 1]
        yi = y_data[i:i + 1]

        nn_model.zero_grad()
        loss_val = loss_fn(random_circuit, nn_model(xi1), nn_model(xi2), yi)
        loss_val.backward(retain_graph=False)

        grad_vec = []
        for p in nn_model.parameters():
            if p.grad is not None:
                grad_vec.append(p.grad.view(-1))
            else:
                grad_vec.append(torch.zeros(0, device=baseline.device))
        grad_vec = torch.cat(grad_vec)

        # outer product
        fisher_matrix += torch.ger(grad_vec, grad_vec)  # ger -> outer product

    # 평균
    fisher_matrix /= n_samples
    return fisher_matrix.detach()


def compute_flatness(hessian_matrix):
    """
    Hessian의 고유값을 이용하여 trace, 최대 고유값, 평균 고유값(평탄도/Sharpness)를 반환합니다.
    """
    eigenvalues = torch.linalg.eigvals(hessian_matrix).real
    trace = eigenvalues.sum().item()
    max_eigenvalue = eigenvalues.max().item()
    average_eigenvalue = eigenvalues.mean().item()
    return {"trace": trace, "max_eigenvalue": max_eigenvalue, "average_eigenvalue": average_eigenvalue}


def compute_condition_number(hessian_matrix):
    """
    Hessian Matrix의 조건수 (largest/smallest eigenvalue)를 계산합니다.
    """
    eigenvalues = torch.linalg.eigvals(hessian_matrix).real
    min_eig = eigenvalues.abs().min().item()
    max_eig = eigenvalues.abs().max().item()
    condition_number = max_eig / (min_eig + 1e-12)
    return condition_number


def compute_local_lipschitz(hessian_matrix):
    """
    Local Lipschitz constant를 Hessian의 최대 고유값(abs)로 근사합니다.
    """
    eigenvalues = torch.linalg.eigvals(hessian_matrix).real
    L = eigenvalues.abs().max().item()
    return L


def sample_gradient_norm_distribution(nn_model, random_circuit, x1_data, x2_data, y_data, loss_fn, num_samples=50,
                                      perturb_scale=0.1):
    """
    여러 샘플에서 gradient norm의 분포를 측정합니다.
    """
    baseline = torch.nn.utils.parameters_to_vector(nn_model.parameters()).detach().clone()
    grad_norms = []
    for _ in range(num_samples):
        perturbation = torch.randn_like(baseline) * perturb_scale
        new_params = baseline + perturbation
        torch.nn.utils.vector_to_parameters(new_params, nn_model.parameters())

        nn_model.zero_grad()
        loss_val = loss_fn(random_circuit, nn_model(x1_data), nn_model(x2_data), y_data)
        loss_val.backward(retain_graph=False)

        grad_vector = []
        for p in nn_model.parameters():
            if p.grad is not None:
                grad_vector.append(p.grad.view(-1))
        if len(grad_vector) == 0:
            grad_norms.append(0.0)
        else:
            grad_vec = torch.cat(grad_vector)
            grad_norms.append(grad_vec.norm().item())

    # 원래 파라미터로 복원
    torch.nn.utils.vector_to_parameters(baseline, nn_model.parameters())
    return grad_norms


def loss_function(circuit_structure, x1_data, x2_data, y_data):
    dev = qml.device("default.qubit", wires=num_qubit)

    def quantum_embedding(x_data):
        for gate, control, target, feature_dim in circuit_structure:
            if gate == 'RX':
                qml.RX(x_data[feature_dim], wires=control)
            elif gate == 'RY':
                qml.RY(x_data[feature_dim], wires=control)
            elif gate == 'RZ':
                qml.RZ(x_data[feature_dim], wires=control)
            elif gate == 'CNOT':
                qml.CNOT(wires=[control, target])
            elif gate == 'H':
                qml.Hadamard(wires=control)
            elif gate == 'RX_arctan':
                qml.RX(torch.arctan(x_data[feature_dim]), wires=control)
            elif gate == 'RY_arctan':
                qml.RY(torch.arctan(x_data[feature_dim]), wires=control)
            elif gate == 'RZ_arctan':
                qml.RZ(torch.arctan(x_data[feature_dim]), wires=control)

    @qml.qnode(dev, interface='torch')
    def circuit(inputs):
        quantum_embedding(inputs[0:4])
        qml.adjoint(quantum_embedding)(inputs[4:8])
        return qml.probs(wires=range(4))

    loss_fn = torch.nn.MSELoss()
    qlayer1 = qml.qnn.TorchLayer(circuit, weight_shapes={})
    x = torch.concat([x1_data, x2_data], 1)
    x = qlayer1(x)
    pred = x[:, 0]
    loss = loss_fn(pred, y_data)
    return loss


def get_good_and_bad(landscape_resuls_list, good_bad_N):
    import pandas as pd
    if len(landscape_resuls_list) < (good_bad_N * 2):
        print(f'good_bad_N:{good_bad_N} is too large for {landscape_resuls_list} circuit, changed.')
        good_bad_N = len(landscape_resuls_list) // 4

    result_simp = {}
    for i in landscape_resuls_list:
        result_simp[i] = []
        for j in landscape_resuls_list[i]['NQE_results']:
            for k in j:
                result_simp[i].append(k)

    data = []
    for i, vals in result_simp.items():
        for v in vals:
            data.append([i, v])
    df = pd.DataFrame(data, columns=['landscape', 'score'])

    filtered_data = []
    for i, group in df.groupby('landscape'):
        median = group['score'].median()
        lower_bound = median * 0.9
        upper_bound = median * 1.1
        filtered_group = group[(group['score'] >= lower_bound) & (group['score'] <= upper_bound)]
        filtered_data.append(filtered_group)

    df_filtered = pd.concat(filtered_data)
    group_stats = df_filtered.groupby('landscape')['score'].agg(['mean', 'std'])

    good_idx = group_stats['mean'].nsmallest(good_bad_N).index
    bad_idx = group_stats['mean'].nlargest(good_bad_N).index

    good_means = group_stats.loc[good_idx, 'mean'].tolist()
    bad_means = group_stats.loc[bad_idx, 'mean'].tolist()

    return good_idx, bad_idx, good_means, bad_means
