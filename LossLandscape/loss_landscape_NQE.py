import pennylane as qml
import torch
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
import functools

num_qubit = 4
dev = qml.device("default.qubit", wires=num_qubit)


def generate_random_circuit(num_gates, gate_set):
    structure = []
    qubit_indices = list(itertools.permutations(range(num_qubit), 2))
    for _ in range(num_gates):
        gate = random.choice(gate_set)
        control, target = random.choice(qubit_indices)
        structure.append([gate, control, target])

    return structure


def loss_function(circuit_structure, params):
    x_data, y_data = params[:8], params[-1]

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

    qlayer1 = qml.qnn.TorchLayer(circuit, weight_shapes={})
    x = qlayer1(x_data)
    pred = x[0]
    loss = abs(y_data - pred)

    return loss


def generate_loss_landscape_data(random_circuit):
    losses = []
    for x1, x2, y in zip(x1_data, x2_data, y_data):
        params = torch.cat([x1, x2, y.unsqueeze(0)])
        loss = loss_function(random_circuit, params)
        losses.append([*params, loss])

    return losses


def evaluate_loss_landscape(random_circuit, samples, loss_fn):
    """
    샘플링 결과를 받아 최소 loss를 주는 파라미터에서
    - Hessian matrix (2차 미분 행렬)
    - Hessian의 고유값, 평균 고유값, 조건수
    - gradient로부터 구한 Fisher Information Matrix (FIM)
    - flatness (여기서는 Hessian의 trace)와 sharpness (최대 고유값)

    을 계산함.

    Args:
      samples: (theta1, theta2, theta3, theta4, loss) tuple 리스트.
      loss_fn: loss 함수 (파라미터 tensor를 받아 scalar 반환)

    Returns:
      metrics: 여러 평가 지표를 dict로 반환.
    """
    loss_fn_fixed = functools.partial(loss_fn, random_circuit)

    best_sample = min(samples, key=lambda s: s[-1])
    best_params = best_sample[:-1]

    # 최소 loss 파라미터에 대해 autograd를 사용하여 미분 계산
    params_tensor = torch.tensor(best_params, dtype=torch.float32, requires_grad=True)
    loss_val = loss_fn_fixed(params_tensor)

    # Hessian 계산: torch.autograd.functional.hessian 사용
    hessian = torch.autograd.functional.hessian(loss_fn_fixed, params_tensor)
    hessian_np = tuple(h.detach().numpy() for h in hessian)

    # Hessian의 고유값 계산 (대칭행렬이므로 eigvalsh 사용)
    eigenvalues = np.linalg.eigvalsh(hessian_np)
    avg_eigenvalue = np.mean(eigenvalues)
    # 최소 고유값이 0에 가까울 경우 조건수는 inf로 처리
    condition_number = np.abs(eigenvalues[-1] / eigenvalues[0]) if np.abs(eigenvalues[0]) > 1e-6 else np.inf

    # Fisher Information Matrix: gradient의 outer product (여기서는 단일 점에서의 값)
    loss_val.backward()  # gradient가 params_tensor.grad에 저장됨
    grad = params_tensor.grad.detach().numpy()
    fim = np.outer(grad, grad)

    # Flatness: Hessian의 trace, Sharpness: 최대 고유값
    flatness = np.trace(hessian_np)
    sharpness = eigenvalues[-1]

    metrics = {
        "best_params": best_params,
        "loss": loss_val.item(),
        "hessian": hessian_np,
        "eigenvalues": eigenvalues,
        "avg_eigenvalue": avg_eigenvalue,
        "condition_number": condition_number,
        "fim": fim,
        "flatness": flatness,
        "sharpness": sharpness
    }

    return metrics


def visualize_loss_landscape(loss_fn, best_params, random_circuit,
                             grid_range=(-1.0, 1.0), grid_points=50, direction1=None, direction2=None):
    """
    loss_fn: (random_circuit, params) 형태로 호출 가능한 loss 함수.
    best_params: 최적 파라미터 (numpy array 또는 torch.Tensor, shape: (n,))
    random_circuit: loss_fn에 필요한 회로 구조
    grid_range: 각 방향에서 이동할 범위 (예: (-1, 1))
    grid_points: grid의 해상도 (예: 50)
    direction1, direction2: 파라미터 공간에서의 이동 방향 (numpy array, shape: (n,)).
                           제공하지 않으면 랜덤으로 생성하며, 서로 직교하도록 정규화.
    """
    # best_params를 numpy array로 변환
    if isinstance(best_params, torch.Tensor):
        best_params = best_params.detach().numpy()
    n_params = len(best_params)

    # 방향이 주어지지 않으면 생성
    if direction1 is None:
        direction1 = np.random.randn(n_params)
        direction1 /= np.linalg.norm(direction1)
    if direction2 is None:
        direction2 = np.random.randn(n_params)
        # direction1과 직교하도록 정규화
        direction2 = direction2 - np.dot(direction2, direction1) * direction1
        direction2 /= np.linalg.norm(direction2)

    alphas = np.linspace(grid_range[0], grid_range[1], grid_points)
    betas = np.linspace(grid_range[0], grid_range[1], grid_points)
    loss_grid = np.zeros((grid_points, grid_points))

    # Grid 상의 각 점에서 loss 계산
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            # 새로운 파라미터: best_params에서 두 방향으로의 이동
            params_new = best_params + alpha * direction1 + beta * direction2
            params_tensor = torch.tensor(params_new, dtype=torch.float32, requires_grad=False)
            loss_val = loss_fn(random_circuit, params_tensor)
            loss_grid[i, j] = loss_val.item() if isinstance(loss_val, torch.Tensor) else loss_val

    # Contour plot
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    cp = ax[0].contourf(alphas, betas, loss_grid, levels=50, cmap='viridis')
    fig.colorbar(cp, ax=ax[0])
    ax[0].set_title("Loss Landscape Contour")
    ax[0].set_xlabel("Direction 1")
    ax[0].set_ylabel("Direction 2")

    # 3D Surface plot
    ax3d = fig.add_subplot(122, projection='3d')
    A, B = np.meshgrid(alphas, betas)
    surf = ax3d.plot_surface(A, B, loss_grid, cmap='viridis', edgecolor='none', alpha=0.9)
    ax3d.set_title("Loss Landscape Surface")
    ax3d.set_xlabel("Direction 1")
    ax3d.set_ylabel("Direction 2")
    ax3d.set_zlabel("Loss")
    fig.colorbar(surf, ax=ax3d, shrink=0.5, aspect=5)

    plt.tight_layout()
    plt.savefig('loss_landscape.png')
    plt.show()


def plot_hessian_eigenvalues(eigenvalues):
    """
    Hessian의 eigenvalue들을 바 플롯으로 시각화.

    eigenvalues: numpy array 형태의 eigenvalue 목록.
    """
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(eigenvalues)), eigenvalues, color='skyblue')
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.title("Hessian Eigenvalues")
    plt.savefig('Hessian.png')
    plt.show()



if __name__ == "__main__":
    gate_set = ["RX", "RY", "RZ", "CNOT", "H", "RX_arctan", "RY_arctan", "RZ_arctan"]
    num_gates = 20

    random_circuit = generate_random_circuit(num_gates, gate_set)
    losses = generate_loss_landscape_data(random_circuit)

    metrics = evaluate_loss_landscape(random_circuit, losses, loss_function)

    best_params = np.array(metrics["best_params"])  # 최적 파라미터
    eigenvalues = metrics["eigenvalues"]

    # Hessian eigenvalues 시각화
    plot_hessian_eigenvalues(eigenvalues)

    # Loss landscape 시각화
    visualize_loss_landscape(loss_function, best_params, random_circuit,
                             grid_range=(-0.5, 0.5), grid_points=50)
    print('rr')