import itertools
import random

import pennylane as qml
import torch
from torch import nn
import matplotlib.pyplot as plt
from data import data_load_and_process, new_data
import plotly.graph_objs as go
import plotly.offline as pyo


num_qubit = 4
dev = qml.device("default.qubit", wires=num_qubit)


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


def generate_random_circuit(num_gates, gate_set):
    structure = []
    qubit_indices = list(itertools.permutations(range(num_qubit), 2))
    for _ in range(num_gates):
        gate = random.choice(gate_set)
        control, target = random.choice(qubit_indices)
        feature_dim = random.choice(range(num_qubit))
        structure.append([gate, control, target, feature_dim])

    return structure


def sample_loss_landscape(nn_model, random_circuit, x1_data, x2_data, y_data, loss_fn, num_sample, perturb_scale):
    baseline = torch.nn.utils.parameters_to_vector(nn_model.parameters()).detach().clone()
    landscape = []

    for _ in range(num_sample):
        # baseline에 노이즈(perturbation)를 추가
        perturbation = torch.randn_like(baseline) * perturb_scale
        new_params = baseline + perturbation

        # nn_model에 새로운 파라미터를 적용 (일시적으로)
        torch.nn.utils.vector_to_parameters(new_params, nn_model.parameters())

        loss = eval_model(nn_model, random_circuit, x1_data, x2_data, y_data, loss_fn)

        perturb_norm = perturbation.norm().item()
        landscape.append((perturb_norm, loss))

    torch.nn.utils.vector_to_parameters(baseline, nn_model.parameters())

    return landscape


def eval_model(nn_model, random_circuit, x1_data, x2_data, y_data, loss_fn):
    output_x1 = nn_model(x1_data)
    output_x2 = nn_model(x2_data)
    loss = loss_fn(random_circuit, output_x1, output_x2, y_data)

    return loss



def loss_function(circuit_structure, x1_data, x2_data, y_data):
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


def compute_hessian(nn_model, random_circuit, x1_data, x2_data, y_data, loss_fn):
    """
    현재 NN 파라미터에서 loss의 Hessian Matrix를 계산합니다.
    """
    baseline = torch.nn.utils.parameters_to_vector(nn_model.parameters()).detach().clone()

    def loss_wrapper(params):
        torch.nn.utils.vector_to_parameters(params, nn_model.parameters())
        return loss_fn(random_circuit, nn_model(x1_data), nn_model(x2_data), y_data)

    hessian_matrix = torch.autograd.functional.hessian(loss_wrapper, baseline)
    return hessian_matrix


def compute_fisher_information(nn_model, random_circuit, x1_data, x2_data, y_data, loss_fn):
    """
    배치 내 각 샘플에 대해 gradient outer product를 계산한 후 평균하여 Fisher Information Matrix를 구합니다.
    """
    n_samples = x1_data.shape[0]
    fisher_matrix = None
    for i in range(n_samples):
        x1_sample = x1_data[i: i + 1]
        x2_sample = x2_data[i: i + 1]
        y_sample = y_data[i: i + 1]
        loss = loss_fn(random_circuit, nn_model(x1_sample), nn_model(x2_sample), y_sample)
        grads = torch.autograd.grad(loss, nn_model.parameters(), retain_graph=False)
        grad_vector = torch.nn.utils.parameters_to_vector(grads).detach()
        outer_product = grad_vector.unsqueeze(1) @ grad_vector.unsqueeze(0)
        fisher_matrix = outer_product if fisher_matrix is None else fisher_matrix + outer_product
    fisher_matrix /= n_samples
    return fisher_matrix


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
    condition_number = max_eig / min_eig if min_eig > 1e-12 else float("inf")
    return condition_number


def compute_local_lipschitz(hessian_matrix):
    """
    Local Lipschitz constant를 Hessian의 최대 고유값(abs)로 근사합니다.
    """
    eigenvalues = torch.linalg.eigvals(hessian_matrix).real
    L = eigenvalues.abs().max().item()
    return L


def interpolation_loss(nn_model, random_circuit, x1_data, x2_data, y_data, loss_fn, params1, params2, num_points=50):
    """
    두 최소점(파라미터 벡터 params1, params2) 사이를 선형 보간하면서 loss를 계산합니다.
    """
    alphas = torch.linspace(0, 1, num_points)
    losses = []
    baseline = torch.nn.utils.parameters_to_vector(nn_model.parameters()).detach().clone()
    for alpha in alphas:
        interp_params = (1 - alpha) * params1 + alpha * params2
        torch.nn.utils.vector_to_parameters(interp_params, nn_model.parameters())
        loss = loss_fn(random_circuit, nn_model(x1_data), nn_model(x2_data), y_data).item()
        losses.append(loss)
    torch.nn.utils.vector_to_parameters(baseline, nn_model.parameters())
    return alphas, losses


def analyze_trajectory(nn_model, trajectory, random_circuit, x1_data, x2_data, y_data, loss_fn):
    """
    최적화 과정에서 저장한 파라미터 trajectory에 대해 각 시점의 loss와 gradient norm을 계산합니다.
    trajectory: [param_vector1, param_vector2, ...] 리스트
    """
    losses = []
    grad_norms = []
    baseline = torch.nn.utils.parameters_to_vector(nn_model.parameters()).detach().clone()
    for params in trajectory:
        torch.nn.utils.vector_to_parameters(params, nn_model.parameters())
        loss = loss_fn(random_circuit, nn_model(x1_data), nn_model(x2_data), y_data)
        losses.append(loss.item())
        grads = torch.autograd.grad(loss, nn_model.parameters(), retain_graph=False)
        grad_vector = torch.nn.utils.parameters_to_vector(grads).detach()
        grad_norms.append(grad_vector.norm().item())
    torch.nn.utils.vector_to_parameters(baseline, nn_model.parameters())
    return {"losses": losses, "grad_norms": grad_norms}


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
        loss = loss_fn(random_circuit, nn_model(x1_data), nn_model(x2_data), y_data)
        grads = torch.autograd.grad(loss, nn_model.parameters(), retain_graph=False)
        grad_vector = torch.nn.utils.parameters_to_vector(grads).detach()
        grad_norms.append(grad_vector.norm().item())
    torch.nn.utils.vector_to_parameters(baseline, nn_model.parameters())
    return grad_norms


# --- 시각화 함수들 ---

def plot_2d_loss_contour(nn_model, random_circuit, x1_data, x2_data, y_data, loss_fn,
                         direction1, direction2, grid_points=30, title="Loss Landscape Contour"):
    """
    주어진 두 방향 (direction1, direction2)에 대해
    baseline 주변의 loss 값을 grid로 샘플링하여 2D contour와 3D surface plot을 그립니다.

    direction1, direction2: baseline과 동일 shape의 파라미터 벡터 (보통 무작위 perturbation)
    """
    baseline = torch.nn.utils.parameters_to_vector(nn_model.parameters()).detach().clone()
    alphas = torch.linspace(-1, 1, grid_points)
    betas = torch.linspace(-1, 1, grid_points)
    loss_grid = torch.zeros(grid_points, grid_points)

    for i, alpha in enumerate(alphas):
        print(f'{i}/{len(alphas)} alpha')
        for j, beta in enumerate(betas):
            perturb = alpha * direction1 + beta * direction2
            new_params = baseline + perturb
            torch.nn.utils.vector_to_parameters(new_params, nn_model.parameters())
            loss = loss_fn(random_circuit, nn_model(x1_data), nn_model(x2_data), y_data).item()
            loss_grid[i, j] = loss
    torch.nn.utils.vector_to_parameters(baseline, nn_model.parameters())

    # 2D Contour Plot
    plt.figure(figsize=(8, 6))
    A, B = torch.meshgrid(alphas, betas, indexing="ij")
    contour = plt.contourf(A.numpy(), B.numpy(), loss_grid.numpy(), levels=50, cmap="viridis")
    plt.xlabel("Alpha")
    plt.ylabel("Beta")
    plt.title(title)
    plt.colorbar(contour)
    plt.show()

    # 3D Surface Plot (matplotlib)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(A.numpy(), B.numpy(), loss_grid.numpy(), cmap="viridis")
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Beta")
    ax.set_zlabel("Loss")
    ax.set_title(title + " (3D Surface)")
    plt.show()

    return alphas, betas, loss_grid


def plot_interactive_loss_surface(alphas, betas, loss_grid,
                                  title="Interactive Loss Surface", filename="loss_surface.html"):
    """
    plotly를 사용하여 interactive 3D surface plot을 html 파일로 저장합니다.
    """

    surface = go.Surface(
        z=loss_grid.numpy(),
        x=alphas.numpy(),
        y=betas.numpy(),
        colorscale="Viridis"
    )
    layout = go.Layout(
        title=title,
        scene=dict(xaxis_title="Alpha", yaxis_title="Beta", zaxis_title="Loss"),
    )
    fig = go.Figure(data=[surface], layout=layout)
    pyo.plot(fig, filename=filename)



if __name__ == "__main__":
    gate_set = ["RX", "RY", "RZ", "CNOT", "H", "RX_arctan", "RY_arctan", "RZ_arctan"]
    num_gates = 20
    sampling_amount = 400
    perturb_scale = 0.1
    num_sample = 100

    X_train, X_test, Y_train, Y_test = data_load_and_process(dataset='kmnist', reduction_sz=num_qubit)
    X1_batch, X2_batch, Y_batch = new_data(sampling_amount, X_train, Y_train)

    random_circuit = generate_random_circuit(num_gates, gate_set)
    nn_model = NN()

    landscape = sample_loss_landscape(nn_model, random_circuit, X1_batch, X2_batch, Y_batch,
                                      loss_function, num_sample, perturb_scale)

    # Hessian 계산 및 관련 지표
    hessian = compute_hessian(nn_model, random_circuit, X1_batch, X2_batch, Y_batch, loss_function)
    flatness_metrics = compute_flatness(hessian)
    cond_number = compute_condition_number(hessian)
    local_lipschitz = compute_local_lipschitz(hessian)
    print("Hessian 관련 지표:", flatness_metrics)
    print("Condition number:", cond_number)
    print("Local Lipschitz Constant:", local_lipschitz)

    # Fisher Information Matrix 계산
    fisher_matrix = compute_fisher_information(nn_model, random_circuit, X1_batch, X2_batch, Y_batch, loss_function)
    print("Fisher Information Matrix shape:", fisher_matrix.shape)

    # Gradient Norm Distribution
    grad_norms = sample_gradient_norm_distribution(nn_model, random_circuit, X1_batch, X2_batch, Y_batch, loss_function,
                                                   num_samples=50, perturb_scale=0.1)
    plt.hist(grad_norms, bins=20, color="skyblue")
    plt.xlabel("Gradient Norm")
    plt.ylabel("Frequency")
    plt.title("Gradient Norm Distribution")
    plt.show()

    # 보간(interpolation) 예시: 임의의 두 파라미터 벡터 선택 (예를 들어, baseline과 baseline에 작은 perturbation 추가)
    baseline = torch.nn.utils.parameters_to_vector(nn_model.parameters()).detach().clone()
    perturb = torch.randn_like(baseline) * 0.05
    params1 = baseline
    params2 = baseline + perturb
    alphas, interp_losses = interpolation_loss(nn_model, random_circuit, X1_batch, X2_batch, Y_batch, loss_function,
                                               params1, params2, num_points=50)
    plt.plot(alphas.numpy(), interp_losses, marker="o")
    plt.xlabel("Interpolation Coefficient (alpha)")
    plt.ylabel("Loss")
    plt.title("Interpolation Between Two Minima")
    plt.show()

    # Trajectory Analysis (예시: 임의로 10단계의 파라미터 trajectory 생성)
    trajectory = []
    for i in range(10):
        traj_params = baseline + torch.randn_like(baseline) * (0.01 * i)
        trajectory.append(traj_params)
    traj_metrics = analyze_trajectory(nn_model, trajectory, random_circuit, X1_batch, X2_batch, Y_batch, loss_function)
    plt.plot(traj_metrics["losses"], marker="o")
    plt.xlabel("Trajectory Step")
    plt.ylabel("Loss")
    plt.title("Loss along Trajectory")
    plt.show()
    plt.plot(traj_metrics["grad_norms"], marker="o")
    plt.xlabel("Trajectory Step")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Norm along Trajectory")
    plt.show()

    # 2D Contour & 3D Surface Plot
    # (임의의 두 방향 벡터: baseline과 동일 차원, 보통 무작위로 생성)
    direction1 = torch.randn_like(baseline) * 0.1
    direction2 = torch.randn_like(baseline) * 0.1
    alphas_grid, betas_grid, loss_grid = plot_2d_loss_contour(nn_model, random_circuit, X1_batch, X2_batch, Y_batch,
                                                              loss_function, direction1, direction2,
                                                              grid_points=30, title="Loss Landscape")

    # Plotly를 통한 interactive 3D plot 생성 (html 파일로 저장)
    plot_interactive_loss_surface(alphas_grid, betas_grid, loss_grid,
                                  title="Interactive Loss Surface", filename="loss_surface.html")
