import torch
from datetime import datetime

from data import data_load_and_process as dataprep
from data import new_data
from model import CNNLSTM, NQEModel, ValueNetwork
from utils import generate_layers, make_arch, plot_policy_loss
from utils_for_analysis import save_probability_animation, save_trajectory

if __name__ == "__main__":
    print(datetime.now())
    # 파라미터
    num_qubit = 4

    max_epoch_PG = 300  # 50
    max_layer_step = 5
    max_epoch_NQE = 50  # 50

    batch_size = 25
    num_layer = 64

    lr_NQE = 0.01
    lr_PG = 0.005
    lr_value = 0.005

    temperature = 0.4
    discount = 0.85

    num_gate_class = 5

    # 미리 만들 것
    layer_set = generate_layers(num_qubit, num_layer)
    X_train, X_test, Y_train, Y_test = dataprep(dataset='kmnist', reduction_sz=num_qubit)

    policy = CNNLSTM(feature_dim=16, hidden_dim=32, output_dim=num_layer, num_layers=1)
    policy.train()

    value_net = ValueNetwork(feature_dim=16, hidden_dim=32)
    value_net.train()

    loss_fn = torch.nn.MSELoss()
    PG_opt = torch.optim.Adam(policy.parameters(), lr=lr_PG)
    value_opt = torch.optim.Adam(value_net.parameters(), lr=lr_value)

    gate_list = None
    arch_list = {}
    prob_list = {}
    layer_list_list = {}

    for pg_epoch in range(max_epoch_PG):
        print(f"{pg_epoch+1}th PG epoch")
        layer_list = []
        reward_list = []
        log_prob_list = []
        value_list = []

        current_arch = torch.randint(0, 1, (1, 1, num_qubit, num_gate_class)).float()

        for layer_step in range(max_layer_step):
            # print(f"building layer {layer_step + 1}th...")
            output = policy.forward(current_arch)
            prob = torch.softmax(output.squeeze() / temperature, dim=-1)

            dist = torch.distributions.Categorical(prob)
            layer_index = dist.sample()
            layer_list.append(layer_index)

            value = value_net(current_arch)  # shape: [1, 1]
            value_list.append(value.squeeze())

            gate_list = [item for i in layer_list for item in layer_set[int(i)]]
            current_arch = make_arch(gate_list, num_qubit)

            NQE_model = NQEModel(gate_list)
            NQE_model.train()
            NQE_opt = torch.optim.SGD(NQE_model.parameters(), lr=lr_NQE)

            for nqe_epoch in range(max_epoch_NQE):
                X1_batch, X2_batch, Y_batch = new_data(batch_size, X_train, Y_train)
                pred = NQE_model(X1_batch, X2_batch)
                loss_nqe = loss_fn(pred, Y_batch)

                NQE_opt.zero_grad()
                loss_nqe.backward()
                NQE_opt.step()

            valid_loss_list = []
            NQE_model.eval()
            for _ in range(batch_size):
                X1_batch, X2_batch, Y_batch = new_data(batch_size, X_test, Y_test)
                with torch.no_grad():
                    pred = NQE_model(X1_batch, X2_batch)
                valid_loss_list.append(loss_fn(pred, Y_batch))

            loss = sum(valid_loss_list) / batch_size
            reward = 1 - loss

            log_prob = dist.log_prob(layer_index)
            log_prob_list.append(log_prob)
            reward_list.append(reward)

        layer_list_list[pg_epoch + 1] = {'layer_list': layer_list}
        prob_list[pg_epoch + 1] = {'prob': prob.detach().tolist()}

        returns = []
        G = 0
        for r in reversed(reward_list):
            G = r + discount * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        values = torch.stack(value_list)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        log_prob_tensor = torch.stack(log_prob_list)
        policy_loss = -(log_prob_tensor * advantages.detach()).mean()
        value_loss = advantages.pow(2).mean()
        total_loss = policy_loss + value_loss

        print(f'policy_loss: {policy_loss}, value_loss: {value_loss}')

        arch_list[pg_epoch + 1] = {"policy_loss": policy_loss.item(), "NQE_loss": loss.item(), "gate_list": gate_list}

        PG_opt.zero_grad()
        value_opt.zero_grad()

        total_loss.backward()

        total_norm = 0.0
        for p in policy.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=1.0)

        PG_opt.step()
        value_opt.step()

    plot_policy_loss(arch_list, 'loss_base_lr05.png')
    save_probability_animation(prob_list, "animation_base_lr05.mp4")
    save_trajectory(layer_list_list, filename="trajectory_base_lr05.png", max_epoch_PG=max_epoch_PG, num_layer=num_layer)

    print(datetime.now())