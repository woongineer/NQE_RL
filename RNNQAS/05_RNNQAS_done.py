import torch
from datetime import datetime

from data import data_load_and_process as dataprep
from data import new_data
from model import CNNLSTM, NQEModel
from utils import generate_layers, make_arch, plot_policy_loss, set_done_loss
from utils_for_analysis import save_probability_animation, save_trajectory, plot_policy_loss_dual_axis

if __name__ == "__main__":
    print(datetime.now())
    # 파라미터
    num_qubit = 4

    max_epoch_PG = 500  # 50
    max_layer_step = 10
    max_epoch_NQE = 50  # 50

    batch_size = 25
    num_layer = 64

    lr_NQE = 0.01
    lr_PG = 0.002

    temperature = 0.5
    discount = 0.9

    num_gate_class = 5

    done_bonus = 0.2

    # 미리 만들 것
    layer_set = generate_layers(num_qubit, num_layer)
    X_train, X_test, Y_train, Y_test = dataprep(dataset='kmnist', reduction_sz=num_qubit)
    soft_done, hard_done = set_done_loss(num_qubit, max_epoch_NQE, batch_size, X_train, Y_train, X_test, Y_test)

    policy = CNNLSTM(feature_dim=16, hidden_dim=32, output_dim=num_layer, num_layers=1)
    policy.train()

    loss_fn = torch.nn.MSELoss()
    PG_opt = torch.optim.Adam(policy.parameters(), lr=lr_PG)

    gate_list = None
    loss = 0
    arch_list = {}
    prob_list = {}
    layer_list_list = {}
    for pg_epoch in range(max_epoch_PG):
        print(f"{pg_epoch+1}th PG epoch")
        layer_list = []
        reward_list = []
        log_prob_list = []

        current_arch = torch.randint(0, 1, (1, 1, num_qubit, num_gate_class)).float()

        done = False

        for layer_step in range(max_layer_step):
            if done:
                break

            print(f"building layer {layer_step + 1}th...")

            output = policy.forward(current_arch)
            prob = torch.softmax(output.squeeze() / temperature, dim=-1)

            dist = torch.distributions.Categorical(prob)
            layer_index = dist.sample()
            layer_list.append(layer_index)

            gate_list = [item for i in layer_list for item in layer_set[int(i)]]
            current_arch = make_arch(gate_list, num_qubit)

            NQE_model = NQEModel(gate_list)
            NQE_model.train()
            NQE_opt = torch.optim.SGD(NQE_model.parameters(), lr=lr_NQE)

            for nqe_epoch in range(max_epoch_NQE):
                X1_batch, X2_batch, Y_batch = new_data(batch_size, X_train, Y_train)
                pred = NQE_model(X1_batch, X2_batch)
                loss = loss_fn(pred, Y_batch)

                NQE_opt.zero_grad()
                loss.backward()
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

            if loss < hard_done:
                print(f"Done triggered at layer {layer_step + 1} with loss={loss:.5f}")
                reward += done_bonus
                done = True

            log_prob = dist.log_prob(layer_index.clone().detach())
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

        log_prob_tensor = torch.stack(log_prob_list)
        policy_loss = -log_prob_tensor * returns
        policy_loss = policy_loss.mean()
        print(f'policy_loss: {policy_loss}')

        arch_list[pg_epoch + 1] = {"policy_loss": policy_loss.item(), "NQE_loss": loss.item(), "gate_list": gate_list}

        PG_opt.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        PG_opt.step()

    plot_policy_loss(arch_list, 'old_loss_done.png')
    save_probability_animation(prob_list, "animation_done.mp4")
    plot_policy_loss_dual_axis(arch_list, 'loss_done.png')
    save_trajectory(layer_list_list, filename="trajectory_done.png", max_epoch_PG=max_epoch_PG, num_layer=num_layer)
    print(datetime.now())

