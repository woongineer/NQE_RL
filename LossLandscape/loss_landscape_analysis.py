from data import data_load_and_process, new_data
from datetime import datetime
import pickle
import numpy as np
from utils import generate_random_circuit, NN, compute_hessian, compute_flatness, compute_condition_number, \
    compute_local_lipschitz, compute_fisher_information, loss_function, sample_gradient_norm_distribution, NQE, \
    get_good_and_bad, rerun_good_bad, compute_QNTK, compute_local_ED, check_lazy_regime_from_params

num_qubit = 4

if __name__ == "__main__":
    gate_set = ["RX", "RY", "RZ", "CNOT", "H", "RX_arctan", "RY_arctan", "RZ_arctan"]
    num_gates_range = [8, 20]
    batch_size = 400
    perturb_scale = 0.1
    num_sample = 100

    batch_size_for_NQE = 25
    iter_for_NQE = 100
    num_of_trial = 5

    num_of_rand_circuit = 100

    good_bad_N = 10

    X_train, X_test, Y_train, Y_test = data_load_and_process(dataset='kmnist', reduction_sz=num_qubit)
    X1_batch, X2_batch, Y_batch = new_data(batch_size, X_train, Y_train)

    nn_model = NN()

    landscape_resuls_list = {}
    for i in range(num_of_rand_circuit):
        print(f"BIG {i}th iteration.........{datetime.now()}")
        random_circuit = generate_random_circuit(num_qubit, num_gates_range, gate_set)

        NQE_results = []
        lazy_checks = []
        for _ in range(num_of_trial):
            valid_loss_list, initial_params, trained_params = NQE(
                random_circuit, batch_size_for_NQE, iter_for_NQE,
                X_train, X_test, Y_train, Y_test
            )
            NQE_results.append(valid_loss_list)
            is_lazy, change_ratio = check_lazy_regime_from_params(initial_params, trained_params)
            lazy_checks.append({"is_lazy": is_lazy, "change_ratio": change_ratio})

        # 평균적인 lazy regime 체크 결과
        avg_change_ratio = np.mean([x["change_ratio"] for x in lazy_checks])
        is_lazy_regime = avg_change_ratio < 0.1  # 평균을 기준으로 판단

        # NQE_results = [NQE(random_circuit, batch_size_for_NQE, iter_for_NQE, X_train, X_test, Y_train, Y_test) for _ in
        #                range(num_of_trial)]
        hessian = compute_hessian(nn_model, random_circuit, X1_batch, X2_batch, Y_batch, loss_function)
        flatness_metrics = compute_flatness(hessian)
        cond_number = compute_condition_number(hessian)
        local_lipschitz = compute_local_lipschitz(hessian)

        fisher_matrix = compute_fisher_information(nn_model, random_circuit, X1_batch, X2_batch, Y_batch, loss_function)
        grad_norms = sample_gradient_norm_distribution(nn_model, random_circuit, X1_batch, X2_batch, Y_batch,
                                                       loss_function, num_samples=50, perturb_scale=0.1)

        # Local ED 계산 및 저장
        local_ed_results = compute_local_ED(fisher_matrix, num_data=batch_size)

        # QNTK 계산 (X1_batch를 사용하여 대표로 분석)
        qntk_results = compute_QNTK(nn_model, random_circuit, X1_batch)

        landscape_resuls_list[i] = {
            "random_circuit": random_circuit,
            "NQE_results": NQE_results,
            "hessian": hessian,
            "flatness_metrics": flatness_metrics,
            "cond_number": cond_number,
            "local_lipschitz": local_lipschitz,
            "fisher_matrix": fisher_matrix,
            "grad_norms": grad_norms,
            "local_ed_results": local_ed_results,
            "qntk_results": qntk_results,
            "lazy_checks": lazy_checks,
            "is_lazy_regime_avg": is_lazy_regime,
            "avg_param_change_ratio": avg_change_ratio
        }

    with open("landscape_list_2.pkl", "wb") as f:
        pickle.dump(landscape_resuls_list, f)

    # good_idx, bad_idx, good_means, bad_means = get_good_and_bad(landscape_resuls_list, good_bad_N)
    # rerun_good, rerun_bad = rerun_good_bad(landscape_resuls_list, good_idx, bad_idx,
    #                                        batch_size_for_NQE, iter_for_NQE, X_train, X_test, Y_train, Y_test, num_of_trial)

    print('debugging point')
