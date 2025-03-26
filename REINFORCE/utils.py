import numpy as np
import torch
from torch import nn



def check_circuit_structure(circuit):
    from collections import defaultdict

    # depth별로 qubit[0]을 저장
    depth_qubits = defaultdict(set)
    depth_cnot_targets = defaultdict(set)

    for gate in circuit:
        d = gate['depth']
        q0 = gate['qubits'][0]
        q1 = gate['qubits'][1]

        # 같은 depth에 같은 qubit 위치가 중복되면 'd' 출력
        if q0 in depth_qubits[d]:
            print('중복')
        else:
            depth_qubits[d].add(q0)

        # CNOT이면 target qubit 저장
        if gate['gate_type'] == 'CNOT' and q1 is not None:
            depth_cnot_targets[d].add(q1)

    for d in depth_qubits:
        # depth d에서 필요한 qubit[0] index
        expected_qubits = {0, 1, 2, 3}
        present_qubits = depth_qubits[d]

        # CNOT의 target으로 사용된 qubit들은 필수에서 제외
        missing_qubits = expected_qubits - present_qubits - depth_cnot_targets[d]

        if missing_qubits:
            print('업슴')


def ordering(circuit):
    return sorted(circuit, key=lambda g: (g['depth'], g['qubits'][0]))


def sample_insert_gate_param(prob_tensor, insert_gate_map, qubit_index, num_of_qubit,
                             circuit, depth_index):
    # value → key 역변환을 위한 dict
    index_to_gate_name = {v: k for k, v in insert_gate_map.items()}

    ###########수정된 부분##########
    # 확률 텐서를 복제해두고, 여기서 중간에 마스킹할 예정
    local_prob = prob_tensor.clone()

    for _ in range(100):  # 최대 100번까지 재시도
        # 현재 local_prob를 기반으로 Categorical 분포
        dist = torch.distributions.Categorical(local_prob)
        sampled_index = dist.sample().item()

        # 해당 index에 해당하는 gate name
        gate_name = index_to_gate_name[sampled_index]

        # 파싱
        if "_" in gate_name:
            parts = gate_name.split("_")
            if parts[0] in ["RX", "RY", "RZ"]:
                gate_type = parts[0]
                param = int(parts[1])
                gate_info = {
                    "gate_type": gate_type,
                    "param": param,
                    "qubits": (qubit_index, None)
                }
            elif parts[0] == "CNOT":
                adder = int(parts[1])
                target = (qubit_index + adder) % num_of_qubit
                gate_info = {
                    "gate_type": "CNOT",
                    "param": None,
                    "qubits": (qubit_index, target)
                }
            else:
                raise ValueError(f"Unknown gate name: {gate_name}")
        elif gate_name == "H":
            gate_info = {
                "gate_type": "H",
                "param": None,
                "qubits": (qubit_index, None)
            }
        elif gate_name == "I":
            gate_info = {
                "gate_type": "I",
                "param": None,
                "qubits": (qubit_index, None)
            }
        else:
            raise ValueError(f"Unknown gate name: {gate_name}")

        # ============== CNOT conflict 체크 로직 추가 ==============
        if gate_info["gate_type"] == "CNOT":
            ctrl, tgt = gate_info["qubits"]
            # 이미 해당 depth에서 tgt가 다른 게이트로 점유되어 있다면 conflict
            conflict = False
            for g in circuit:
                if g["depth"] == depth_index:
                    # g["qubits"]가 (x, y)일 수도 있으니 tgt가 포함되었는지 체크
                    if tgt in g["qubits"]:
                        conflict = True
                        break

            if conflict:
                # 해당 게이트 index를 확률 0으로 만들고 재정규화 후 다시 샘플링
                local_prob[sampled_index] = 0.0
                prob_sum = local_prob.sum()
                if prob_sum < 1e-9:
                    # 더이상 뽑을 게이트가 전혀 없다면 에러
                    raise ValueError("[sample_insert_gate_param] 모든 게이트가 마스킹되었습니다.")
                local_prob = local_prob / prob_sum
                continue  # 재샘플링으로 넘어감

        # 여기까지 conflict가 없다면, 그대로 gate_info 반환
        return gate_info

    # 만약 100번을 시도해도 valid 게이트를 못 뽑으면 에러
    raise ValueError("[sample_insert_gate_param] 유효한 게이트를 찾지 못했습니다.")
    ###########수정된 부분##########



def sample_remove_position(prob_tensor):
    """
    Args:
        prob_tensor (torch.Tensor): shape (depth, qubit), 확률값 [0, 1]

    Returns:
        (depth_idx, qubit_idx): 선택된 위치의 인덱스
    """
    # 확률 tensor를 flat하게 만들고
    flat_probs = prob_tensor.flatten()  # shape: (depth * qubit,)

    # 정규화 (합이 1이 되도록)
    norm_probs = flat_probs / flat_probs.sum()

    # multinomial 샘플링
    idx = torch.multinomial(norm_probs, num_samples=1).item()

    depth = idx // prob_tensor.shape[1]
    qubit = idx % prob_tensor.shape[1]
    return depth, qubit


class FixedLinearProjection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.linear.weight.requires_grad = False

        # 초기화 방법: 정규분포, 아이덴티티 일부, 등등
        with torch.no_grad():
            torch.nn.init.normal_(self.linear.weight, mean=0.0, std=0.1)

    def forward(self, x):
        # x: (B, C, D, Q) → (B, D, Q, C)
        x = x.permute(0, 2, 3, 1)
        out = self.linear(x)  # (B, D, Q, hidden_dim)
        return out


def representer(circuit_info, num_qubits, depth, gate_types):
    gate_channel_mapping = {}
    channel_counter = 0
    rotation_set = list(range(num_qubits))

    # 1-qubit rotation gates (RX, RY, RZ) with param index
    for g in gate_types:
        if g.startswith("R"):
            for p in rotation_set:
                gate_channel_mapping[f"{g}_{p}"] = channel_counter
                channel_counter += 1

    # H gate
    if "H" in gate_types:
        gate_channel_mapping["H"] = channel_counter
        channel_counter += 1

    # CNOTs (control ≠ target)
    for q0 in range(num_qubits):
        for q1 in range(num_qubits):
            if q0 == q1:
                continue
            gate_channel_mapping[f"CNOT_{q0}_{q1}"] = channel_counter
            channel_counter += 1

    # 마지막 채널: NoGate (Identity)
    gate_channel_mapping["I"] = channel_counter
    total_channels = channel_counter + 1

    tensor = np.zeros((depth, num_qubits, total_channels), dtype=np.float32)

    for gate in circuit_info:
        d = gate["depth"]
        q0, q1 = gate["qubits"]
        g_type = gate["gate_type"]
        p_idx = gate["param"]

        key = None
        if g_type.startswith("R"):
            key = f"{g_type}_{p_idx}"
            if key in gate_channel_mapping:
                ch = gate_channel_mapping[key]
                tensor[d, q0, ch] = 1.0

        elif g_type == "H":
            key = "H"
            ch = gate_channel_mapping[key]
            tensor[d, q0, ch] = 1.0

        elif g_type == "CNOT":
            key = f"CNOT_{q0}_{q1}"
            if key in gate_channel_mapping:
                ch = gate_channel_mapping[key]
                tensor[d, q0, ch] = 1.0
                tensor[d, q1, ch] = 1.0

    # 나머지 채널에 NoGate 표시
    for d in range(depth):
        for q in range(num_qubits):
            if tensor[d, q, :total_channels - 1].sum() == 0:
                tensor[d, q, gate_channel_mapping["I"]] = 1.0

    # reshape to (1, C, D, Q)
    tensor = tensor.transpose(2, 0, 1)  # (C, D, Q)
    tensor = tensor[np.newaxis, :, :, :]  # (1, C, D, Q)
    return torch.tensor(tensor, dtype=torch.float32)


def fill_identity_gates(circuit_info, num_of_qubit, total_depth):
    filled_circuit_info = []

    for d in range(total_depth):
        # 해당 depth에서 이미 사용된 qubit 찾기
        used_qubits = set()
        for info in circuit_info:
            if info["depth"] == d:
                used_qubits.add(info["qubits"][0])
                if info["gate_type"] == "CNOT":
                    used_qubits.add(info["qubits"][1])
                filled_circuit_info.append(info)

        # 사용되지 않은 qubit에는 Identity gate 추가
        for q in range(num_of_qubit):
            if q not in used_qubits:
                filled_circuit_info.append({
                    "gate_type": "I",
                    "depth": d,
                    "qubits": (q, None),
                    "param": None
                })

    # depth 순으로 정렬해서 리턴
    filled_circuit_info.sort(key=lambda x: x["depth"])
    return filled_circuit_info
