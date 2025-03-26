def remover(circuit, qubit_index, depth_index):
    # 조건에 맞는 gate들을 모두 찾기
    matched_gates = [
        (i, gate) for i, gate in enumerate(circuit)
        if gate['depth'] == depth_index and qubit_index in gate['qubits']
    ]

    matched_gates = list({id(g): (i, g) for i, g in matched_gates}.values())

    if len(matched_gates) == 0:
        raise ValueError("No matching gate for given qubit_index & depth_index")
    elif len(matched_gates) > 1:
        raise ValueError("More than 2 matching gate for given qubit_index & depth_index")

    idx, gate = matched_gates[0]

    if gate['gate_type'] == 'CNOT':
        # 기존 CNOT 게이트를 제거
        del circuit[idx]
        # 두 개의 'I' 게이트로 분할 삽입
        for q in gate['qubits']:
            circuit.insert(idx, {
                'gate_type': 'I',
                'depth': depth_index,
                'qubits': (q, None),
                'param': None
            })
            idx += 1  # 다음 위치에 삽입
    else:
        gate['gate_type'] = 'I'
        gate['param'] = None
        gate['qubits'] = (qubit_index, None)  # 단일 qubit만 남기기

    return circuit


def inserter(circuit, depth_index, insert_decision):
    new_circuit = circuit.copy()
    gate_type = insert_decision["gate_type"]
    param = insert_decision["param"]
    qubits = insert_decision["qubits"]

    if gate_type == "CNOT":
        ctrl, tgt = qubits

        # 먼저 ctrl과 tgt 위치를 찾아둔 뒤, 수정/삭제는 나중에 일괄 반영
        ctrl_idx = None
        tgt_idx = None
        for i, gate in enumerate(new_circuit):
            if gate["depth"] == depth_index and gate["qubits"][0] == ctrl:
                ctrl_idx = i
            elif gate["depth"] == depth_index and gate["qubits"][0] == tgt:
                tgt_idx = i

        # ctrl 위치를 CNOT으로 변경
        if ctrl_idx is not None:
            new_circuit[ctrl_idx] = {
                "gate_type": "CNOT",
                "depth": depth_index,
                "qubits": (ctrl, tgt),
                "param": None
            }

        # tgt 위치는 뒤에서 지우는게 안전 (인덱스 에러 방지)
        # 인덱스가 ctrl_idx보다 크거나 작냐에 따라 순서 주의
        if tgt_idx is not None:
            if tgt_idx > ctrl_idx:
                del new_circuit[tgt_idx]
            else:
                # tgt_idx < ctrl_idx면, ctrl_idx가 1 줄어든 상태로 조정
                del new_circuit[tgt_idx]
                if ctrl_idx is not None:
                    ctrl_idx -= 1

    else:
        # RX, RY, RZ, H, I 등
        for i, gate in enumerate(new_circuit):
            if gate["depth"] == depth_index and gate["qubits"][0] == qubits[0]:
                new_circuit[i] = {
                    "gate_type": gate_type,
                    "depth": depth_index,
                    "qubits": qubits,
                    "param": param
                }

    return new_circuit

