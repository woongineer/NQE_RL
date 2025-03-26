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
    """
    circuit: list of gate dicts
    qubit_index: int
    depth_index: int
    insert_decision: dict with keys: gate_type, param, qubits
    """
    new_circuit = circuit.copy()
    gate_type = insert_decision["gate_type"]
    param = insert_decision["param"]
    qubits = insert_decision["qubits"]

    # 먼저 CNOT인지 아닌지 확인
    if gate_type == "CNOT":

        ctrl, tgt = qubits

        for i, gate in enumerate(new_circuit):
            if (
                gate["depth"] == depth_index
                and gate["qubits"][0] == ctrl
            ):
                new_circuit[i] = {
                    "gate_type": "CNOT",
                    "depth": depth_index,
                    "qubits": (ctrl, tgt),
                    "param": None
                }
            if (
                gate["depth"] == depth_index
                and gate["qubits"][0] == tgt
            ):
                del new_circuit[i]

    else:
        # RX, RY, RZ, H, I 등의 1-qubit gate
        for i, gate in enumerate(new_circuit):
            if gate["depth"] == depth_index and gate["qubits"][0] == qubits[0]:
                new_circuit[i] = {
                    "gate_type": gate_type,
                    "depth": depth_index,
                    "qubits": qubits,
                    "param": param
                }

    return new_circuit
