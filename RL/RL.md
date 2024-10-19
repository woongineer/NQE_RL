## RL 파일 정리

[1] RL_legacy/RefCodeQAS: QAS 공부용 파일, 안씀 \
[2] RL_legacy/RL_NQE: Batch 문제를 해결하지 못한 파일. Batch 처리를 mean으로 하면 안되면 다시 보기 \
[3] RL_legacy/RL_NQE_batch_fix: Batch 문제를 mean으로 해결한 파일 \
[4] RL_legacy/RL_NQE_batch_fix_NN: RL 코드에 NQE를 접목한 파일 \
[5] RL_legacy/RL_NQE_compare: batch를 mean으로 해결한 버전으로 RL embedding과 ZZ embedding 비교 파일 

### 2024.09.24 탁님 피드백

[6] RL_legacy/RL_NQE_compare_pauli: 원래는 pauli XYZ measurement를 state에 넣어줬는데, 그거를 탁님 의견에 따라 qml.probs로 바꿈 \
[7] RL_legacy/RL_NQE_compare_pauli_reward: 원래는 reward를 policy gradient 코드에서 갖다 썼는데, 이번에는 그냥 state값으로 줌 

### 2024.09.27 교수님 피드백

loss 바꾸기/action 여러개/data빼고 state만.\
그런데 다시 보니까 그게 문제가 아니다.\
일단 교수님이 sum을 하라고 하셨는데 그러면 log_probs 부분은 dimension이 여전히 그대로인 상태임 -> 일단 sum하는 부분을 return 구하는 부분으로 함.(교수님 조언과 살짝 다름..)
학습의 맨 마지막 꺼로 모든 embedding을 대신해버렸음... 이러면 data에 맞춰서 학습을 한 의미가 없다.\
총 dataset에 맞는 embedding 구조를 쓴다 - 라고 해석할 수도 있을 것 같은데...
1) RL학습
2) RL학습 policy model로 batch마다 action sequence 생성
3) action sequence로 NQE 학습
   1) 근데 이러면 개념상, 1개 batch만 학습해야함... 다음 batch가 오면 또 action sequence를 만드는 거니까.
   2) 이러면 model_fidelity에 action_sequence랑 x데이터 두개를 다 넣어서 학습할 수도 있을 것 같은데...
4) action sequence + NQE model로 x'와 action_sequence 만든 뒤 QCNN 학습
5) action sequence + NQE model + QCNN model로 QCNN inference

3번이 결국 제일 걸린다. batch 1개에 대해서 학습하는걸 학습이라고 할 수 있나??? 그렇다고 여러 batch에 대해서 학습하면, 하나의 action_sequence로 모든 데이터를 처리하는건데, 그거는 뭔가 강화학습을 한 의미가 퇴색되는 것 같은데,,,


[8] RL_legacy/RL_NQE_DK: 일단 data 빼고 state만 하는거랑, batch를 죽이던걸 살리는 방향으로 바꿈. \
[9] RL_legacy/RL_NQE_DK_gpt: 위 파일에서 policy model에 따라 inference 후 NQE를 돌리는 코드. GPT가 짬. 위 이유때문에 이대로 가면 안되겠다는 생각이 들어서 검수는 안함. 

### 2024.10.01 혼자 생각
여태까지는 RL로 구조를 만들고 거기서 NQE를 돌릴 생각이었는데, 위 이유로 힘들 것 같다.\
RL에다가 NQE를 돌리면 개별 architecture마다 x->x'를 학습할텐데, zz embedding 실험에서는 1개 architecture에 여러 data를 통해 학습을 진행하니 unfair한거 아닐까?\
뭔가 U(RL(x))->U(NQE(RL(x)))가 제대로 될지 감이 안오네...\
반면 x'에다가 RL을 학습하면, U(x)->U(x')->U(RL(x'))아닌가...?
- RL-NQE 방식: RL generalization 가능하지만 NQE가 data specific임
- NQE-RL 방식: NQE랑 RL둘다 generalization 됨...? 이게 맞나?

[10] RL_legacy/NQE_RL_datawise.py: 위 기준에 맞춰서 NQE를 한 다음에 그 entire x'로 RL을 하는거. 근데 이렇게 해도 QCNN학습 부분에서는 학습이 한 RL 구조당 한 batch만 돌아간다.

### 2024.10.04 피드백
아니 개별 policy로 datawise하게 quantum structure를 뽑아내는게 아니였나봄...\
하나의 RL 모델이 entire dataset을 describe하는거를 생각했던듯...\
탁님에게 loss를 어디서 sum을 할 지도 물어봄. log_probs들을 sum하는게 맞을 것 같다고 하는데 log_likelihood같은 얘기면 그쪽일거라 하심\
하튼 iterative + per qubit 구조로 1개의 구조를 만들어내는게 목표다\
per qubit 으로 하기 위해서는 CNOT을 어떻게 할 지 생각해야하는데, QAS 논문에서는 바로 다음 qubit하고 하는거로 했네?\
https://github.com/qdevpsi3/quantum-arch-search/blob/main/src/qas_gym/utils.py#L7 \


[11] RL_legacy/NQE_RL_single: 위 시나리오대로 NQE 후 RL
[12] 중요 RL_legacy/NQE_RL_iter: NQE-RL-NQE-RL... iterative하게 한거. 20241011까지 기준 코드
[13] RL_legacy/NQE_RL_iter_complex: 12번 코드의 NN 부분 더 복잡하게 한거
[14] RL_legacy/NQE_RL_iter_policyfix: 12번 코드에서는 X_test할때 다시 action sequence를 안뽑음. 이거는 X_test에 대해 새로 뽑음
[15] RL_legacy/NQE_A2C: Policy Gradient 대신 A2C 써봄, gpt

### 2024.10.11 발표 피드백
이전까지의 내용 및 결과는 overleaf로 정리해서 발표함.\
[석훈] Policy based는 continuous일 때 하는 것 같다. Action space가 discrete하고 적으니 Value based RL을 써야 할 것 같다 \
G는 scaler인데 gradient가 의미가 있나? G랑 곱하는게 무슨 의미인가?\
state와 trace distance 사이의 연관성이 있나? \
Episodes가 지날수록 trace distance가 증가하는지를 확인해야함(class distance 함수) \
Gate의 Parameter까지 가능...? \
Identity도 옵션으로\
ZZ feature embedding에서 하나씩 바꿔보는 식으로 학습하는건 어떤가? ZZ->XX 이렇게 2qubit은 2qubit으로\
처음에 ZZ 안쓰고 하기?\
NQE랑 Policy를 loop 돌릴때 fine tuning 방식으로 하기\
Test에서도 action sequence를 새로 뽑는거는 아닌 것 같다.\

[16] RL_legacy/trace_distance_original: 탁님 코드에서 trace distance부분만 뽑아온 것. 공부용
[17] RL_legacy/NQE_RL_iter_trace: NQE_RL_iter에서 RL의 trace distance 계산 부분 추가
