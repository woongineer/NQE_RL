## RL 파일 정리

[1] RefCodeQAS: QAS 공부용 파일, 안씀 \
[2] RL_NQE: Batch 문제를 해결하지 못한 파일. Batch 처리를 mean으로 하면 안되면 다시 보기 \
[3] RL_NQE_batch_fix: Batch 문제를 mean으로 해결한 파일 \
[4] RL_NQE_batch_fix_NN: RL 코드에 NQE를 접목한 파일 \
[5] RL_NQE_compare: batch를 mean으로 해결한 버전으로 RL embedding과 ZZ embedding 비교 파일 

### 2024.09.24 탁님 피드백

[6] RL_NQE_compare_pauli: 원래는 pauli XYZ measurement를 state에 넣어줬는데, 그거를 탁님 의견에 따라 qml.probs로 바꿈 \
[7] RL_NQE_compare_pauli_reward: 원래는 reward를 policy gradient 코드에서 갖다 썼는데, 이번에는 그냥 state값으로 줌 

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


[8] RL_NQE_DK: 일단 data 빼고 state만 하는거랑, batch를 죽이던걸 살리는 방향으로 바꿈. \
[9] RL_NQE_DK_gpt: 위 파일에서 policy model에 따라 inference 후 NQE를 돌리는 코드. GPT가 짬. 위 이유때문에 이대로 가면 안되겠다는 생각이 들어서 검수는 안함. 

### 2024.10.01 혼자 생각
여태까지는 RL로 구조를 만들고 거기서 NQE를 돌릴 생각이었는데, 위 이유로 힘들 것 같다.\
RL에다가 NQE를 돌리면 개별 architecture마다 x->x'를 학습할텐데, zz embedding 실험에서는 1개 architecture에 여러 data를 통해 학습을 진행하니 unfair한거 아닐까?\
뭔가 U(RL(x))->U(NQE(RL(x)))가 제대로 될지 감이 안오네...\
반면 x'에다가 RL을 학습하면, U(x)->U(x')->U(RL(x'))아닌가...?
- RL-NQE 방식: RL generalization 가능하지만 NQE가 data specific임
- NQE-RL 방식: NQE랑 RL둘다 generalization 됨...? 이게 맞나?