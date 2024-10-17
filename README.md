## 국민대학교 자율주행컴퓨팅 과제#2-강화학습 (Deep Q Network 적용한 Car Racing 과제)

### 1. DQN 코드 작성
 - src/DQN.py 내 write code 밑에 코드를 작성하였습니다. 각 버젼별로 첨부하였습니다.
 - 결과물 (결과물 비디오-> video folder 및 Avgreturn graph는 png형태) 첨부하였습니다.   

### 2. src/CNN.py 내 네트워크 구조를 변경시켜보면서 그 결과를 비교 분석 할 것 (e.g. 효용이 없다, 성능 향상이 있다 등)
 - CNN 내 layer 추가, CNN kernel_size 및 stride 변화를 주었고 상세 설명은 pdf에 하였습니다.
 - class cnn, improved cnn으로 class 2개를 한 py에 넣었습니다. 
   
### v2- video output
![Video](https://github.com/kimmyeongjune/reinforcement_learning-HW2/blob/master/v2_video/car_racing_v2_30%2C000_first.gif?raw=true)

### 추가로 95200step을 더 학습 시켜서 평균적으로 return이 650정도에 달했고, 학습 parameter는 다음과 같습니다.
 - lr=0.0001,
 - epsilon=1.0,
 - epsilon_min=0.05,
 - gamma=0.99,
 - batch_size=32,
 - warmup_steps=5000,
 - buffer_size=int(1e5),
 - target_update_interval=5000
 epsilon decay만 진행하고 gamma는 건들이지 않았으며, 장기보상에 더 집중하도록 훈련을 진행하였습니다.
