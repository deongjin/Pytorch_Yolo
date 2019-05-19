# 3. 손실 함수와 Optimizer 정의하기
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 이제, 분류에 대한 교차 엔트로피 손실(Cross-Entropy loss)과 momentum을 갖는
# SGD를 사용합니다.

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)