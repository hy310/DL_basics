import torch
import torch.nn as nn

class ImageClassifier(nn.Module):     # nn.Module 을 상속받아 Image Classifier 라는 클래스를 생성

    # 보통 init 과 forward 정도 두가지 함수만 overwrite 해도 상속받았기에 잘 작동함
    
    def __init__(self,
                 input_size,
                 output_size):
        self.input_size = input_size       # MNIST 의 경우, 입력사이즈는 784
        self.output_size = output_size     # output 은 10

        super().__init__()

        self.layers = nn.Sequential(            # nn.Sequential 이용해 쉽게 구현
            nn.Linear(input_size, 500),         # 784 차원을 받아 500 으로 빼주는 linear layer
            nn.LeakyReLU(),                # 입력값을 0~1 사이값으로 (leaky 는 살짝의 음수값까지)
            nn.BatchNorm1d(500),           # normalize 통해 x,y축 기준 모두 고르게
            nn.Linear(500, 400),
            nn.LeakyReLU(),
            nn.BatchNorm1d(400),
            nn.Linear(400, 300),
            nn.LeakyReLU(),
            nn.BatchNorm1d(300),
            nn.Linear(300, 200),
            nn.LeakyReLU(),
            nn.BatchNorm1d(200),
            nn.Linear(200, 100),
            nn.LeakyReLU(),
            nn.BatchNorm1d(100),
            nn.Linear(100, 50),
            nn.LeakyReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, output_size),    # 마지막으로 50차원을 output 인 10으로 빼주고 
            nn.LogSoftmax(dim=-1),         # 텐서 (bs,hs) 중 hidden size 에 대해서 softmax 
            # 각 sample 별로 softmax 가 들어가야 하니 (dim=-1) ( 마지막 부분만)
        )

        # 위의 전체를 self.layers 라는 객체로 할당
        
    def forward(self, x):     # 단순히 x 라는 입력이 들어왔을 때 layer 에 집어넣기
        # |x| = (batch_size, input_size)

        y = self.layers(x)
        # |y| = (batch_size, output_size)

        return y
