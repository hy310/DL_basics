from copy import deepcopy

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

class Trainer():
    # 모델을 트레이닝 하는 trainer 클래스 선언

    def __init__(self, model, optimizer, crit):
        self.model = model
        self.optimizer = optimizer
        self.crit = crit

        super().__init__()

    def _train(self, x, y, config):

        
        self.model.train()  # 꼭 호출해주어야함

        # Shuffle before begin.
        # 시작하기 전에 데이터 셔플해야하는데, 데이터와 레이블값 함께 셔플되어야함
        # 인덱스 부여
        indices = torch.randperm(x.size(0), device=x.device)
        # x.size(0) = batch size, 0부터 batch size-1 까지 random 수열을 만들어라
        
        x = torch.index_select(x, dim=0, index=indices).split(config.batch_size, dim=0)
        # index_select 를 통해 인덱스별로 재배열 가능
        # x 라는 텐서를 어떤 디멘션에 대해 주어진 인덱스대로 하나씩 뽑아서 가져오는 것 
        # split 은 batch size 만큼 chunking 하여 미니배치 만드는것
        
        y = torch.index_select(y, dim=0, index=indices).split(config.batch_size, dim=0)
        
        # x, y 동시에 index_select.split 통해 x,y 쌍으로 가져갈 수 있음
        # 랜덤 셔플링 이후 미니배치 chunking 까지 완료
        # 매 에폭마다 이 세줄 실행

        total_loss = 0

        for i, (x_i, y_i) in enumerate(zip(x, y)):
            y_hat_i = self.model(x_i)
            loss_i = self.crit(y_hat_i, y_i.squeeze())
            # 위에서 crit 은 cross entropy 로 할당해놓음
            # y_i 는 (bs,1) 로 들어와있을까봐 (bs,) 로 squeeze 해주는 것
            # squeeze 해야지 (bs,) 로 되어서 cross entropy 에 넣을 수 있음

            # Initialize the gradients of the model.
            self.optimizer.zero_grad() #모델 optimizer 에 gradient 있으면 초기화시켜줌
            loss_i.backward()
            # loss 에 대해서 싹 back propagation

            self.optimizer.step()
            # step 을 진행하면 gradient descent one step 진행. weight 한번 업데이트됨.

            # verbose = 얼마나 수다스러울것인가. 현재 loss 찍어주기
            if config.verbose >= 2:
                print("Train Iteration(%d/%d): loss=%.4e" % (i + 1, len(x), float(loss_i)))

            # Don't forget to detach to prevent memory leak.
            # pytorch 는 연산할 때 즉시 computation graph 가 생성됨. 
     # 현재 loss_i 는 지금까지 사용된 computation graph 가 누적된 상황(따라서 back pro 가능)
            # float 를 안하고 물리면 또 total_loss tensor 로 저장되어서 모든 iteration 
            # 결과가 다 물리게 됨. 엄청난 메모리 손실
            total_loss += float(loss_i)

        return total_loss / len(x)

    def _validate(self, x, y, config):
        # Turn evaluation mode on.
        self.model.eval()     # 꼭 호출해주어야함
        # evaluation mode 로 안바꿔주면 성능 떨어짐.

        # Turn on the no_grad mode to make more efficintly.
        with torch.no_grad(): # gradient descent 필요없으니 꼭 no grad 해야 더 빠르게 진행
            # Shuffle before begin.
            indices = torch.randperm(x.size(0), device=x.device)
            x = torch.index_select(x, dim=0, index=indices).split(config.batch_size, dim=0)
            y = torch.index_select(y, dim=0, index=indices).split(config.batch_size, dim=0)

            total_loss = 0

            for i, (x_i, y_i) in enumerate(zip(x, y)):
                y_hat_i = self.model(x_i)
                loss_i = self.crit(y_hat_i, y_i.squeeze())

                if config.verbose >= 2:
                    print("Valid Iteration(%d/%d): loss=%.4e" % (i + 1, len(x), float(loss_i)))

                total_loss += float(loss_i)

            return total_loss / len(x)

    def train(self, train_data, valid_data, config):
        # train data 인 x 의 경우 2차원의 tensor 가 들어옴
        # l train data l = [ (bs, 784), (bs,1) ] 의 인풋과 레이블값 함께
        
        lowest_loss = np.inf  # 무한대로 초기화
        best_model = None

        for epoch_index in range(config.n_epochs):
            train_loss = self._train(train_data[0], train_data[1], config)
            # train_data[0] = 데이터, train_data[1] = 레이블
            # 각각의 epoch 에 대한 평균 loss
            valid_loss = self._validate(valid_data[0], valid_data[1], config)

            # You must use deep copy to take a snapshot of current best weights.
            if valid_loss <= lowest_loss:
                lowest_loss = valid_loss
                best_model = deepcopy(self.model.state_dict())

            print("Epoch(%d/%d): train_loss=%.4e  valid_loss=%.4e  lowest_loss=%.4e" % (
                epoch_index + 1,
                config.n_epochs,
                train_loss,
                valid_loss,
                lowest_loss,
            ))

        # Restore to best model.
        self.model.load_state_dict(best_model)
