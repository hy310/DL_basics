import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from model import ImageClassifier
from trainer import Trainer
from utils import load_mnist

def define_argparser():
    ## 여러 개 인자 줄 수 있음
    
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)
    # 한 컴퓨터 내 gpu 여러개일 수 있으니 gpu_id 입력. cpu는 -1

    p.add_argument('--train_ratio', type=float, default=.8)

    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--n_epochs', type=int, default=20)
    p.add_argument('--verbose', type=int, default=2)
    # 2가 최대. 1이면 한 에폭 끝날때마다 찍힘. 2는 한 에폭 내에서 iteration 찍어줌

    config = p.parse_args()

    return config


def main(config):
    # Set device based on user defined configuration.
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)

    x, y = load_mnist(is_train=True)
    # Reshape tensor to chunk of 1-d vectors.
    x = x.view(x.size(0), -1)
    # view 를 하게 되면, 원래는 (lDl, 28, 28) 이 (lDl, 784) 로 flatten 될 것임

    train_cnt = int(x.size(0) * config.train_ratio)
    valid_cnt = x.size(0) - train_cnt

    # Shuffle dataset to split into train/valid set.
    indices = torch.randperm(x.size(0))
    x = torch.index_select(
        x,
        dim=0,
        index=indices
    ).to(device).split([train_cnt, valid_cnt], dim=0)
    y = torch.index_select(
        y,
        dim=0,
        index=indices
    ).to(device).split([train_cnt, valid_cnt], dim=0)

    print("Train:", x[0].shape, y[0].shape)
    print("Valid:", x[1].shape, y[1].shape)

    model = ImageClassifier(28**2, 10).to(device)
    optimizer = optim.Adam(model.parameters())
    crit = nn.NLLLoss()

    trainer = Trainer(model, optimizer, crit)

    trainer.train((x[0], y[0]), (x[1], y[1]), config)

    # Save best model weights.
    # 보통 모델 뿐만 아니라 config 도 함께 저장해줌
    
    torch.save({
        'model': trainer.model.state_dict(),
        # key, value 의 dictionary 값으로 저장해줌
        # pickle 형식처럼 저장할 수 있는 torch save
        'config': config,
    }, config.model_fn)

if __name__ == '__main__':
    config = define_argparser()
    main(config)
