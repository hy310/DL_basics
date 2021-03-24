def load_mnist(is_train=True, flatten=True):
    from torchvision import datasets, transforms
    # torchvision 사용해서 간단히 MNIST 데이터셋 가져오기

    dataset = datasets.MNIST(
        '../data', train=is_train, download=True,
        # test 셋은 필요 없으니 train set 만 불러오기 (is_train)
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    )

    x = dataset.data.float() / 255.
    # 데이터 가져오면 0~255 정수 값으로 input 이 있기 때문에 255로 나눠주어 0~1 사이 값 갖도록
    y = dataset.targets

    if flatten:
        x = x.view(x.size(0), -1)
    # flatten 까지 해서 return 할 수 있도록

    return x, y

# 이게 일종의 data loader 가 되는 것임. 
# train.py 에서 load_mnist 부분이 data loader 에서 데이터 불러오는 부분