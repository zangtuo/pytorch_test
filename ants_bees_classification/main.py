import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os
import torchvision
import visdom

data_transforms = {
    'train': transforms.Compose([
        # 随机切成224x224 大小图片 统一图片格式
        transforms.RandomResizedCrop(224),
        # 图像翻转
        transforms.RandomHorizontalFlip(),
        # to-tensor 归一化(0,255) >> (0,1)  normalize  channel=（channel-mean）/std
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        # 图片大小缩放 统一图片格式
        transforms.Resize(256),
        # 以中心裁剪
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

data_dir = './data_ants_bees'
BATCH_SIZE = 4
# trans data
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
# load data
data_loaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True) for x in ['train', 'val']}

data_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
print(data_sizes, class_names)
print(data_sizes['val'])

# inputs, classes = next(iter(data_loaders['train']))#查看图片
# viz = visdom.Visdom()
# out = torchvision.utils.make_grid(inputs)
# inp = torch.transpose(out, 0, 2)
# mean = torch.FloatTensor([0.485, 0.456, 0.406])
# std = torch.FloatTensor([0.229, 0.224, 0.225])
# inp = std * inp + mean
# inp = torch.transpose(inp, 0, 2)
# viz.images(inp)

model1 = torchvision.models.resnet18(pretrained=True)  # 预训练模型加载
num_ftrs = model1.fc.in_features
model1.fc = nn.Linear(num_ftrs, 2)
epoch = 10
LR = 0.001
train_step = 0
test_step = 0
loss_function = nn.CrossEntropyLoss()  # 交叉熵
optimizer = optim.SGD(model1.parameters(), lr=LR, momentum=0.9)  # 优化器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
for i in range(epoch):
    model1.train()
    print("-------------------第{}次训练开始------------------".format(i + 1))
    for data in data_loaders['train']:
        inputs, targets = data
        outputs = model1(inputs)
        loss = loss_function(outputs, targets)

        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 优化开始
        train_step = train_step + 1
        if train_step % 10 == 0:
            print("训练次数：{}，loss：{}".format(train_step, loss.item()))

    model1.eval()  # 测试
    test_loss = 0
    test_accuracy = 0

    with torch.no_grad():
        for data in data_loaders['val']:
            inputs, targets = data
            outputs = model1(inputs)
            loss = loss_function(outputs, targets)
            test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            test_accuracy += accuracy
    print("整体测试集的loss：{}".format(test_loss))
    print("整个测试的准确率：{}".format(test_accuracy/data_sizes['val']))
# 图片的处理
# data_transform = transforms.Compose([
#     transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化处理
# ])
# train_data = torchvision.datasets.ImageFolder(root="./data_ants_bees/train", transform=data_transform)
# test_data = torchvision.datasets.ImageFolder(root="./data_ants_bees/val", transform=data_transform)
# print(test_data.imgs)
