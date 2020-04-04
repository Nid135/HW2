import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
import time

BATCH_SIZE = 128
NUM_EPOCHS = 10

# preprocessing
normalize = transforms.Normalize(mean=[.5], std=[.5])
transform = transforms.Compose([transforms.ToTensor(), normalize])

# download and load the data
train_dataset = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=transform, download=False)

# encapsulate them into dataloader form
train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

class SimpleNet(nn.Module):
# TODO:define model
    def __init__(self, ind, h1, h2, outd):
        super(SimpleNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(ind, h1),
            nn.BatchNorm1d(h1), nn.ReLU(True))
        self.layer2 = nn.Sequential(
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(h2, outd))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


#model = SimpleNet()
model = SimpleNet(28 * 28, 300, 100, 10)
model = model.cuda()

# TODO:define loss function and optimiter
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for images, labels in tqdm(train_loader):
    images = images.view(images.size(0), -1)
    images = images.cuda()
    labels = labels.cuda()

    out = model(images)
    loss = criterion(out, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval()
train=[]
test=[]
for epoch in range(NUM_EPOCHS):
    count = 0
    for images, labels in tqdm(train_loader):
        images = images.view(images.size(0), -1)
        images = images.cuda()
        labels = labels.cuda()

        out = model(images)
        _, predict = torch.max(out, 1)
        correct = (predict == labels).sum()
        count += correct.item()
    train.append(count / (len(train_dataset)))

    count = 0
    for images, labels in tqdm(test_loader):
        images = images.view(images.size(0), -1)
        images = images.cuda()
        labels = labels.cuda()

        out = model(images)
        _, predict = torch.max(out, 1)
        correct = (predict == labels).sum()
        count += correct.item()
    test.append(count / (len(test_dataset)))

print('train accuracy:',sum(train)/len(train))
print('test accuracy:',sum(test)/len(test))

