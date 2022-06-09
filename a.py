import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import cupy as np


BATCH_SIZE = 128
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=BATCH_SIZE, shuffle=True)
'''


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(
    'data', train=True, download=True, transform=transform),
    batch_size=BATCH_SIZE,
    shuffle=True)

test_loader = torch.utils.data.DataLoader(datasets.CIFAR10(
    'data', train=False, transform=transform),
    batch_size=BATCH_SIZE,
    shuffle=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
           'ship', 'truck')

'''


def soft_max(x):
    return np.exp(x) / np.sum(np.exp(x))


class myLinear(nn.Linear):
    def __init__(self, n_input, n_out, K=8, lr=1e-1):
        super().__init__(n_input, n_out)
        limit = np.sqrt(6 / (n_input + n_out))
        lamb = 0.1
        self.weights_sets = np.random.uniform(-limit,
                                              limit, size=(n_out, n_input, K))
        self.bias_sets = np.random.uniform(-limit, limit, size=(n_out, K))
        self.weights_score = np.random.uniform(
            0, lamb*limit, size=(n_out, n_input, K))
        self.bias_score = np.random.uniform(0, lamb*limit, size=(n_out, K))
        self.bias_times = np.ones(shape=(n_out, K))
        self.weight_times = np.ones(shape=(n_out, n_input, K))
        self.total_times = K
        self.lr = lr
        self.K = K
        self.epsilon = 1e-2
        self.update_para()

    def update_score(self):
        self.weights_score -= self.lr * \
            np.repeat(np.expand_dims(np.asarray(self.weight.grad.cpu()), 2),
                      self.K, axis=2) * self.weights_sets
        self.bias_score -= self.lr * \
            np.repeat(np.expand_dims(np.asarray(self.bias.grad.cpu()), 1),
                      self.K, axis=1) * self.bias_sets

    def update_score2(self):
        temp = self.bias_sets * self.bias_indicator
        temp2 = self.weights_sets * self.weight_indicator
        self.weights_score -= self.lr * \
            np.repeat(np.expand_dims(np.asarray(self.weight.grad.cpu()), 2),
                      self.K, axis=2) * temp2
        self.bias_score -= self.lr * \
            np.repeat(np.expand_dims(np.asarray(self.bias.grad.cpu()), 1),
                      self.K, axis=1) * temp

    def update_para(self):
        self.bias_indicator = (self.bias_score == self.bias_score.max(
            axis=1, keepdims=1)).astype(float)
        temp = np.sum(self.bias_sets * self.bias_indicator, axis=1)
        self.bias.data = torch.as_tensor(
            temp.reshape(-1)).to(torch.float32).to(DEVICE)

        self.weight_indicator = (self.weights_score == self.weights_score.max(
            axis=2, keepdims=2)).astype(float)
        temp2 = np.sum(self.weights_sets * self.weight_indicator, axis=2)
        self.weight.data = torch.as_tensor(temp2).to(torch.float32).to(DEVICE)

    def update_para2(self):
        n_out, n_input, K = self.weights_sets.shape
        temp = np.zeros(n_out)
        for i in range(n_out):
            temp[i] = np.random.choice(
                self.bias_sets[i], p=soft_max(self.bias_score[i]))
        self.bias.data = torch.from_numpy(temp).to(torch.float32)
        temp2 = np.zeros((n_out, n_input))
        for i in range(n_out):
            for j in range(n_input):
                temp2[i][j] = np.random.choice(
                    self.weights_sets[i][j], p=soft_max(self.weights_score[i][j]))
        self.weight.data = torch.from_numpy(temp2).to(torch.float32)

    def update_para3(self):
        n_out, n_input, K = self.weights_sets.shape
        a = np.where(np.random.random(size=(n_out)) < self.epsilon, 1, 0)
        a = np.repeat(np.expand_dims(a, 1), self.K, axis=1)
        b = np.random.random(size=(n_out, K))
        b = (b == b.max(axis=1, keepdims=1)).astype(float)
        c = (self.bias_score == self.bias_score.max(
            axis=1, keepdims=1)).astype(float)
        self.bias_indicator = a*b+(1-a)*c
        temp = np.sum(self.bias_sets * self.bias_indicator, axis=1)
        self.bias.data = torch.from_numpy(temp.reshape(-1)).to(torch.float32)

        a = np.where(np.random.random(
            size=(n_out, n_input)) < self.epsilon, 1, 0)
        a = np.repeat(np.expand_dims(a, 2), self.K, axis=2)
        b = np.random.random(size=(n_out, n_input, K))
        b = (b == b.max(axis=2, keepdims=2)).astype(float)
        c = (self.weights_score == self.weights_score.max(
            axis=2, keepdims=2)).astype(float)
        self.weight_indicator = a*b+(1-a)*c
        temp2 = np.sum(self.weights_sets * self.weight_indicator, axis=2)
        self.weight.data = torch.from_numpy(temp2).to(torch.float32)

    def update_para4(self, alpha=1e-2):
        bias_UCB = self.bias_score+alpha * \
            np.sqrt(np.log(self.total_times)/self.bias_times)
        self.bias_indicator = (bias_UCB == bias_UCB.max(
            axis=1, keepdims=1)).astype(float)
        self.bias_times += self.bias_indicator
        temp = np.sum(self.bias_sets * self.bias_indicator, axis=1)
        self.bias.data = torch.from_numpy(temp.reshape(-1)).to(torch.float32)

        weight_UCB = self.weights_score+alpha * \
            np.sqrt(np.log(self.total_times)/self.weight_times)
        self.weight_indicator = (weight_UCB == weight_UCB.max(
            axis=2, keepdims=2)).astype(float)
        self.weight_times += self.weight_indicator
        temp2 = np.sum(self.weights_sets * self.weight_indicator, axis=2)
        self.weight.data = torch.from_numpy(temp2).to(torch.float32)

    def update_total(self):
        self.update_score2()
        self.update_para()
        self.total_times += 1


class myCnn(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, K=8, lr=1e-1):
        super().__init__(in_channels, out_channels, kernel_size)
        limit = np.sqrt(6 / (2 * kernel_size**2))
        self.weights_sets = np.random.uniform(-limit, limit, size=(
            out_channels, in_channels, kernel_size, kernel_size, K))
        self.bias_sets = np.random.uniform(-limit,
                                           limit, size=(out_channels, K))
        self.weights_score = np.random.uniform(0, limit, size=(
            out_channels, in_channels, kernel_size, kernel_size, K))
        self.bias_score = np.random.uniform(0, limit, size=(out_channels, K))
        self.bias_times = np.ones(shape=(out_channels, K))
        self.weight_times = np.ones(
            shape=(out_channels, in_channels, kernel_size, kernel_size, K))
        self.total_times = K
        self.lr = lr
        self.K = K
        self.epsilon = 1e-2
        self.update_para()

    def update_score(self):
        self.weights_score -= self.lr * \
            np.repeat(np.expand_dims(np.asarray(self.weight.grad.cpu()), 4),
                      self.K, axis=4) * self.weights_sets
        self.bias_score -= self.lr * \
            np.repeat(np.expand_dims(np.asarray(self.bias.grad.cpu()), 1),
                      self.K, axis=1) * self.bias_sets

    def update_score2(self):
        temp = self.bias_sets * self.bias_indicator
        temp2 = self.weights_sets * self.weight_indicator
        self.weights_score -= self.lr * \
            np.repeat(np.expand_dims(np.asarray(self.weight.grad.cpu()), 4),
                      self.K, axis=4) * temp2
        self.bias_score -= self.lr * \
            np.repeat(np.expand_dims(np.asarray(self.bias.grad.cpu()), 1),
                      self.K, axis=1) * temp

    def update_para(self):
        self.bias_indicator = (self.bias_score == self.bias_score.max(
            axis=1, keepdims=1)).astype(float)
        temp = np.sum(self.bias_sets * self.bias_indicator, axis=1)
        self.bias.data = torch.as_tensor(
            temp.reshape(-1)).to(torch.float32).to(DEVICE)

        self.weight_indicator = (self.weights_score == self.weights_score.max(
            axis=4, keepdims=4)).astype(float)
        temp2 = np.sum(self.weights_sets * self.weight_indicator, axis=4)
        self.weight.data = torch.as_tensor(temp2).to(torch.float32).to(DEVICE)

    def update_para3(self):
        out_channels, in_channels, kernel_size, _, K = self.weights_sets.shape
        a = np.where(np.random.random(
            size=(out_channels)) < self.epsilon, 1, 0)
        a = np.repeat(np.expand_dims(a, 1), self.K, axis=1)
        b = np.random.random(size=(out_channels, K))
        b = (b == b.max(axis=1, keepdims=1)).astype(float)
        c = (self.bias_score == self.bias_score.max(
            axis=1, keepdims=1)).astype(float)
        self.bias_indicator = a*b+(1-a)*c
        temp = np.sum(self.bias_sets * self.bias_indicator, axis=1)
        self.bias.data = torch.from_numpy(temp.reshape(-1)).to(torch.float32)

        a = np.where(np.random.random(size=(out_channels, in_channels,
                     kernel_size, kernel_size)) < self.epsilon, 1, 0)
        a = np.repeat(np.expand_dims(a, 4), self.K, axis=4)
        b = np.random.random(
            size=(out_channels, in_channels, kernel_size, kernel_size, K))
        b = (b == b.max(axis=4, keepdims=4)).astype(float)
        c = (self.weights_score == self.weights_score.max(
            axis=4, keepdims=4)).astype(float)
        self.weight_indicator = a*b+(1-a)*c
        temp2 = np.sum(self.weights_sets * self.weight_indicator, axis=4)
        self.weight.data = torch.from_numpy(temp2).to(torch.float32)

    def update_para4(self, alpha=1e-2):
        bias_UCB = self.bias_score+alpha * \
            np.sqrt(np.log(self.total_times)/self.bias_times)
        self.bias_indicator = (bias_UCB == bias_UCB.max(
            axis=1, keepdims=1)).astype(float)
        self.bias_times += self.bias_indicator
        temp = np.sum(self.bias_sets * self.bias_indicator, axis=1)
        self.bias.data = torch.from_numpy(temp.reshape(-1)).to(torch.float32)

        weight_UCB = self.weights_score+alpha * \
            np.sqrt(np.log(self.total_times)/self.weight_times)
        self.weight_indicator = (weight_UCB == weight_UCB.max(
            axis=4, keepdims=4)).astype(float)
        self.weight_times += self.weight_indicator
        temp2 = np.sum(self.weights_sets * self.weight_indicator, axis=4)
        self.weight.data = torch.from_numpy(temp2).to(torch.float32)

    def update_total(self):
        self.update_score2()
        self.update_para()
        self.total_times += 1


class LeNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = myCnn(1, 6, 5)
        self.conv2 = myCnn(6, 16, 5)
        self.fc1 = myLinear(256, 300)
        self.fc2 = myLinear(300, 100)
        self.fc3 = myLinear(100, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def update_para(self, train=1):
        for x in self.modules():
            if isinstance(x, myLinear) or isinstance(x, myCnn):
                if train:
                    x.update_total()
                else:
                    x.update_para()


class CONV2(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = myCnn(3, 64, 3)
        self.conv2 = myCnn(64, 64, 3)
        self.fc1 = myLinear(12544, 256)
        self.fc2 = myLinear(256, 256)
        self.fc3 = myLinear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5)
        x = self.fc2(x)
        x = F.dropout(x, p=0.5)
        x = self.fc3(x)
        return x

    def update_para(self, train=1):
        for x in self.modules():
            if isinstance(x, myLinear) or isinstance(x, myCnn):
                if train:
                    x.update_total()
                else:
                    x.update_para()


class CONV4(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = myCnn(3, 64, 3)
        self.conv2 = myCnn(64, 64, 3)
        self.conv3 = myCnn(64, 128, 3)
        self.conv4 = myCnn(128, 128, 3)
        self.fc1 = myLinear(3200, 256)
        self.fc2 = myLinear(256, 256)
        self.fc3 = myLinear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5)
        x = self.fc2(x)
        x = F.dropout(x, p=0.5)
        x = self.fc3(x)
        return x

    def update_para(self, train=1):
        for x in self.modules():
            if isinstance(x, myLinear) or isinstance(x, myCnn):
                if train:
                    x.update_total()
                else:
                    x.update_para()


model = LeNet().to(DEVICE)
#model = CONV2().to(DEVICE)
#model = CONV4().to(DEVICE)

loss_fun = torch.nn.CrossEntropyLoss()
n_epochs = 5
max_epoch = 10
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train(model, device, train_loader, optimizer, epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fun(output, target)
        loss.backward()
        model.update_para()
        '''
        if epoch < max_epoch:
            # model.update_para()
            model.update_para(loss.item())
        else:
            optimizer.step()
        '''
        if (batch_idx + 1) % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.update_para(train=0)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fun(output, target).item()
            _, pred = torch.max(output, dim=1)
            correct += (pred == target).sum().item()

    test_loss /= len(test_loader.dataset)
    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return correct / len(test_loader.dataset)


if __name__ == "__main__":
    result = []
    for epoch in range(1, EPOCHS + 1):
        train(model, DEVICE, train_loader, optimizer, epoch)
        result.append(test(model, DEVICE, test_loader))

    np.savetxt('6.txt', result)
