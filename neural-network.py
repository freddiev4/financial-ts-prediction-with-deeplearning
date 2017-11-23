import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt


class FinancialTimeSeriesNetwork(nn.Module):

    def __init__(self):
        super(FinancialTimeSeriesNetwork, self).__init__()
        # TODO: define dimensions of each layer
        # TODO: figure out what type of model to use
        self.fc1 = nn.Linear()
        self.fc2 = nn.Linear()
        self.fc3 = nn.Linear()
        self.fc4 = nn.Linear()
        self.fc5 = nn.Linear()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return x


net = FinancialTimeSeriesNetwork()
print(net)

# TODO: define the input here
# using minutely prices
# along with the labels for the data

# Thinking that maybe I should use Zipline here for actually
# running a backtest & loading the data
# in which case I'd either have to use the Quandl bundle
# or create a custom bundle that loads IEX Trading data
trainset = None
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=4,
    shuffle=False,
    num_workers=2
)

testset = None
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=4,
    shuffle=False,
    num_workers=2
)


# use Stochastic Gradient Descent as the optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)
criterion = nn.Softmax()


# XXX: Somewhere in here should be the probabilistic strategy
# that is mentioned in the paper
# and the idea of trade-opening using soft-information


# in your training loop:
running_loss = []
for epoch in range(1):
    for i, data in enumerate(trainloader, 0):
        # XXX: I don't think we're actually using labels here
        # as it is time-series data
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the gradient buffers
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss.append(loss.data[0])
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

plt.plot(running_loss)
plt.title("Running Loss of the NN")


# TODO: check if the NN has actually learned anything below
