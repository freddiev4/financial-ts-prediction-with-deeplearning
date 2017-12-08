import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt


class FinancialTimeSeriesNetwork(nn.Module):

    def __init__(self):
        """
        Section 3.2; Figure 2 has an image of the architecture
        Section 5.2; implementation details of each layer

        Uses ReLU layers; in PyTorch terms this is equal to creating
        nn.Linear() layers, and calling .clamp() in the forward pass
        to compute the weights

        NOTE: .clamp() works for both Tensor and Variable
        """
        super(FinancialTimeSeriesNetwork, self).__init__()

        self.input_layer = nn.Linear(60, 500)
        self.m1 = nn.Linear(500, 200)
        self.m2 = nn.Linear(200, 40)
        self.m3 = nn.Linear(40, 20)
        self.output_layer = nn.Linear(20, 2)

    def forward(self, x):
        x = self.input_layer(x).clamp(min=0)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.m1(x).clamp(min=0)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.m2(x).clamp(min=0)
        x = self.m3(x).clamp(min=0)

        y_pred = self.output_layer(x).clamp(min=0)
        return y_pred


model = FinancialTimeSeriesNetwork()
print(model)

# Section 3.3 - Preprocessing
# Section 3.4 - Labeling the Dataset
# TODO: define the input here
# using minutely prices
# along with the labels for the data


# Thinking that maybe I should use Zipline here for actually
# running a backtest & loading the data
# in which case I'd either have to use the Quandl bundle
# or create a custom bundle that loads IEX Trading data

# Section 5; Experimental Results
# Test Set: June 23rd 2014 to June 22nd 2016
# Validation Set: June 23rd 2013 to June 22nd 2014
# Training Set: all previous data
trainset = None
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=100,  # Section 5.2; batch size of 100
    shuffle=False,
    num_workers=2
)

testset = None
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=100,  # Section 5.2; batch size of 100
    shuffle=False,
    num_workers=2
)

# "Our model does not account for changes in the market dynamics, and
# implicitly assumes that a strategy learnt using the data up to 2013,
# can be applied to trades in 2016."


# Section 5.2; initial learning rate of 0.001
# "We apply Stochastic Gradient Descent to train the models
# with 100 batch size and 0.5 dropout rate"
optimizer = optim.SGD(model.parameters(), lr=0.001)
criterion = nn.Softmax()


# XXX: Somewhere in here should be the probabilistic strategy
# that is mentioned in the paper
# and the idea of trade-opening using soft-information


# in your training loop:
running_loss = []
for epoch in range(1):
    for i, data in enumerate(trainloader, 0):
        # XXX: Section 3.4 & Section 5.1 talk about labeling the data
        # 60 Features; 1 Label
        # Each feature is the price at every minute for 60 minutes
        # Each label is the trend (-1 or 1)
        x, y = data
        x, y = Variable(x), Variable(y)

        y_pred = model(x)
        loss = criterion(y_pred, y)

        # zero gradients, perform a backward pass, and update the weights
        optimizer.zero_grad()
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
