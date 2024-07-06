import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super(ResidualBlock, self).__init__()
        self.convolution_layer_1 = nn.LazyConv2d(
            num_channels, kernel_size=3, padding=(1, 1), stride=strides
        )
        self.convolution_layer_2 = nn.LazyConv2d(
            num_channels, kernel_size=3, padding=(1, 1)
        )

        if use_1x1conv:
            self.convolution_layer_3 = nn.LazyConv2d(
                num_channels, kernel_size=1, stride=strides
            )
        else:
            self.convolution_layer_3 = None

        self.batch_norm_layer_1 = nn.LazyBatchNorm2d()
        self.batch_norm_layer_2 = nn.LazyBatchNorm2d()

    def forward(self, X):
        Y = F.relu(self.batch_norm_layer_1(self.convolution_layer_1(X)))
        Y = self.batch_norm_layer_2(self.convolution_layer_2(Y))
        if self.convolution_layer_3:
            X = self.convolution_layer_3(X)
        Y += X
        return F.relu(Y)


class ResNet(nn.Module):
    def __init__(self, shape):
        super(ResNet, self).__init__()
        self.input_shape = shape
        self.net = nn.Sequential(
            self.part_1(),  # conv and pooling
            self.part_2(),  # residual blocks
            self.part_3(),  # global pooling and dense layer
        )
        self._initialize_weights()
        self.to(try_gpu())

    def part_1(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            # nn.LazyConv2d(32, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

    def part_2(self):
        return nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(128, use_1x1conv=True, strides=2),
            ResidualBlock(128),
            ResidualBlock(256, use_1x1conv=True, strides=2),
            ResidualBlock(256),
            ResidualBlock(512, use_1x1conv=True, strides=2),
            ResidualBlock(512),
            #     ResidualBlock(32),
            #     ResidualBlock(32),
            #     ResidualBlock(64, use_1x1conv=True, strides=2),
            #     ResidualBlock(64),
            #     ResidualBlock(128, use_1x1conv=True, strides=2),
            #     ResidualBlock(128),
            #     ResidualBlock(256, use_1x1conv=True, strides=2),
            #     ResidualBlock(256),
        )

    def part_3(self):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LazyLinear(10),
            # nn.Softmax(dim=1), # can be removed if loss function is CrossEntropyLoss
        )

    def _initialize_weights(self):
        # Initialize weights by running a dummy forward pass
        dummy_input = torch.randn(*self.input_shape)
        _ = self.forward(dummy_input)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.LazyConv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LazyBatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Linear, nn.LazyLinear)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, X):
        return self.net(X)

    def train_model(self, train_iter, test_iter, num_epochs, lr, device):
        model = self
        model.train()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(num_epochs):
            train_loss, train_accuracy, n = 0.0, 0.0, 0
            for X, y in train_iter:
                X, y = X.to(device), y.to(device)
                y_hat = model(X)
                l = self.loss(y_hat, y)
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                train_loss += l
                train_accuracy += (y_hat.argmax(axis=1) == y).sum()
                n += y.size(0)
            test_accuracy = self.evaluate_accuracy(test_iter, model)
            print(
                f"epoch {epoch + 1}, "
                f"train loss {train_loss / n:.4f}, "
                f"train accuracy {train_accuracy / n:.3f}, "
                f"test accuracy {test_accuracy:.3f}"
            )

    def evaluate_accuracy(self, data_iter, net):
        net.eval()
        acc_sum, n = 0.0, 0
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            with torch.no_grad():
                y_hat = net(X)
            acc_sum += (y_hat.argmax(axis=1) == y).sum()
            n += y.size(0)
        net.train()
        return acc_sum / n

    def loss(self, y_hat, y):
        evaluator = nn.CrossEntropyLoss()
        return evaluator(y_hat, y)
