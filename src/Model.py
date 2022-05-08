import torch.nn as nn

class Model(nn.Module):
    batchSize = 50
    imgSize = (128, 32)
    maxTextLen = 32

    def __init__(self):
        super(Model, self).__init__()

        self.conv_pool_1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size=(5, 5), padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2,2),stride = (2,2), padding = 0)
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels = 8, out_channels = 64, kernel_size=(5, 5), padding = 2),
            nn.ReLU(),
        )
        
        self.conv_pool_BN_3 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size=(3, 3), padding = 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2,2),stride = (2,2), padding = 0)
        )

        self.conv_4 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size=(3, 3), padding = (1,2)),
            nn.ReLU(),
        )

        self.conv_pool_BN_5 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size=(3, 3), padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size = (2,2),stride = (2,2), padding = 0)
        )

        # self.conv_BN_6 = nn.Sequential(
        #     nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size=(3, 3), padding = 2),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        # )

        # self.conv_pool_7 = nn.Sequential(
        #     nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size=(5, 5), padding = 2),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size = (2,2),stride = (2,2), padding = 0)
        # )

        self.lstm = nn.LSTM(input_size = 32, hidden_size = 32, num_layers = 2, batch_first = True, bidirectional = True)
        self.logsoftmax = nn.LogSoftmax(dim = 2)

    def forward(self, x):
        # x = x.view(x.size(0), 1, x.size(1), x.size(2))
        x = self.conv_pool_1(x)
        x = self.conv_2(x)
        x = self.conv_pool_BN_3(x)
        x = self.conv_4(x)
        x = self.conv_pool_BN_5(x)
        # x = self.conv_BN_6(x)
        # x = self.conv_pool_7(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = x.permute(0, 2, 1)
        # print(x.shape[0])
        x, _ = self.lstm(x)
        x = x.permute(0, 2, 1)
        x = self.logsoftmax(x)
        return x
