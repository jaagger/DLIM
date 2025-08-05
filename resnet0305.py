import torch
import torch.nn as nn
import torch.fft as fft
import math

class CustomPeriodicSin(nn.Module):
    def __init__(self):
        super(CustomPeriodicSin, self).__init__()

    def forward(self, x, period):
        # Normalize input to the range [0, period]
        # x = torch.remainder(x, period)
        # Scale the input to the range [0, pi]
        scaled_x = 2 * (x / period) * math.pi
        output = torch.sin(scaled_x)
        return output


class CustomPeriodicCos(nn.Module):
    def __init__(self):
        super(CustomPeriodicCos, self).__init__()

    def forward(self, x, period):
        # Normalize input to the range [0, period]
        # x = torch.remainder(x, period)
        # Scale the input to the range [0, pi]
        scaled_x = 2 * (x / period) * math.pi
        output = torch.cos(scaled_x)

        return output


class Residual1(nn.Module):
    def __init__(self, inputchannels, numchannels, use_1conv=False, strides=1):
        super(Residual1, self).__init__()
        self.Relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=inputchannels, out_channels=numchannels, kernel_size=3, stride=strides,
                               padding=1)
        self.bn1 = nn.BatchNorm1d(numchannels)
        self.conv2 = nn.Conv1d(in_channels=numchannels, out_channels=numchannels, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm1d(numchannels)
        if use_1conv:
            self.conv3 = nn.Conv1d(in_channels=inputchannels, out_channels=numchannels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.Relu(y)
        y = self.conv2(y)
        y = self.bn2(y)

        if self.conv3:
            x = self.conv3(x)

        y = self.Relu(y + x)
        return y


class Residual2(nn.Module):
    def __init__(self, inputchannels, numchannels, use_1conv=False, strides=1):
        super(Residual2, self).__init__()
        self.Relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=inputchannels, out_channels=numchannels, kernel_size=3, stride=strides,
                               padding=1)
        self.bn1 = nn.BatchNorm1d(numchannels)
        self.conv2 = nn.Conv1d(in_channels=numchannels, out_channels=numchannels, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm1d(numchannels)
        if use_1conv:
            self.conv3 = nn.Conv1d(in_channels=inputchannels, out_channels=numchannels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.Relu(y)
        y = self.conv2(y)
        y = self.bn2(y)

        if self.conv3:
            x = self.conv3(x)

        y = self.Relu(y + x)
        return y


class Residual3(nn.Module):
    def __init__(self, inputchannels, numchannels, use_1conv=False, strides=1):
        super(Residual3, self).__init__()
        self.Relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=inputchannels, out_channels=numchannels, kernel_size=3, stride=strides,
                               padding=1)
        self.bn1 = nn.BatchNorm1d(numchannels)
        self.conv2 = nn.Conv1d(in_channels=numchannels, out_channels=numchannels, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm1d(numchannels)
        if use_1conv:
            self.conv3 = nn.Conv1d(in_channels=inputchannels, out_channels=numchannels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.Relu(y)
        y = self.conv2(y)
        y = self.bn2(y)

        if self.conv3:
            x = self.conv3(x)

        y = self.Relu(y + x)
        return y


class Residual4(nn.Module):
    def __init__(self, inputchannels, numchannels, use_1conv=False, strides=1):
        super(Residual4, self).__init__()
        self.Relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=inputchannels, out_channels=numchannels, kernel_size=3, stride=strides,
                               padding=1)
        self.bn1 = nn.BatchNorm1d(numchannels)
        self.conv2 = nn.Conv1d(in_channels=numchannels, out_channels=numchannels, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm1d(numchannels)
        if use_1conv:
            self.conv3 = nn.Conv1d(in_channels=inputchannels, out_channels=numchannels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.Relu(y)
        y = self.conv2(y)
        y = self.bn2(y)

        if self.conv3:
            x = self.conv3(x)

        y = self.Relu(y + x)
        return y


class Residual5(nn.Module):
    def __init__(self, inputchannels, numchannels, use_1conv=False, strides=1):
        super(Residual5, self).__init__()
        self.Relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=inputchannels, out_channels=numchannels, kernel_size=3, stride=strides,
                               padding=1)
        self.bn1 = nn.BatchNorm1d(numchannels)
        self.conv2 = nn.Conv1d(in_channels=numchannels, out_channels=numchannels, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm1d(numchannels)
        if use_1conv:
            self.conv3 = nn.Conv1d(in_channels=inputchannels, out_channels=numchannels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.Relu(y)
        y = self.conv2(y)
        y = self.bn2(y)

        if self.conv3:
            x = self.conv3(x)

        y = self.Relu(y + x)
        return y


class Nmf2(nn.Module):
    def __init__(self, Residual1):
        super(Nmf2, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Conv1d(1, 64, 3, 1, 1),
            nn.BatchNorm1d(64),

            nn.LeakyReLU(),
            nn.MaxPool1d(2, 1)

        )
        self.fc2 = nn.Sequential(
            Residual1(64, 64, use_1conv=False, strides=1),
            Residual1(64, 64, use_1conv=False, strides=1),

        )
        self.fc3 = nn.Sequential(
            Residual1(64, 128, use_1conv=True, strides=1),
            Residual1(128, 128, use_1conv=False, strides=1),

        )
        self.fc4 = nn.Sequential(
            Residual1(128, 256, use_1conv=True, strides=2),
            Residual1(256, 256, use_1conv=False, strides=1),

        )

        self.fc5 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        output_NmF2 = self.fc1(x)
        output_NmF2 = self.fc2(output_NmF2)
        output_NmF2 = self.fc3(output_NmF2)
        output_NmF2 = self.fc4(output_NmF2)
        output_NmF2 = self.fc5(output_NmF2)

        return output_NmF2 


class Hmf2(nn.Module):
    def __init__(self, Residual2):
        super(Hmf2, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Conv1d(1, 64, 3, 1, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(2, 1)
        )
        self.fc2 = nn.Sequential(
            Residual2(64, 64, use_1conv=False, strides=1),
            Residual2(64, 64, use_1conv=False, strides=1)
        )
        self.fc3 = nn.Sequential(
            Residual2(64, 128, use_1conv=True, strides=1),
            Residual2(128, 128, use_1conv=False, strides=1)
        )
        self.fc4 = nn.Sequential(
            Residual2(128, 256, use_1conv=True, strides=2),
            Residual2(256, 256, use_1conv=False, strides=1)
        )
 
        self.fc5 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
         
            nn.Linear(256, 1),
        )

    def forward(self, x):
        output_HmF2 = self.fc1(x)
        output_HmF2 = self.fc2(output_HmF2)
        output_HmF2 = self.fc3(output_HmF2)
        output_HmF2 = self.fc4(output_HmF2)
        output_HmF2 = self.fc5(output_HmF2)

        return output_HmF2


class H0(nn.Module):
    def __init__(self, Residual3):
        super(H0, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Conv1d(1, 64, 3, 1, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(2, 1)
        )
        self.fc2 = nn.Sequential(
            Residual3(64, 64, use_1conv=False, strides=1),
            Residual3(64, 64, use_1conv=False, strides=1)
        )
        self.fc3 = nn.Sequential(
            Residual3(64, 128, use_1conv=True, strides=1),
            Residual3(128, 128, use_1conv=False, strides=1)
        )
        self.fc4 = nn.Sequential(
            Residual3(128, 256, use_1conv=True, strides=2),
            Residual3(256, 256, use_1conv=False, strides=1)
        )
        self.fc5 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        output_H0 = self.fc1(x)
        output_H0 = self.fc2(output_H0)
        output_H0 = self.fc3(output_H0)
        output_H0 = self.fc4(output_H0)
        output_H0 = self.fc5(output_H0)

        return output_H0


class Dhds(nn.Module):
    def __init__(self, Residual4):
        super(Dhds, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Conv1d(1, 64, 3, 1, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(2, 1)
        )
        self.fc2 = nn.Sequential(
            Residual4(64, 64, use_1conv=False, strides=1),
            Residual4(64, 64, use_1conv=False, strides=1)
        )
        self.fc3 = nn.Sequential(
            Residual4(64, 128, use_1conv=True, strides=1),
            Residual4(128, 128, use_1conv=False, strides=1)
        )
        self.fc4 = nn.Sequential(
            Residual4(128, 256, use_1conv=True, strides=2),
            Residual4(256, 256, use_1conv=False, strides=1)
        )

        self.fc5 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        output_Dhds = self.fc1(x)
        output_Dhds = self.fc2(output_Dhds)
        output_Dhds = self.fc3(output_Dhds)
        output_Dhds = self.fc4(output_Dhds)
        output_Dhds = self.fc5(output_Dhds)

        return output_Dhds


class A2(nn.Module):
    def __init__(self, Residual5):
        super(A2, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Conv1d(1, 64, 3, 1, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(2, 1)
        )
        self.fc2 = nn.Sequential(
            Residual5(64, 64, use_1conv=False, strides=1),
            Residual4(64, 64, use_1conv=False, strides=1)
        )
        self.fc3 = nn.Sequential(
            Residual5(64, 128, use_1conv=True, strides=1),
            Residual5(128, 128, use_1conv=False, strides=1)
        )
        self.fc4 = nn.Sequential(
            Residual5(128, 256, use_1conv=True, strides=2),
            Residual5(256, 256, use_1conv=False, strides=1)
        )

        self.fc5 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        output_A2 = self.fc1(x)
        output_A2 = self.fc2(output_Dhds)
        output_A2 = self.fc3(output_Dhds)
        output_A2 = self.fc4(output_Dhds)
        output_A2 = self.fc5(output_Dhds)
        return output_A2



