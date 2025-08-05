import torch
import torch.nn as nn
import torch.fft as fft

import math


class FourierLayer(nn.Module):
    def __init__(self, input_size, num_components):
        super(FourierLayer, self).__init__()
        self.num_components = num_components
        # 使用凯明初始化方法初始化权重
        self.weights_real = nn.Parameter(torch.randn(input_size, num_components))
        self.weights_imag = nn.Parameter(torch.randn(input_size, num_components))

        nn.init.kaiming_normal_(self.weights_real, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.weights_imag, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        # Expand the last dimension to represent the imaginary part (set to zero)
        x_complex = torch.view_as_complex(torch.stack([x, torch.zeros_like(x)], dim=-1))

        # Apply Fourier transformation
        x_fft = fft.fft(x_complex)

        # Calculate real and imaginary parts of the Fourier coefficients
        real_part = torch.matmul(x_fft.real, self.weights_real) - torch.matmul(x_fft.imag, self.weights_imag)
        imag_part = torch.matmul(x_fft.real, self.weights_imag) + torch.matmul(x_fft.imag, self.weights_real)

        # Concatenate real and imaginary parts
        output = torch.cat([real_part, imag_part], dim=-1)
        return output


class CustomPeriodicSin(nn.Module):
    def __init__(self):
        super(CustomPeriodicSin, self).__init__()

    def forward(self, x, period):
        # Normalize input to the range [0, period]
        # x = torch.remainder(x, period)

        # Scale the input to the range [0, pi]
        scaled_x = 2 * (x / period) * math.pi

        # Apply sin function
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

        # Apply sin function
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
            Residual1(256, 512, use_1conv=True, strides=1),
            Residual1(512, 512, use_1conv=False, strides=1),

        )
        self.fc6 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            #   nn.Dropout(p=0.3),
            nn.Linear(256, 1),
            # nn.Softplus()

        )

    def forward(self, x):
        # Pass input through Fourier layer

        # Pass through fully connected layers
        output_NmF2 = self.fc1(x)
        output_NmF2 = self.fc2(output_NmF2)
        output_NmF2 = self.fc3(output_NmF2)
        output_NmF2 = self.fc4(output_NmF2)
        output_NmF2 = self.fc6(output_NmF2)

        return output_NmF2  # 4.8


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
            Residual2(256, 512, use_1conv=True, strides=2),
            Residual2(512, 512, use_1conv=False, strides=1)
        )
        self.fc6 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            #  nn.Dropout(p=0.3),
            nn.Linear(256, 1),
            # nn.Softplus()
        )

    def forward(self, x):
        output_HmF2 = self.fc1(x)
        output_HmF2 = self.fc2(output_HmF2)
        output_HmF2 = self.fc3(output_HmF2)
        output_HmF2 = self.fc4(output_HmF2)
        output_HmF2 = self.fc6(output_HmF2)

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
            Residual3(256, 512, use_1conv=True, strides=2),
            Residual3(512, 512, use_1conv=False, strides=1)
        )
        self.fc6 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            #    nn.Dropout(p=0.3),
            nn.Linear(256, 1),
            # nn.Softplus()

        )

    def forward(self, x):
        output_H0 = self.fc1(x)
        output_H0 = self.fc2(output_H0)
        output_H0 = self.fc3(output_H0)
        output_H0 = self.fc4(output_H0)
        output_H0 = self.fc6(output_H0)

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
            Residual4(256, 512, use_1conv=True, strides=2),
            Residual4(512, 512, use_1conv=False, strides=1)
        )
        self.fc6 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            #  nn.Dropout(p=0.3),
            nn.Linear(256, 1),
            # nn.Softplus()
        )

    def forward(self, x):
        output_Dhds = self.fc1(x)
        output_Dhds = self.fc2(output_Dhds)
        output_Dhds = self.fc3(output_Dhds)
        output_Dhds = self.fc4(output_Dhds)
        output_Dhds = self.fc6(output_Dhds)

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
            Residual5(256, 512, use_1conv=True, strides=2),
            Residual5(512, 512, use_1conv=False, strides=1)
        )
        self.fc6 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            #  nn.Dropout(p=0.3),
            nn.Linear(256, 1),
            # nn.Softplus()
        )

    def forward(self, x):
        output_Dhds = self.fc1(x)
        output_Dhds = self.fc2(output_Dhds)
        output_Dhds = self.fc3(output_Dhds)
        output_Dhds = self.fc4(output_Dhds)
        output_Dhds = self.fc6(output_Dhds)

        return output_Dhds


def NET_Ne_equation1(altitude: torch.Tensor, NmF2: torch.Tensor, hmF2: torch.Tensor, H0: torch.Tensor,
                     dHs_dh: torch.Tensor) -> torch.Tensor:
    """
    Combines 4 NET sub-models into a linear alpha-Chapman equation to get electron density.

    :param altitude: torch.Tensor
    :param NmF2: torch.Tensor
    :param hmF2: torch.Tensor
    :param H0: torch.Tensor
    :param dHs_dh: torch.Tensor

    :return: Ne (electron density in el./cm3).

    """

    Hs_lin = dHs_dh * (altitude - hmF2) + H0
    z = (altitude - hmF2) / Hs_lin
    Ne = NmF2 * torch.exp(0.5 * (1 - z - torch.exp(-z)))
    # NET works for the topside, and therefore predictions for which h<hmF2 are put to NaN:
    # mask_bottomside = (altitude < hmF2)
    # Ne[mask_bottomside] = float('nan')  # Using float('nan') for NaN in PyTorch
    return Ne

#
# input_size = 4  # Number of features (a, b, c)
# num_fourier_components = 3
#
# FFt = FourierLayer(input_size, num_fourier_components)
# x=torch.tensor([1, 2, 3, 4]).float()
# y = FFt(x)
# print(y.shape)
