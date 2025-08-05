import os
import warnings
from matplotlib import cm
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

from resnet0305 import *
import datetime as dt
import torch
import numpy as np

def ne_profile_function(latobs, lonobs, ltobs, doyobs, f107obs, kpobs, symhobs, nmf2obs=np.nan, hmf2obs=np.nan):
    """
    （输入默认值为 np.nan，非 nan 时执行自定义操作）
    参数:

    返回:
        result: 处理后的结果（类型根据具体操作而定）
    """

    torch.manual_seed(66)

    # 判断是否有 CUDA 可用
    if torch.cuda.is_available():
        device = torch.device("cuda")  # 使用 GPU
    else:
        device = torch.device("cpu")  # 使用 CPU
    custom_periodic_sin = CustomPeriodicSin()
    custom_periodic_sin = custom_periodic_sin.to(device)

    custom_periodic_cos = CustomPeriodicCos()
    custom_periodic_cos = custom_periodic_cos.to(device)

    modelnmf2 = Nmf2(Residual1)
    modelnmf2 = modelnmf2.to(device)
    modelnmf2.load_state_dict(
        torch.load(r'models\train_nmf2.pth'))

    modela1 = Dhds(Residual4)
    modela1 = modela1.to(device)
    modela1.load_state_dict(
        torch.load(r'models\train_a1.pth'))
    modela2 = Dhds(Residual4)
    modela2 = modela2.to(device)
    modela2.load_state_dict(
        torch.load(r'models\train_a2.pth'))

    modelh0 = H0(Residual3)
    modelh0 = modelh0.to(device)

    modelh0.load_state_dict(
        torch.load(r'models\train_h0.pth'))

    modelhmf2 = Hmf2(Residual2)
    modelhmf2 = modelhmf2.to(device)
    modelhmf2.load_state_dict(
        torch.load(r'models\train_hmf2.pth'))

    modelnmf2.eval()
    modela1.eval()
    modelh0.eval()
    modelhmf2.eval()
    modela2.eval()

    doy_fixed = doyobs
    kp = kpobs
    f107 = f107obs
    symh = symhobs

    lt_fixed = ltobs


    input_scaler9 = torch.load(r'models\kp_scaler.pkl')
    input_scaler10 = torch.load(r'models\f107_scaler.pkl')
    input_scaler11 = torch.load(r'models\symh_scaler.pkl')

    Kp_fixed = torch.tensor(input_scaler9.transform(np.array([kp]).reshape(-1, 1)).flatten()).to(device)

    # 对 f107 使用 input_scaler10 进行缩放
    f107_fixed = torch.tensor(input_scaler10.transform(np.array([f107]).reshape(-1, 1)).flatten()).to(device)

    # 对 symh 使用 input_scaler11 进行缩放
    symh_fixed = torch.tensor(input_scaler11.transform(np.array([symh]).reshape(-1, 1)).flatten()).to(device)

    # 生成 x 数组
    lat = np.arange(latobs, latobs + 1, 1)
    # lat = lat.astype(float)
    # 生成 y 数组
    lon = np.arange(90, 800, 1)
    # lon = np.arange(450, 500, 4)
    # lon = lon.astype(float)

    lat_grid, lon_grid = np.meshgrid(lat, lon)

    lon_grid_flat = lon_grid.flatten()
    lat_grid_flat = lat_grid.flatten()
    data_grids = []

    for UT in np.arange(11, 24, 28):

        # 创建空列表用于存储 kp、f107 和 symh 张量
        kp_tensors = []
        f107_tensors = []
        symh_tensors = []
        DOY_tensors = []
        cosin_tensors = []
        # 遍历网格中的每个点，并为每个点创建一个张量
        for lat_point, lon_point in zip(lat_grid_flat, lon_grid_flat):

            longitude = lonobs  # 当前点的经度
            # LT = UT + longitude / 15  # 计算地方时
            # LT=12
            LT = lt_fixed

            if LT >= 24:
                LT -= 24
            if LT <= 0:
                LT += 24

            # fft_in_tensor = torch.tensor([doy_fixed, lt_fixed, lat_point, lon_point], dtype=torch.float32)
            # fft_in_tensors.append(fft_in_tensor)

            doy_sin = custom_periodic_sin(torch.tensor(doy_fixed), 365.25)
            lt_sin = custom_periodic_sin(torch.tensor(LT), 24)
            glon_sin = custom_periodic_sin(torch.tensor(longitude), 360)

            doy_cos = custom_periodic_cos(torch.tensor(doy_fixed), 365.25)
            lt_cos = custom_periodic_cos(torch.tensor(LT), 24)
            glon_cos = custom_periodic_cos(torch.tensor(longitude), 360)

            # 创建当前数据点的 kp、f107 和 symh 张量
            kp_tensor = torch.tensor(Kp_fixed, dtype=torch.float32)
            f107_tensor = torch.tensor(f107_fixed, dtype=torch.float32)
            symh_tensor = torch.tensor(symh_fixed, dtype=torch.float32)
            kp_tensors.append(kp_tensor)
            f107_tensors.append(f107_tensor)
            symh_tensors.append(symh_tensor)

            cos_tensor = torch.tensor(
                [doy_sin, lt_sin, glon_sin, doy_cos, lt_cos, glon_cos, lat_point / 90, kp_tensor, f107_tensor,
                 symh_tensor],
                dtype=torch.float32)
            cosin_tensors.append(cos_tensor)

        with torch.no_grad():
            #   g_lat = g_lat / 90

            # input_tensor = torch.stack([doy_sin, lt_sin, glon_sin, doy_cos, lt_cos, glon_cos, g_lat, Kp, f107, symh],
            #                            dim=1)

            input_tensor = torch.stack(cosin_tensors).to(device)

            input_tensor = input_tensor.view(input_tensor.size(0), 1, -1)
            nmf2 = modelnmf2(input_tensor)
            nmf2 = nmf2.view(-1)

            hmf2 = modelhmf2(input_tensor)
            hmf2 = hmf2.view(-1)

            h0 = modelh0(input_tensor)
            h0 = h0.view(-1)
            # fft_out4 = FFt4(fft_in)
            a1 = modela1(input_tensor)
            a1 = a1.view(-1)
            a2 = modela2(input_tensor)
            a2 = a2.view(-1)
            input_scaler7 = torch.load(r'models\a1_scaler.pkl')
            input_scaler4 = torch.load(r'models\nmf2_scaler.pkl')
            input_scaler5 = torch.load(r'models\hmf2_scaler.pkl')
            input_scaler6 = torch.load(r'models\h0_scaler.pkl')
            input_scaler12 = torch.load(r'models\a2_scaler.pkl')

            nmf2 = torch.tensor(input_scaler4.inverse_transform(nmf2.cpu().numpy().reshape(-1, 1)).flatten())
            # nmf2_net = torch.tensor(input_scaler4.inverse_transform(nmf2_R.cpu().numpy().reshape(-1, 1)).flatten())
            #
            hmf2 = torch.tensor(input_scaler5.inverse_transform(hmf2.cpu().numpy().reshape(-1, 1)).flatten())
            h0 = torch.tensor(input_scaler6.inverse_transform(h0.cpu().numpy().reshape(-1, 1)).flatten())
            dhds = torch.tensor(input_scaler7.inverse_transform(a1.cpu().numpy().reshape(-1, 1)).flatten())
            a2 = torch.tensor(input_scaler12.inverse_transform(a2.cpu().numpy().reshape(-1, 1)).flatten())

            nmf2 = np.exp(nmf2)
            #
            # print(nmf2[400])
            # nmf2 = np.log10(nmf2)

            tensor_shape = hmf2.size()

            # 获取张量的长度（第一个维度的大小）
            length = tensor_shape[0]
            Ne_model = torch.Tensor(0)
            # alt = torch.full((length,), 450)
            alt = lon_grid_flat

            if not np.isnan(hmf2obs):
                # 当 param2 不为 nan 时执行的操作
                hmf2 = torch.full((len(a1),), hmf2obs).flatten()
            else:
                processed_param2 = np.nan

            if not np.isnan(nmf2obs):
                # 当 param2 不为 nan 时执行的操作
                nmf2 = torch.full((len(a1),), nmf2obs).flatten()
            else:
                processed_param2 = np.nan

            # 遍历每个元素
            for element in range(length):
                if alt[element] < hmf2[element]:
                    new_data = nmf2[element] * torch.exp(1 / 2 * (1 - (alt[element] - hmf2[element]) / (
                            h0[element] + a2[element] * (alt[element] - hmf2[element])) - torch.exp(
                        -(alt[element] - hmf2[element]) / (
                                h0[element] + a2[element] * (alt[element] - hmf2[element])))))
                if alt[element] >= hmf2[element]:
                    new_data = nmf2[element] * torch.exp(1 / 2 * (1 - (alt[element] - hmf2[element]) / (
                            h0[element] + dhds[element] * (alt[element] - hmf2[element])) - torch.exp(
                        -(alt[element] - hmf2[element]) / (
                                h0[element] + dhds[element] * (alt[element] - hmf2[element])))))
                    # 向空数组中添加数据
                # new_data = new_data.unsqueeze(0)
                new_data = new_data.view(-1)
                Ne_model = torch.cat((Ne_model, new_data), dim=0)

            Ne_model = Ne_model * 1000000

    return Ne_model,alt

lat=20
lon=20
lt=12
doy = 103
kp = 0.7
f107 = 75.3
symh = -16

NEnow,alt=ne_profile_function(lat, lon, lt, doy, f107, kp, symh)

print(NEnow,alt)