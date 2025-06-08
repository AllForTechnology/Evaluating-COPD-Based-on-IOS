import torch
import random
import platform
import copy
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm.notebook import tqdm  # for progress bar
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Type, Any, Callable, Union, List, Optional
from sklearn.metrics import confusion_matrix, roc_curve, auc
from scipy.signal import resample
import torch.nn.init as init
import math
import numpy as np
from Model import COPDModel
from myloss import FeatureAndClassWeightedBCELoss
from scipy.fft import fft
import pandas as pd
import matplotlib.pyplot as plt
import sys

system_name = platform.system()
model_num = 0


# 不做数据增强、不降采样、五折交叉  一维数据输入
class DemographicEmbed(nn.Module):
    def __init__(self, input_dim=4, embedding_dim=64, hidden_dim=32):
        super(DemographicEmbed, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # BatchNorm 层
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)

        return x


def conv1x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv1d:
    """1x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv1d:
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock1d(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock1d, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv1x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class IOSDataset(Dataset):
    def __init__(self, data, train_dataset=None, if_FEV1FVC=False):
        self.data = data
        self.flow_data = data['flow']
        self.pressure_data = data['pressure']
        self.vol_data = data['vol']
        self.label_data = copy.deepcopy(data['label1'])  # 先尝试2分类
        self.other_info = [[lst[20], lst[23], lst[26], lst[29], lst[32], lst[35]] for lst in data['info']]
        self.fill_nan()
        self.info_data = []
        self.if_FEV1FVC = if_FEV1FVC
        for lst, other in zip(data['info'], self.other_info):
            if lst[1] == '男':
                gender = 1
            else:
                gender = 0
            self.info_data.append(np.array([float(lst[4]), float(lst[2]), float(lst[5]), float(gender),
                                            float(other[0]), float(other[1]), float(other[2]), float(other[3]),
                                            float(other[4]), float(other[5])]))
        if self.if_FEV1FVC:
            self.FEV1FVC = []
            fev1fvc_buffer = data['info']
            for index in range(0, len(fev1fvc_buffer)):
                if isinstance(fev1fvc_buffer[index][12], float):  # 没做舒张实验
                    self.FEV1FVC.append(float(fev1fvc_buffer[index][11]))
                else:
                    self.FEV1FVC.append(float(fev1fvc_buffer[index][12]))
            self.FEV1FVC = torch.tensor(self.FEV1FVC, dtype=torch.float32)

        if train_dataset is None:
            self.scaler_scatter = StandardScaler()
            self.info_data = self.scaler_scatter.fit_transform(self.info_data)
        else:
            self.info_data = train_dataset.scaler_scatter.transform(self.info_data)

        print("The size of self.info_data is: ", len(self.info_data), len(self.info_data[0]))

        t = 30
        fs = 400
        windows_length = t * fs
        del_index = []
        for i in range(len(self.flow_data)):
            if len(self.flow_data[i]) < windows_length:
                del_index.append(i)

        for i in del_index[::-1]:
            del self.flow_data[i]
            del self.pressure_data[i]
            del self.vol_data[i]
            del self.label_data[i]
            del self.info_data[i]

        self.flow_data = [lst[:windows_length] for lst in self.flow_data]
        self.pressure_data = [lst[:windows_length] for lst in self.pressure_data]
        self.vol_data = [lst[:windows_length] for lst in self.vol_data]

        self.flow_data_tensor = torch.stack([torch.tensor(seq, dtype=torch.float32) for seq in self.flow_data])
        self.pressure_data_tensor = torch.stack([torch.tensor(seq, dtype=torch.float32) for seq in self.pressure_data])
        self.vol_data_tensor = torch.stack([torch.tensor(seq, dtype=torch.float32) for seq in self.vol_data])

        self.data_cat_tensor = torch.cat((self.flow_data_tensor.unsqueeze(2),
                                          self.pressure_data_tensor.unsqueeze(2),
                                          self.vol_data_tensor.unsqueeze(2)), dim=2)
        self.label_data_tensor = torch.tensor(self.label_data, dtype=torch.long)
        # self.info_data_tensor = torch.stack([torch.tensor(seq, dtype=torch.float32) for seq in self.info_data])
        self.info_data_tensor = torch.tensor(self.info_data, dtype=torch.float32)

    def fill_nan(self):
        standard = []
        for i in self.other_info:
            buffer = []
            for j in i:
                buffer.append(float(j))
            standard.append(buffer)
        standard = np.array(standard).T
        for i in range(len(standard)):
            mean_value = np.nanmean(standard[i])
            standard[i] = np.where(np.isnan(standard[i]), mean_value, standard[i])
        self.other_info = standard.T.tolist()

    def __len__(self):
        return len(self.flow_data)

    def __getitem__(self, idx):
        if self.if_FEV1FVC:
            return self.data_cat_tensor[idx], self.label_data_tensor[idx], self.info_data_tensor[idx], self.FEV1FVC[idx]
        else:
            return self.data_cat_tensor[idx], self.label_data_tensor[idx], self.info_data_tensor[idx]


def calculate_kappa(TN, TP, FN, FP):
    N = TN + TP + FN + FP
    p_o = (TP + TN) / N
    p1 = (TP + FN) / N
    p2 = (FP + TN) / N
    p1_hat = (TP + FP) / N
    p2_hat = (TN + FN) / N
    p_e = p1 * p1_hat + p2 * p2_hat
    kappa = (p_o - p_e) / (1 - p_e)
    return kappa


class ResNet1d(nn.Module):
    def __init__(
            self,
            block: Type[Union[BasicBlock1d]],
            layers: List[int],
            num_lead: int = 12,
            backbone_out_dim: int = 512,
            dropout_rate=0.5,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet1d, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv1d(num_lead, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512 * block.expansion, backbone_out_dim)
        self.softmax = nn.Softmax(dim=-1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            pass

    def _make_layer(self, block: Type[Union[BasicBlock1d]],
                    planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.dropout(x)
        x = self.fc(x)
        # x=self.softmax(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x.transpose(-1, -2))


class demographic_model(nn.Module):
    def __init__(self, configs):
        super(demographic_model, self).__init__()
        self.convmodel = ResNet1d(BasicBlock1d, [2, 2, 2, 2], num_lead=configs.num_lead, backbone_out_dim=512,
                                  dropout_rate=configs.dropout_rate)
        self.dgmodel = DemographicEmbed(configs.info_input_dim, configs.embedding_dim, configs.hidden_dim)

        kernel_size = int(512 / configs.embedding_dim)
        self.downsample = nn.MaxPool1d(kernel_size=kernel_size, stride=kernel_size)
        self.fc = nn.Linear(2 * configs.embedding_dim, configs.output_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc1 = nn.Linear(32, 2, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, data, info):
        output_data = self.convmodel(data)

        output_info = self.dgmodel(info)

        output_data = self.downsample(output_data)
        output = torch.cat((output_data, output_info), dim=1)
        output = self.fc(output)

        return output


class pdConfigs:
    def __init__(self):
        self.num_lead = 3
        self.dropout_rate = 0.5
        self.info_input_dim = 10
        self.embedding_dim = 64
        self.hidden_dim = 32
        self.output_dim = 2


def initialize_weights(model):
    """Initialize the weights of the model."""
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def train_and_validate(model, train_loader, val_loader, epochs, optimizer, criterion, device='cpu', run=None):
    global model_num
    initialize_weights(model)

    # 用于记录每个 epoch 的训练和验证损失及准确率
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    TP_list, FP_list, TN_list, FN_list = [], [], [], []
    Recall_list, Precision_list, F1_list = [], [], []
    kappa_list = []

    model.to(device)
    softmax = nn.Softmax(dim=-1)
    # 检查是否有可用的GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    for epoch in tqdm(range(epochs), desc='Training Progress'):
        # 训练阶段
        model.train()
        total_train_loss, correct_train, total_train = 0, 0, 0
        for batch_idx, (data, label, info, fev1fvc) in enumerate(train_loader):
            data, label, info = data.to(device), label.to(device), info.to(device)
            fev1fvc = fev1fvc.to(device)
            optimizer.zero_grad()
            output = model(data, info)
            loss = criterion(output, label, fev1fvc)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct_train += (predicted == label).sum().item()
            total_train += label.size(0)

        train_loss = total_train_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        total_val_loss, correct_val, total_val = 0, 0, 0
        TP, FP, TN, FN = [], [], [], []
        label_list, pred_list = [], []

        with torch.no_grad():
            for data, label, info, fev1fvc in val_loader:
                data, label, info = data.to(device), label.to(device), info.to(device)
                fev1fvc = fev1fvc.to(device)
                output = model(data, info)
                loss = criterion(output, label, fev1fvc)
                total_val_loss += loss.item()
                _, predicted = torch.max(output, 1)
                correct_val += (predicted == label).sum().item()
                cm = confusion_matrix(label.cpu(), predicted.cpu())
                if len(cm) == 1:
                    cm = np.array([[label.shape[0], 0], [0, 0]])
                TP.append(cm[1][1])
                FP.append(cm[0][1])
                TN.append(cm[0][0])
                FN.append(cm[1][0])
                label_list.extend(label.cpu().numpy())
                pred_list.extend(softmax(output.cpu()).numpy()[:, 1])
                total_val += label.size(0)

        TP, FP, TN, FN = sum(TP), sum(FP), sum(TN), sum(FN)
        print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
        TP_list.append(TP)
        FP_list.append(FP)
        TN_list.append(TN)
        FN_list.append(FN)
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        F1 = 2 * Precision * Recall / (Precision + Recall)
        kappa = calculate_kappa(TN, TP, FN, FP)

        Precision_list.append(Precision if ~np.isnan(Precision) else 0)
        Recall_list.append(Recall if ~np.isnan(Recall) else 0)
        F1_list.append(F1 if ~np.isnan(F1) else 0)
        kappa_list.append(kappa)
        print(f"Precision: {Precision:.4f}, Recall: {Recall:.4f}, F1: {F1:.4f}, Kappa: {kappa:.4f}, ")
        # 计算并保存验证集的平均损失和准确率
        val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # 打印每个 epoch 的损失和准确率
        print(f"Epoch [{epoch + 1}/{epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # 返回训练和验证的损失和准确率列表
    model_num += 1

    return (train_losses, train_accuracies, val_losses, val_accuracies,
            TP_list, FP_list, TN_list, FN_list, Precision_list, Recall_list, F1_list, kappa_list)


def data_preprocessing(data, n_split=1):
    flow_data = data['flow']
    pressure_data = data['pressure']
    vol_data = data['vol']
    label_data = copy.deepcopy(data['label1'])  # 先尝试2分类
    info_data = copy.deepcopy(data['info'])

    t = 30
    fs = 400
    windows_length = t * fs

    del_index = []
    for i in range(len(flow_data)):
        if len(flow_data[i]) < windows_length:
            del_index.append(i)

    for i in del_index[::-1]:
        del flow_data[i]
        del pressure_data[i]
        del vol_data[i]
        del label_data[i]
        del info_data[i]

    split = StratifiedShuffleSplit(n_splits=n_split, test_size=0.2, random_state=42)
    train_data = {'flow': [], 'pressure': [], 'vol': [], 'label1': [], 'info': []}
    test_data = {'flow': [], 'pressure': [], 'vol': [], 'label1': [], 'info': []}
    for train_index, test_index in split.split(flow_data, label_data):
        train_data['flow'].append([flow_data[i] for i in train_index])
        train_data['pressure'].append([pressure_data[i] for i in train_index])
        train_data['vol'].append([vol_data[i] for i in train_index])
        train_data['label1'].append([label_data[i] for i in train_index])
        train_data['info'].append([info_data[i] for i in train_index])

        test_data['flow'].append([flow_data[i] for i in test_index])
        test_data['pressure'].append([pressure_data[i] for i in test_index])
        test_data['vol'].append([vol_data[i] for i in test_index])
        test_data['label1'].append([label_data[i] for i in test_index])
        test_data['info'].append([info_data[i] for i in test_index])

    return train_data, test_data


def collate_fn(batch, is_train=False):
    data, label, info, fev1fvc = zip(*batch)

    data = torch.stack(data)
    label = torch.stack(label)
    info = torch.stack(info)
    fev1fvc = torch.stack(fev1fvc)

    if is_train:
        data = data_augmentation(data)

    return data, label, info, fev1fvc


def data_augmentation(cat_data, noise_factor=0.01, scaling_factor=0.1):
    flow = cat_data[:, :, 0]
    pressure = cat_data[:, :, 1]
    vol = cat_data[:, :, 2]

    # 添加噪声
    noise_factor_flow = noise_factor * 4 * torch.randn_like(flow)
    flow = flow + noise_factor_flow

    noise_factor_pressure = noise_factor * 0.8 * torch.randn_like(pressure)
    pressure = pressure + noise_factor_pressure

    noise_factor_vol = noise_factor * 2 * torch.randn_like(vol)
    vol = vol + noise_factor_vol

    # 随机缩放
    scaling_factor = 1 + scaling_factor * torch.randn(flow.size(0), 1)
    scaling_factor = scaling_factor.repeat(1, flow.size(1))
    flow *= scaling_factor
    pressure *= scaling_factor
    vol *= scaling_factor

    cat_data[:, :, 0] = flow
    cat_data[:, :, 1] = pressure
    cat_data[:, :, 2] = vol

    # 随机平移
    shifts = torch.randint(-5, 5, (cat_data.size(0),))
    for i in range(cat_data.size(0)):
        cat_data[i] = torch.roll(cat_data[i], shifts=shifts[i].item(), dims=0)  # dims=0 是第 1 维度（长度）

    return cat_data


def start(train_data, test_data, run=None):
    train_dataset = IOSDataset(train_data, if_FEV1FVC=True)
    test_dataset = IOSDataset(test_data, if_FEV1FVC=True)

    batch_size = 16
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
        drop_last=True,
        # 加入数据增强-----------------------------------------
        collate_fn=lambda batch: collate_fn(batch, is_train=True)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=lambda batch: collate_fn(batch, is_train=False)
    )

    torch.cuda.empty_cache()
    model = COPDModel()
    weight_for_class1 = 6.0
    weight_for_class0 = 1.0
    loss = FeatureAndClassWeightedBCELoss(weight_positive=weight_for_class1, weight_negative=weight_for_class0)

    # 定义优化器
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loss, train_accuracy, val_loss, val_accuracy, TP, FP, TN, FN, Precision, Recall, F1, kappa = train_and_validate(
        model, train_loader, test_loader,
        epochs=600, optimizer=optimizer,
        criterion=loss, device=device, run=run)

    return train_loss, train_accuracy, val_loss, val_accuracy, TP, FP, TN, FN, Precision, Recall, F1, kappa


def center_control():
    data = np.load(r'dataset/dataset1.npy', allow_pickle=True).item()

    train_data_5fold, test_data_5fold = data_preprocessing(data, n_split=5)

    train_data = {}
    test_data = {}

    for i in range(len(train_data_5fold['flow'])):
        train_data = {
            'flow': train_data_5fold['flow'][i],
            'pressure': train_data_5fold['pressure'][i],
            'vol': train_data_5fold['vol'][i],
            'label1': train_data_5fold['label1'][i],
            'info': train_data_5fold['info'][i]
        }
        test_data = {
            'flow': test_data_5fold['flow'][i],
            'pressure': test_data_5fold['pressure'][i],
            'vol': test_data_5fold['vol'][i],
            'label1': test_data_5fold['label1'][i],
            'info': test_data_5fold['info'][i]
        }
        (train_loss, train_accuracy, test_loss, test_accuracy,
         TP, FP, TN, FN, Precision, Recall, F1, kappa) = start(train_data, test_data)


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    center_control()
