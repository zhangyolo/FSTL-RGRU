# _*_ coding : utf-8  _*_
# @Time : 2024-08-01  21:59
# @Author :  yolo_ZYM
# @File :  gaigru1
# @Project : multheaddaypred
# _*_ coding : utf-8  _*_
# @Time : 2024-05-14  16:54
# @Author :  yolo_ZYM
# @File :  gru_
# @Project : seq2seq
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset

# 随机数种子
from uitls.metrics import metric

np.random.seed(0)
'''
单变量预测！！

用camels数据据进行训练

'''


device = torch.device("cuda:0" )
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,pre_len):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size ,1)  # 乘以2是因为使用了双向LSTM
        self.pre_len=pre_len
    def forward(self, x):
        # x 的形状应该是 (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        # lstm_out 的形状是 (batch_size, sequence_length, hidden_size * 2) 因为是双向LSTM
        output = self.fc(lstm_out[:,-pre_len :, :])  # 取最后一个时刻的输出
        return output



class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,pre_len):
        super(GRU, self).__init__()
        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size ,1)  # 乘以2是因为使用了双向LSTM
        self.pre_len=pre_len
    def forward(self, x):
        # x 的形状应该是 (batch_size, sequence_length, input_size)
        lstm_out, _ = self.gru(x)
        # lstm_out 的形状是 (batch_size, sequence_length, hidden_size * 2) 因为是双向LSTM
        output = self.fc(lstm_out[:,-pre_len :, :])  # 取最后一个时刻的输出
        return output

class TimeSeriesDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):

        sequence, label = self.sequences[index]
        # sequence = torch.tensor(sequence, dtype=torch.float32)
        # label = torch.tensor(label, dtype=torch.float32)
        return sequence,label

# 初始化模型、损失函数和优化器

def calculate_mae(y_true, y_pred):
    # 平均绝对误差
    # mae = np.mean(np.abs(y_true - y_pred))
    y_true = y_true.detach().cpu().numpy() if y_true.is_cuda else y_true.numpy()
    y_pred = y_pred.detach().cpu().numpy() if y_pred.is_cuda else y_pred.numpy()

    # 计算MAE
    mae = np.mean(np.abs(y_true - y_pred))
    return mae
import numpy as np
import torch


def NSE(sim_tensor, obs_tensor):
    # pred = torch.from_numpy(pred)
    # y = torch.from_numpy(y)
    # # pred = pred.detach().cpu() if pred.is_cuda else pred.numpy()
    # # y = y.detach().cpu() if y.is_cuda else y.numpy()
    #
    # return 1 - (torch.sum((torch.square(y - pred))) / torch.sum(torch.square(y - torch.mean(y) * torch.ones(y.shape))))

    import torch

    # 假设sim_tensor是模拟值的张量，obs_tensor是观测值的张量
    # 确保它们具有相同的形状
    assert sim_tensor.shape == obs_tensor.shape

    # 计算观测值的平均值
    obs_mean = torch.mean(obs_tensor)

    # 计算模拟值与观测值的差值
    diff = sim_tensor - obs_tensor

    # 计算差值的平方
    diff_squared = diff.pow(2)

    # 计算差值的平方和
    sse = torch.sum(diff_squared)

    # 计算观测值的方差
    var_obs = torch.sum((obs_tensor - obs_mean).pow(2))

    # 计算NSE
    nse = 1 - (sse / var_obs)

    # print(nse.item())  # 打印NSE值，.item()将0维张量转换为Python数字
    return nse

 


from statsmodels.tsa.seasonal import STL


def NSE1(pred, y):
    pred = torch.from_numpy(pred)
    y = torch.from_numpy(y)
    return 1 - (torch.sum((torch.square(y - pred))) / torch.sum(torch.square(y - torch.mean(y) * torch.ones(y.shape))))
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path=id, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        quend = "_qu.pt"
        jiend = "_ji.pt"

        self.ji = "{}{}".format(self.path, jiend)
        self.qu = "{}{}".format(self.path, quend)
    def __call__(self, val_loss, model,model1):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model,model1)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model,model1)
            self.counter = 0

    def save_checkpoint(self, val_loss, model,model1):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.ji)
        # torch.save(model1.state_dict(), self.qu)
        self.val_loss_min = val_loss
# wittern and saved in utils.py

import torch
class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """
    def __init__(
        self, optimizer, patience=3, min_lr=1e-6, factor=0.5
    ):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
                verbose=True
            )
    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)

def create_inout_sequences(input_data, tw, pre_len):
    # 创建时间序列数据专用的数据分割器
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        if (i + tw + pre_len) > len(input_data):
            break
        train_label = input_data[i + tw:i + tw + pre_len]
        inout_seq.append((train_seq, train_label))
    return inout_seq

import torch
import torch.nn as nn

import torch
import torch.nn as nn

# 源文件夹和目标文件夹路径
source_folder = r'D:\Code\pytoch_code\multheaddaypred\data2\14'
target_folder =r'D:\Code\pytoch_code\multheaddaypred\data_pre\14'
import  os
# # 确保目标文件夹存在
# if not os.path.exists(target_folder):
#     os.makedirs(target_folder)

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_folder):
    if filename.endswith('.csv'):  # 只处理CSV文件
        file_path = os.path.join(source_folder, filename)

        # 读取CSV文件

        data = pd.read_csv(file_path, header=None)  # 假设第一行是列名
        id=filename[:8]
        """
        数据定义部分
        """
        Train = True  # 训练还是预测
        pre_len = 1  # 预测未来数据的长度
        csv_file = "D:\\Code\\pytoch_code\\multheaddaypred\\metrics_gruGAI2-1.csv"
        begin2="D:\\Code\\pytoch_code\\multheaddaypred\\data2\\14\\"
        end2="_streamflow_clear.csv"

        full_path2 = "{}{}{}".format(begin2, id,end2)
        true_data = pd.read_csv(full_path2)  # 填你自己的数据地址,自动选取你最后一列数据为特征列

        quend = "_qu.pt"
        jiend = "_ji.pt"

        ji = "{}{}".format(id, jiend)
        qu = "{}{}".format(id, quend)

        test_size = 0.15  # 训练集和测试集的尺寸划分
        train_size = 0.70  # 训练集和测试集的尺寸划分
        valid_size=0.15

        train_window = 10  # 观测窗口
        epochs = 100
        # 10个epoch还没有到收敛的时候

        # early stopping patience; how long to wait after last time validation loss improved.
        patience = 10
        patience1=3
        # 定义标准化优化器
        scaler_train = MinMaxScaler(feature_range=(0, 1))
        scaler_test = MinMaxScaler(feature_range=(0, 1))

        # 训练集和测试集划分
        train_data = true_data[:int(train_size * len(true_data))]
        valid_data=true_data[int(train_size * len(true_data)):int(train_size * len(true_data))+int(valid_size * len(true_data))]
        test_data = true_data[-int(test_size * len(true_data)):]

        print(len(true_data))
        print("训练集尺寸:", len(train_data))

        print("验证集尺寸:", len(valid_data))
        print("测试集尺寸:", len(test_data))

        # # 进行标准化处理
        train_data_normalized = scaler_train.fit_transform(train_data)
        valid_data_normalized = scaler_test.fit_transform(valid_data)
        test_data_normalized = scaler_test.fit_transform(test_data)

        # train_data_normalized=train_data.values
        # test_data_normalized=test_data.values
        # 转化为深度学习模型需要的类型Tensor

        train_data_normalized = torch.FloatTensor(train_data_normalized)
        valid_data_normalized = torch.FloatTensor(valid_data_normalized)
        test_data_normalized = torch.FloatTensor(test_data_normalized)


        # 定义训练器的的输入
        train_inout_seq = create_inout_sequences(train_data_normalized, train_window, pre_len)
        valid_inout_seq = create_inout_sequences(valid_data_normalized, train_window, pre_len)
        test_inout_seq = create_inout_sequences(test_data_normalized, train_window, pre_len)

        # 创建数据集
        train_dataset = TimeSeriesDataset(train_inout_seq)
        valid_dataset = TimeSeriesDataset(valid_inout_seq)
        test_dataset = TimeSeriesDataset(test_inout_seq)

        # 创建 DataLoader
        batch_size = 32  # 你可以根据需要调整批量大小
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)


        gru = GRU(input_size=1, num_layers=1, hidden_size=128, pre_len=pre_len)
        # 将模型中的所有参数转换为Double类型
        gru.to(device)


        qugru =GRU(input_size=1,num_layers=1, hidden_size=train_window, pre_len=pre_len)
        qugru.to(device)
        loss_function = nn.MSELoss()
        loss_function.to(device)
        optimizer = torch.optim.Adam(gru.parameters(), lr=0.005)
        optimizer1 = torch.optim.Adam(qugru.parameters(), lr=0.005)


        if Train:
            results = []
            reals = []
            gru.train()  # 训练模式
            losss = []
            lossnse = []

            # to track the training loss as the model trains
            train_losses = []
            # to track the validation loss as the model trains
            valid_losses = []
            # to track the average training loss per epoch as the model trains
            avg_train_losses = []
            # to track the average validation loss per epoch as the model trains
            avg_valid_losses = []

            # initialize the early_stopping object
            early_stopping = EarlyStopping(patience=patience, verbose=True,path=id)
            reduce_lr = LRScheduler(
                optimizer,
                factor=0.5,
                patience=patience1
            )

            for k in range(1,1+epochs):
                loss = 0
                losnse=0
                gru.train()
                start_time = time.time()  # 计算起始时间
                for seq, labels in train_loader:
                    optimizer.zero_grad()
                    optimizer1.zero_grad()
                    seq=seq.to(device)
                    labels=labels.to(device)

                    y_pred=gru(seq)

                    # exit(0)
                    single_loss = loss_function(y_pred, labels)

                    single_loss.backward()
                    train_losses.append(single_loss.item())

                    optimizer.step()
                    optimizer1.zero_grad()

                    # print(f'epoch: {k:3} loss: {single_loss.item():10.8f}')
                    loss+=single_loss.item()
                    nse = NSE(y_pred, labels)
                    losnse+=nse.item()
                    # losnse.append(nse.detach().numpy())
                    # tensor(0.9728)  这个结果说明，预测结果好哇

                    print(nse)

                losss.append(loss / len(train_loader))
                print("myloss:")
                print(loss / len(train_loader))
                lossnse.append(losnse / len(train_loader))
                print(losnse)
                print(f'epoch: {k:3} loss: {loss / len(train_loader):10.8f}')
                # torch.save(gru.state_dict(), '{}.pth'.format(id))
                # torch.save(qugru.state_dict(), 'lstmmodel.pth')
                print(f"模型已保存,用时:{(time.time() - start_time) / 60:.4f} min")

                gru.eval()
                for seq,labels in valid_loader:
                    seq=seq.to(device)
                    labels = labels.to(device)
                    a, b, c = seq.shape
                    y_pred = gru(seq)

                    # exit(0)
                    single_loss = loss_function(y_pred, labels)
                    valid_losses.append(single_loss.item())

                # print training/validation statistics
                # calculate average loss over an epoch
                train_loss = np.average(train_losses)
                valid_loss = np.average(valid_losses)
                avg_train_losses.append(train_loss)
                avg_valid_losses.append(valid_loss)

                epoch_len = epochs

                # 查看当前优化器的学习率
                print('Initial learning rate:', optimizer.defaults['lr'])
                # 查看当前学习率
                for param_group in optimizer.param_groups:
                    print(f"Current learning rate: {param_group['lr']}")


                print_msg = (f'[{k:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                             f'train_loss: {train_loss:.5f} ' +
                             f'valid_loss: {valid_loss:.5f}')

                print(print_msg)

                # clear lists to track next epoch
                train_losses = []
                valid_losses = []

                # early_stopping needs the validation loss to check if it has decresed,
                # and if it has, it will make a checkpoint of the current model

                reduce_lr(valid_loss)
                early_stopping(valid_loss, gru,qugru)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            # visualize the loss as the network trained-
            fig = plt.figure(figsize=(10, 8))
            plt.plot(range(1, len(avg_train_losses) + 1), avg_train_losses, label='Training Loss')
            plt.plot(range(1, len(avg_valid_losses) + 1), avg_valid_losses, label='Validation Loss')

            # find position of lowest validation loss
            minposs = avg_valid_losses.index(min(avg_valid_losses)) + 1
            plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

            plt.xlabel('epochs')
            plt.ylabel('loss')
          #  plt.ylim(0, 0.5)  # consistent scale
            plt.xlim(0, len(avg_train_losses) + 1)  # consistent scale
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.close()
            fig.savefig('{}_loss_plot.png'.format(id), bbox_inches='tight')

            print("模型预测结果：", results)
            print("预测误差MAE:", losss)
            print("NSE:", lossnse)
            print(len(results))

            plt.plot(lossnse)
            plt.title("NSE-loss")
            plt.savefig("{}_NSE-loss1".format(id))
            plt.close()
            plt.plot(losss)
            plt.title("Loss")
            plt.savefig("{}_Loss1".format(id))
            plt.figure()
            plt.style.use('ggplot')


        # else:
            # 加载模型进行预测
            # gru.load_state_dict(torch.load('save_model_cmal_vmd1515revin-early-5-15.pth'))
            # load the last checkpoint with the best model
            gru.load_state_dict(torch.load('{}_ji.pt'.format(id)))
            # qugru.load_state_dict(torch.load('model1.pt'))

            print(gru)
            gru.eval()  # 评估模式

            results = []
            reals = []
            losss = []
            losnse=[]
            for seq, labels in test_loader:
                seq=seq.to(device)
                labels = labels.to(device)
                a, b, c = seq.shape
                # 使用循环来批量处理时间序列
                # seq = revin_layer(seq, 'norm')

                pred =gru(seq)
                # pred = revin_layer(pred, 'denorm')

                mae = calculate_mae(pred, labels)  # MAE误差计算绝对值(预测值  - 真实值)
                nse=NSE(pred, labels)
                losss.append(mae)
                losnse.append(nse)
                p = 0
                for j in range(batch_size):
                    for i in range(pre_len):#如果要一步一步的计算，这里需要注释掉
                        p=i
                        reals.append(labels[j][p][0])
                        # reals.append(labels[j][1][0].detach().numpy())
                        results.append(pred[j][p][0])


            # reals = scaler_test.inverse_transform(np.array(reals).reshape(1, -1))[0]
            # results = scaler_test.inverse_transform(np.array(results).reshape(1, -1))[0]
            reals = scaler_test.inverse_transform(np.array(reals).reshape(1, -1))[0]
            results = scaler_test.inverse_transform(np.array(results).reshape(1, -1))[0]


            nse = NSE1(results, reals)
            # tensor(0.9728)  
            print(nse)
            print("模型预测结果：", results)
            print("模型预测结果：", reals)
            print("预测误差MAE:", losss)
            print(len(results))

            # mae, mse, rmse, mape, mspe, rse,nd, nrmse ,nse= metric(reals, results)
            # print('nd:{}, nrmse:{}, mse:{}, mae:{}, rse:{}, mape:{}'.format(nd, nrmse, mse, mae, rse, mape))

            plt.plot(losnse)
            plt.close()
            plt.figure()
            plt.style.use('ggplot')

            # 创建折线图
            plt.plot(reals, label='real', color='blue')  # 实际值
            plt.plot(results, label='forecast', color='red', linestyle='--')  # 预测值

            # 增强视觉效果
            plt.grid(True)
            plt.title('real vs forecast')
            plt.xlabel('time')
            plt.ylabel('value')
            plt.legend()
            plt.savefig('{}_test.png'.format(id))
            plt.close()
            # 这里输出的评价指标怎么这么大呢，是不是不合理哦？？
            mae, mse, rmse, mape, mspe, rse, nd, nrmse, nse, kge, bias = metric(reals, results)
            print(
                'nd:{}, nrmse:{}, mse:{}, mae:{}, rse:{}, mape:{}, nse:{}, kge:{}, bias:{}'.format(nd, nrmse, mse, mae,
                                                                                                   rse, mape, nse, kge,
                                                                                                   bias))
            test_mape = np.mean(np.abs((results - reals) / reals))  # 平均绝对百分比误差
            # rmse
            test_rmse = np.sqrt(np.mean(np.square(results - reals)))  # 均方根误差
            # mae
            test_mae = np.mean(np.abs(results - reals))  # 平均绝对误差
            # R2
            from sklearn.metrics import r2_score
            test_r2 = r2_score(reals, results)
            from sklearn.metrics import mean_squared_error
            mse = mean_squared_error(reals, results)

            print('LSTM测试集的mape:', test_mape, ' rmse:', test_rmse, ' mae:', test_mae, ' R2:', test_r2, 'mse:', mse, 'nse:', nse)
            # 假设这是模型计算得到的指标
            data = {
                'id': [id],  # 确保'id'是一个列表
                'nse': [nse],  # 将nse转换为列表
                'mape': [mape],
                'r^2': [test_r2],
                'rse': [rse],
                'nrmse': [nrmse],
                'nd': [nd],
                'kge': [kge],
                'bias': [bias]  # bias也转换为列表
            }
            import csv

            # 将数据转换为DataFrame
            # df = pd.DataFrame(data)

            # df.to_csv("D:\\Code\\pytoch_code\\data_chuli\\metrics.csv", index="id", encoding='utf_8_sig')
            # 尝试以追加模式打开文件
            try:
                df = pd.read_csv(csv_file)
    
                new_df = pd.DataFrame(data)
                # 假设'id'列是索引或者我们想要基于它合并数据（但在这个例子中，我们直接追加）
                df = df.append(new_df, ignore_index=True)  # ignore_index=True用于重置索引
            except FileNotFoundError:
                # 如果文件不存在，直接创建新的DataFrame
                df = pd.DataFrame(data)

                # 将DataFrame写回CSV文件，不会覆盖原有数据
            df.to_csv(csv_file, index=False, encoding='utf_8_sig')  # index=False表示不保存索引到CSV文件

print('所有文件已处理完毕，图表已保存到目标文件夹。')