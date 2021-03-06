import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

Batch = 32  # 批次数据大小
Epoch = 100  # 迭代次数
Lr = 1e-2  # 学习率
Dropout = 0.1  # 遗忘概率
LrDecayEpoch = 50  # 迭代对应次数后，减小学习率

TrainSamplePath = 'TrainSamples.csv'
TrainLabelPath = 'TrainLabels.csv'
TestSamplePath = 'TestSamples.csv'  # 测试集样本路径
TestLabelPath = 'TestLabels.csv'  # 测试集标签路径
TestPredPath = 'Result.csv'  # 测试集预测结果路径
ModelPath = 'large_model.pth'  # 模型文件路径

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class MnistDataset(Dataset):
    def __init__(self, data_arr, label_arr=None):
        self.data = data_arr
        self.label = label_arr

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = {'data': self.data[idx]}
        if self.label is not None:
            item['label'] = self.label[idx]
        return item


def split_data(data_path, label_path, ratio=(0.8, 0.1, 0.1)):
    data_np, label_np = np.genfromtxt(data_path, delimiter=',', dtype=np.float32), \
        np.genfromtxt(label_path, delimiter=',', dtype=np.int64)
    data_num = len(data_np)

    indices = np.arange(data_num)
    np.random.shuffle(indices)

    train_num, train_val_num = int(
        data_num * ratio[0]), int(data_num * (1 - ratio[2]))
    train_x, train_y = data_np[indices[:train_num]
                               ], label_np[indices[:train_num]]
    val_x, val_y = data_np[indices[train_num:train_val_num]
                           ], label_np[indices[train_num:train_val_num]]
    test_x, test_y = data_np[indices[train_val_num:]
                             ], label_np[indices[train_val_num:]]
    return train_x, train_y, val_x, val_y, test_x, test_y


def get_model():
    model = nn.Sequential(
        nn.Linear(76, 4096),
        nn.BatchNorm1d(4096),
        nn.ReLU(),

        nn.Dropout(0.5),

        nn.Linear(4096, 4096),
        nn.BatchNorm1d(4096),
        nn.ReLU(),

        nn.Dropout(0.5),

        nn.Linear(4096, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),

        nn.Dropout(0.5),

        nn.Linear(1024, 10)
    )
    return model


# def get_model():
#     model = nn.Sequential(
#         nn.Linear(76, 128 * 8),
#         nn.BatchNorm1d(128 * 8),
#         nn.ReLU(),

#         nn.Dropout(0.1),

#         nn.Linear(128 * 8, 64 * 4),
#         nn.BatchNorm1d(64 * 4),
#         nn.ReLU(),

#         nn.Dropout(0.2),

#         nn.Linear(64 * 4, 64 * 4),
#         nn.BatchNorm1d(64 * 4),
#         nn.ReLU(),

#         nn.Dropout(0.4),

#         nn.Linear(64 * 4, 10)
#     )
#     return model


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train():
    def eval_data(_model, data_loader, _loss_func):
        _model.eval()
        _avg_loss, _right_num, _data_num = 0, 0, 0
        for _batch in data_loader:
            _batch_data, _batch_label = _batch['data'].to(
                device), _batch['label'].to(device)
            _predict = _model(_batch_data)
            _loss = _loss_func(_predict, _batch_label)
            _avg_loss += _loss.item() / len(data_loader)

            _, _predict_ids = _predict.max(1)
            _right_num += _predict_ids.eq(_batch_label).sum().item()
            _data_num += len(_batch_data)
        return _right_num, _data_num, _avg_loss, _right_num / _data_num

    train_x, train_y, dev_x, dev_y, test_x, test_y = split_data(
        TrainSamplePath, TrainLabelPath, ratio=(0.8, 0.1, 0.1))
    print('Train: ' + str(train_x.shape) + '\nVal: ' +
          str(dev_x.shape) + '\nTest: ' + str(test_x.shape))
    mean_vec, std_vec = np.mean(train_x, axis=0), np.std(train_x, axis=0)
    train_dataset = MnistDataset((train_x - mean_vec) / std_vec, train_y)
    train_dataloader = DataLoader(
        train_dataset, batch_size=Batch, shuffle=True, num_workers=4)
    dev_dataset = MnistDataset((dev_x - mean_vec) / std_vec, dev_y)
    dev_dataloader = DataLoader(
        dev_dataset, batch_size=Batch, shuffle=False, num_workers=4)
    test_dataset = MnistDataset((test_x - mean_vec) / std_vec, test_y)
    test_dataloader = DataLoader(
        test_dataset, batch_size=Batch, shuffle=False, num_workers=4)

    model = get_model()
    model.to(device)
    model.apply(weights_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=Lr)
    loss_func = nn.CrossEntropyLoss()

    best_dev_acc = 0
    for epoch in range(Epoch):
        for param_group in optimizer.param_groups:  # 动态更新学习率
            param_group['lr'] = Lr * (0.1 ** (epoch // LrDecayEpoch))
        print('-' * 80 + '\nCurrent lr = %f' %
              (Lr * (0.1 ** (epoch // LrDecayEpoch))))
        model.train()

        avg_loss, right_num, data_num = 0, 0, len(train_dataset)
        for idx, batch in enumerate(train_dataloader):
            batch_data, batch_label = batch['data'].to(
                device), batch['label'].to(device)
            predict = model(batch_data)
            _, predict_ids = predict.max(1)  # 找到第二维的最大值的索引
            right_num += predict_ids.eq(batch_label).sum().item()  # 预测正确的数目
            loss = loss_func(predict, batch_label)
            avg_loss += loss.item() / len(train_dataloader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Epoch [%3d/%3d] avg [loss = %.15f], [acc = %.4f%%]  [%d/%d]' % (
            epoch + 1, Epoch, avg_loss, right_num / data_num * 100, right_num, data_num))

        with torch.no_grad():  # 评估开发集上的模型效果，并保存最好的模型
            right_num, data_num, avg_loss, dev_acc = eval_data(
                model, dev_dataloader, loss_func)
            print('Avg Dev loss = %.15f, acc = %.4f%%  [%d/%d].' %
                  (avg_loss, dev_acc * 100, right_num, data_num))
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                print('Best Dev Acc: %.4f%%' % best_dev_acc)
                save_dict = {
                    'model': model.state_dict(), 'optim': optimizer.state_dict(),
                    'mean_vec': mean_vec, 'std_vec': std_vec
                }
                torch.save(save_dict, ModelPath)
    right_num, data_num, _, test_acc = eval_data(
        model, test_dataloader, loss_func)
    print('Test data = %.4f%%  [%d/%d]' %
          (test_acc * 100, right_num, data_num))


def eval_model(data_path, label_path=None):
    dataset = MnistDataset(np.genfromtxt(
        data_path, delimiter=',', dtype=np.float32))
    dataloader = DataLoader(dataset, batch_size=Batch,
                            shuffle=False, num_workers=4)

    model = get_model()
    ckpt = torch.load(ModelPath)
    mean_vec, std_vec = torch.tensor(ckpt['mean_vec']).to(
        device), torch.tensor(ckpt['std_vec']).to(device)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()

    predict = np.empty((len(dataset), 10), dtype=np.float32)
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            batch_data = batch['data'].to(device)
            predict[idx * Batch:(idx + 1) *
                    Batch] = model((batch_data - mean_vec) / std_vec).cpu()

    predict = np.argmax(predict, axis=1)
    if label_path:
        label_np = np.genfromtxt(label_path, delimiter=',', dtype=np.float32)
        right_num = np.sum(label_np == predict)
        print('Test data = %.4f%%  [%d/%d]' % (right_num /
              len(dataset) * 100, int(right_num), len(dataset)))

    with open(TestPredPath, 'w') as f:
        f.write('\n'.join(map(str, predict.tolist())))


if __name__ == '__main__':
    # train()
    eval_model(TestSamplePath)
    # eval_model(TrainSamplePath, TrainLabelPath)
    # eval_model(TestSamplePath, TestLabelPath)
