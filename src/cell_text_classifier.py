import pandas as pd
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from tensorboardX import SummaryWriter

from model.hwdb import HWDB
from model.resnet_101 import ResNet101
from model.train import train, valid


def get_road_names() -> pd.Series:
    return pd.read_excel("../references/road_names_of_sh.xlsx", sheet_name="Sheet1")["路名"].unique()


def get_road_sections_by_road_name(road_name: str) -> pd.Series:
    road_table = pd.read_excel("../references/road_names_of_sh.xlsx", sheet_name="Sheet1")
    road_sections = pd.Series(road_table[road_table["路名"] == road_name]["路段"].unique())
    return road_sections


class CellTextClassifier:
    def __init__(self, cell_type: str, road_name: str = None):
        self.cell_type = cell_type
        self.cell_type_class_num_dict = {
            "col_material": 5,
            "col_client": 5,
            "road_name": 15795,
        }
        self.data_set_path = "G:/artificial_CN_handwriting/"
        self.road_name = road_name
        if self.cell_type == "col_material":
            self.class_list = ["硅", "硅管", "钢", "钢管", "波纹管", "波", "PVC", "PE"]
        elif self.cell_type == "col_client":
            self.class_list = ["移动", "联通", "信息", "未知", "有线"]
        elif self.cell_type == "road_name":
            self.class_list = get_road_names()
        elif self.cell_type == "road_section":
            self.class_list = get_road_sections_by_road_name(self.road_name)

        # Hyper parameters for training
        self.epochs = 40
        self.batch_size = 16
        self.lr = 0.02
        self.device = "cuda:0"
        self.check_point = None
        self.log_path = f"../logs/{self.cell_type}_batch_{self.batch_size}_lr_{self.lr}"
        self.save_path = '../checkpoints/' + self.cell_type + '/'
        self.input_size = 128

    def train_model(self):
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        dataset = HWDB(path=self.data_set_path + self.cell_type, transform=transform)
        print("训练集数据:", dataset.train_size)
        print("测试集数据:", dataset.test_size)
        train_loader, test_loader = dataset.get_loader(self.batch_size)

        net = ResNet101(len(self.class_list))
        if self.check_point:
            net.load_state_dict(torch.load(self.check_point))
            print("已加载预训练模型")
        if torch.cuda.is_available():
            net.to(self.device)
            print("已将模型加载至 " + self.device)

        print('网络结构：\n')
        if self.device == 'cuda:0':
            summary(net, input_size=(3, self.input_size, self.input_size), device='cuda')
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=self.lr)
        # optimizer = optim.Adam(net.parameters(), lr=lr)
        writer = SummaryWriter(self.log_path)
        max_correct = 0
        fluctuate_epoch_cnt = 0
        for epoch in range(self.epochs):
            train(epoch, net, criterion, optimizer, train_loader, writer=writer, device=self.device)
            correct = valid(epoch, net, test_loader, writer=writer, device=self.device)
            if correct >= max_correct:
                fluctuate_epoch_cnt = 0
                max_correct = correct
                print("epoch%d 结束, 正在保存模型..." % epoch)
                torch.save(net.state_dict(),
                           self.save_path + f'{self.cell_type}_ResNet101_fs{self.input_size}_bs{self.batch_size}_SGD_{epoch + 0}.pth')
            else:
                fluctuate_epoch_cnt += 1
                print("epoch%d 精确率波动，不会保存模型" % epoch)
                if fluctuate_epoch_cnt >= 3:
                    fluctuate_epoch_cnt = 0
                    print("epoch%d 结束, 下一轮训练学习率将减半" % epoch)
                    self.lr /= 2


if __name__ == "__main__":
    classifier = CellTextClassifier(cell_type="col_client")
    classifier.train_model()
