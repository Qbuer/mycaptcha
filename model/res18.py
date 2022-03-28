import pytorch_lightning as pl
import torch.nn as nn
import torch
import torch.nn.functional as F
# from torch.optim.radam import RAdam
from torch import optim
from lib.scheduler import GradualWarmupScheduler
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock
from lib.radam import RAdam

from dataModule import LabeltoStr

class Res18Model(pl.LightningModule):
    def __init__(self, class_num=62):
        super(Res18Model, self).__init__()
        model_ft = ResNet(BasicBlock, [2, 2, 2, 2])
        self.base_model = nn.Sequential(*list(model_ft.children())[:-3])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.sign = nn.Sigmoid()
        in_plances = 256
        ratio = 8
        # self.a_fc1 = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Conv2d(in_plances,in_plances//ratio,1,bias=False)
        # )
        self.a_fc1 = nn.Conv2d(in_plances,in_plances//ratio,1,bias=False)
    
        self.a_relu = nn.ReLU()

        self.a_fc2 = nn.Conv2d(in_plances//ratio, in_plances, 1, bias=False)

        # self.a_fc2 = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Conv2d(in_plances//ratio, in_plances, 1, bias=False)
        # )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.reduce_layer = nn.Conv2d(512, 256, 1)

        # self.classifier = ClassBlock(512, 1024)
        self.fc1 = nn.Sequential(nn.Dropout(0.5),
                                 nn.Linear(256, class_num))
        self.fc2 = nn.Sequential(nn.Dropout(0.5),
                                 nn.Linear(256, class_num))
        self.fc3 = nn.Sequential(nn.Dropout(0.5),
                                 nn.Linear(256, class_num))
        self.fc4 = nn.Sequential(nn.Dropout(0.5),
                                 nn.Linear(256, class_num))
        self.l1 = nn.Linear(28 * 28, 10)
        self.criterion = nn.CrossEntropyLoss()

        self.res = []

    def forward(self, x):
        
        bs = x.shape[0]
        x = self.base_model(x)
        # channel attention   
        avgout = self.a_fc2(self.a_relu(self.a_fc1(self.avgpool(x))))
        maxout = self.a_fc2(self.a_relu(self.a_fc1(self.maxpool(x))))
        ca = self.sign(avgout+maxout)
        # joint
        x = x * ca.expand_as(x)

        # fuse avgpool and maxpool
        xx1 = self.avg_pool(x)#.view(bs, -1).squeeze()
        xx2 = self.max_pool(x)#.view(bs, -1).squeeze()
        # xx1 = self.avg_pool(x)
        # xx2 = self.max_pool(x)
        # fuse the feature by concat
        x = torch.cat([xx1, xx2], dim=1)
        x = self.reduce_layer(x).view(bs,-1)
        # print(x.shape)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x)
        x4 = self.fc4(x)
        return x1, x2, x3, x4

    def training_step(self, batch, batch_idx):
        
        x, label, _ = batch
        label1, label2, label3, label4 = label[:, 0], label[:, 1], label[:, 2], label[:, 3]
        y1, y2, y3, y4 = self(x)
        loss1, loss2, loss3, loss4 = self.criterion(y1, label1), self.criterion(
                y2, label2), self.criterion(y3, label3), self.criterion(y4, label4)
        loss = loss1 + loss2 + loss3 + loss4
        self.log("train_loss", loss, on_epoch=True, on_step=True)
        return loss

    # def on_train_epoch_end(self):
    #     # self.log("lr", self.lr_schedulers().get_last_lr()[0], on_epoch=True, on_step=False, prog_bar=True)
    #     pass

    def validation_step(self, batch, batch_idx):

        x, label, _ = batch 
        batch_size = x.shape[0]
        label1, label2, label3, label4 = label[:, 0], label[:, 1], label[:, 2], label[:, 3]
        y1, y2, y3, y4 = self(x)
        loss1, loss2, loss3, loss4 = self.criterion(y1, label1), self.criterion(
                y2, label2), self.criterion(y3, label3), self.criterion(y4, label4)
        
        loss = loss1 + loss2 + loss3 + loss4

        small_bs = x.size()[0]  # get the first channel
        topK = 1
        y1, y2, y3, y4 = y1.topk(topK, dim=1)[1].view(small_bs, topK), \
            y2.topk(topK, dim=1)[1].view(small_bs, topK), \
            y3.topk(topK, dim=1)[1].view(small_bs, topK), \
            y4.topk(topK, dim=1)[1].view(small_bs, topK)
        y1, y2, y3, y4 = y1.unsqueeze(2), y2.unsqueeze(2), y3.unsqueeze(2), y4.unsqueeze(2)
        y = torch.cat((y1, y2, y3, y4), dim=2)
        label = label.unsqueeze(1)
        diff = (y != label)
        diff = diff.sum(2).min(dim=1)[0]
        
        diff = (diff != 0)
        res = diff.sum(0).item()
        correct_num = (small_bs - res)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('val_acc', correct_num/ batch_size,
                 on_step=False, on_epoch=True, prog_bar=True)

        

        # for debug, 查看预测错的数据是否有“特征”
        # y (none, topk, 4)
        # label (none, 1, 4)
        label = label.squeeze(1)

        y = y[:,0,:].squeeze(1)
        for y_, l_ in zip(y.tolist(), label.tolist()):
            if y_ == l_: continue
            self.res.append((LabeltoStr(y_), LabeltoStr(l_)))
        

    def predict_step(self, batch, batch_idx):
        x, _, label = batch 
        y1, y2, y3, y4 = self(x)
        y1, y2, y3, y4 = y1.topk(1, dim=1)[1].view(1, 1), y2.topk(1, dim=1)[1].view(1, 1), \
                         y3.topk(1, dim=1)[1].view(1, 1), y4.topk(1, dim=1)[1].view(1, 1)
        y = torch.cat((y1, y2, y3, y4), dim=1)
        # print(x,label,y)
        decLabel = LabeltoStr([y[0][0], y[0][1], y[0][2], y[0][3]])
        return label[0], decLabel



    def configure_optimizers(self):
        optimizer = RAdam(self.parameters(),
                      lr=3e-4,
                      betas=(0.9, 0.999),
                      weight_decay=6.5e-4)
        # optimizer = optim.Adam(self.parameters(), lr=3e-4)
        scheduler_after = optim.lr_scheduler.StepLR(optimizer,
                                                step_size=20,
                                                gamma=0.5)
        scheduler = GradualWarmupScheduler(optimizer, 8, 10, after_scheduler=scheduler_after)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }
