import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models import build_segmentor
from mmcv.utils import Config

from pspnet import pspnet_res101
from flownet import FlowNets
from warpnet import warp


class DFF(nn.Module):
    def __init__(self, num_classes=19, weight_res101=None, weight_flownet=None):
        super(DFF, self).__init__()

        # reference branch选用pspnet_res50
        self.net_feat = pspnet_res101()

        # net_task = 1*1 Conv + softmax
        # 论文里面说有没有这个1x1conv没什么区别，多加一个conv可以为以后需要时调参数,或者说更常规
        self.net_task = nn.Conv2d(num_classes, num_classes, kernel_size=1, stride=1, padding=0)

        # 光流场‘O()’选择FlowNets，预测的光流图
        self.flownet = FlowNets()

        # 用于传播关键帧到当前帧的可学习函数‘W()’，即将预测的光流图和关键帧的语义分割图进行融合
        self.warp = warp()

        # 权重初始化
        self.weight_init(weight_res101, weight_flownet)

        # 交叉熵损失函数
        self.criterion_semantic = nn.CrossEntropyLoss(ignore_index=255)

    def weight_init(self, weight_res101, weight_flownet):
        if weight_res101 is not None:

            # 加载预训练权重
            weight = torch.load(weight_res101, map_location='cpu')
            weight = weight['state_dict']

            # 加载预训练权重
            self.net_feat.model.load_state_dict(weight, False)

            # 冻结backdone的参数，仅调整decode_head的参数
            self.net_feat.fix_backbone()

        if weight_flownet is not None:
            weight = torch.load(weight_flownet, map_location='cpu')
            self.flownet.load_state_dict(weight, True)

        # 为了使得网络中信息更好的流动，每一层输出的方差应该尽量相等
        nn.init.xavier_normal_(self.net_task.weight)
        self.net_task.bias.data.fill_(0)
        print('pretrained weight loaded')

    # ---------------------------------Input-----------------------------------------------
    # gt = [batch_size,1,512,1024]
    # im_flow_list = [batch_size,3,2,512,1024]
    # im_seg_list = [batch_size,3,2,512,1024]
    # -------------------------------------------------------------------------------------
    def forward(self, im_seg_list, im_flow_list, gt=None):

        # 输入的视频数据参数值，依次为 bastchsize, 通道, 关键帧间隔时间, 帧高度, 帧宽度
        n, c, t, h, w = im_seg_list.shape

        # 推理关键帧的语义结果
        pred = self.net_feat(im_seg_list[:, :, 0, :, :])
        # pred.shape = [1,19,64,128]

        # 双线性插值等比放大2倍
        pred = F.interpolate(pred, scale_factor=2, mode='bilinear', align_corners=False)
        # pred.shape = [1,19.128.256]

        # 计算关键帧的光流传播:首先将关键帧和当前帧的tensor在通道处堆叠,然后传入flownet,从而根据关键帧和当前帧计算光流,
        flow = self.flownet(torch.cat([im_flow_list[:, :, -1, :, :], im_flow_list[:, :, 0, :, :]], dim=1))

        # 将关键帧的pred传入warp()，然后和当前帧的flow继续一个W()函数，输出pred
        pred_result = self.warp(pred, flow)

        # 将经过warp输出的pred放到task网络里面
        pred_result = self.net_task(pred_result)
        # 双线性插值放大4倍
        pred_result = F.interpolate(pred_result, scale_factor=4, mode='bilinear', align_corners=False)
        # pred_result.shape = [1,19,512,1024]

        if gt is not None:
            loss = self.criterion_semantic(pred_result, gt)

            # .unsqueeze(0) 表示，在第一个位置增加维度
            loss = loss.unsqueeze(0)
            return loss
        else:
            return pred_result

    def evaluate(self, im_seg_list, im_flow_list):
        out_list = []
        t = im_seg_list.shape[2]
        pred = self.net_feat(im_seg_list[:, :, 0, :, :])
        # pred.shape = [1,19,64,128]

        pred = F.interpolate(pred, scale_factor=2, mode='bilinear', align_corners=False)
        # pred.shape = [1,19,128,256]

        # 将经过net_feat的关键帧，再经过net_task处理
        out = self.net_task(pred)

        # 长宽均放大4倍
        out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=False)
        # out.shape = [1,19,512,1024]

        # 输入：out.shape = torch.Size([1, 19, 512, 1024])
        out = torch.argmax(out, dim=1)
        # 输出：out.shape = torch.Size([1, 512, 1024])

        out_list.append(out)

        # 当前帧和关键帧做一个光流估计
        flow = self.flownet(torch.cat([im_flow_list[:, :, -1, :, :], im_flow_list[:, :, 0, :, :]], dim=1))

        print(pred.shape)
        print(flow.shape)
        # 扔进‘W()’函数里
        pred_result = self.warp(pred, flow)
        print(pred_result.shape)

        # 对堆叠结果进行卷积
        pred_result = self.net_task(pred_result)
        pred_result = F.interpolate(pred_result, scale_factor=4, mode='bilinear', align_corners=False)

        # 取最大的可能性结果
        out = torch.argmax(pred_result, dim=1)
        out_list.append(out)

        return out_list

    def set_train(self):
        self.net_feat.eval()
        self.net_feat.model.decode_head.conv_seg.train()
        self.net_task.train()
        self.flownet.train()


if __name__ == '__main__':
    model = DFF(weight_res101=None, weight_flownet=None)
    model.cuda().eval()

    im_seg_list = torch.rand([1, 3, 5, 512, 1024]).cuda()
    im_flow_list = torch.rand([1, 3, 5, 512, 1024]).cuda()
    with torch.no_grad():
        out_list = model.evaluate(im_seg_list, im_flow_list)
        print(len(out_list), out_list[0].shape)
