import torch
import torch.nn as nn
import torch.nn.functional as F
from mmpose.registry import MODELS
from mmengine.model import BaseModule

@MODELS.register_module()
class KDLoss(nn.Module):

    """ PyTorch version of KD for Pose """

    def __init__(self,
                 name,
                 use_this,
                 weight=1.0,
                 ):
        super(KDLoss, self).__init__()

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.kl_loss = nn.KLDivLoss(reduction='none')
        self.weight = weight

    def forward(self, pred, pred_t, beta, target_weight):
        ls_x, ls_y = pred
        lt_x, lt_y = pred_t

        lt_x = lt_x.detach()
        lt_y = lt_y.detach()

        num_joints = ls_x.size(1)
        loss = 0

        loss += (
            self.loss(ls_x, lt_x, beta, target_weight))
        loss += (
            self.loss(ls_y, lt_y, beta, target_weight))

        # 对每'种'关键点 做平均 (不区分x和y坐标) 
        # --- 平均到 一个图片中 每‘种‘关键点的 平均KL散度 
        return loss / num_joints

    def loss(self, logit_s, logit_t, beta, weight):
        
        N = logit_s.shape[0]
        Bins = logit_s.shape[-1]

        if len(logit_s.shape) == 3:
            K = logit_s.shape[1]
            logit_s = logit_s.reshape(N * K, -1)
            logit_t = logit_t.reshape(N * K, -1)

        # beta是温度因子  
        # N*W(H)
        s_i = self.log_softmax(logit_s * beta)
        t_i = F.softmax(logit_t * beta, dim=1)

        # 跟 KLDiscretLoss 不同, 这里是 沿着 类别维度 求和 
        # kd
        loss_all = torch.sum(self.kl_loss(s_i, t_i), dim=1)
        # --> loss平均到一个照片   (也考虑没有标注的点, 不会丢弃没标注的点)
        # .sum(dim=1) -> (N, )
        # .mean() -> (1,)
        loss_all = loss_all.reshape(N, K).sum(dim=1).mean()
        loss_all = self.weight * loss_all

        return loss_all