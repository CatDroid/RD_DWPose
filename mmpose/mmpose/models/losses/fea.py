import torch
import torch.nn as nn
import torch.nn.functional as F
from mmpose.registry import MODELS
from mmengine.model import BaseModule

@MODELS.register_module()
class FeaLoss(nn.Module):

    """PyTorch version of feature-based distillation
   
    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map. 
        alpha_fea (float, optional): Weight of dis_loss. Defaults to 0.00007
    """
    def __init__(self,
                 name,
                 use_this,
                 student_channels,
                 teacher_channels,
                 alpha_fea=0.00007,
                 ):
        super(FeaLoss, self).__init__()
        self.alpha_fea = alpha_fea

        if teacher_channels != student_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None

    def forward(self,
                preds_S,
                preds_T):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
        """

        if self.align is not None:
            outs = self.align(preds_S)
        else:
            outs = preds_S

        loss = self.get_dis_loss(outs, preds_T)

        return loss

    def get_dis_loss(self, preds_S, preds_T):
        # 归约是求和 
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape

        # 并没有求CHW的平均，只是平均到一个样本 (?估计是因为只用backbone最后一层的输出)
        dis_loss = loss_mse(preds_S, preds_T)/N*self.alpha_fea

        return dis_loss