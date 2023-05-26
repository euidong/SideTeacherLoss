import os
from model import Model
import torch


class SideTeacherLoss:
    def __init__(self, student, base_loss, device, teachers = [], prefix='model', cls=Model, alpha=0.0001, dist: str = "l2-neg") -> None:
        self.student = student
        self.base_loss = base_loss
        self.alpha = alpha
        self.param_teachers = []
        file_list = os.listdir('param')
        if len(teachers) > 0:
            self.teachers = teachers
            for t in self.teachers:
                param = []
                for p in t.parameters():
                    p.requires_grad = False
                    param.append(p)
                self.param_teachers.append(param)
        else:
            self.teachers = []
            idx = 0
            for file in file_list:
                if prefix in file:
                    m = cls().to(device)
                    m.load(idx)
                    self.teachers.append(m)
                    param = []
                    for p in m.parameters():
                        p.requires_grad = False
                        param.append(p)
                    self.param_teachers.append(param)
                    idx += 1

        self.dist = dist
        self.regularizers = {
            "l2-neg": lambda x, y: -torch.sum((x-y) ** 2),
            "l1-neg": lambda x, y: -torch.sum(torch.abs(x-y)),
            "fro-neg": lambda x, y: -torch.sum(torch.norm(x-y, p='fro')),
            "nuc-neg": lambda x, y: -torch.sum(torch.norm(x-y, p='nuc')),
            "l2-recp": lambda x, y: 1 / torch.sum((x-y) ** 2),
            "l1-recp": lambda x, y: 1 / torch.sum(torch.abs(x-y)),
            "fro-recp": lambda x, y: 1 / torch.sum(torch.norm(x-y, p='fro')),
            "nuc-recp": lambda x, y: 1 / torch.sum(torch.norm(x-y, p='nuc')),
        }
    
    def __call__(self, y_pred, y):
        loss = self.base_loss(y_pred, y)
        cnt = 0
        for s_p in self.student.parameters():
            teacher_losses = []

            for t_ps in self.param_teachers:
                if "nuc" in self.dist and len(s_p.shape) != 2:
                    reg = self.alpha * self.regularizers[self.dist](s_p.view(s_p.shape[0], -1), t_ps[cnt].view(t_ps[cnt].shape[0], -1))
                    teacher_losses.append(reg.item())
                else:
                    reg = self.alpha * self.regularizers[self.dist](s_p, t_ps[cnt])
                    teacher_losses.append(reg.item())

            teacher_loss_tensor = torch.FloatTensor(teacher_losses)
            loss += self.alpha * torch.max(teacher_loss_tensor)
            cnt += 1
        return loss
    