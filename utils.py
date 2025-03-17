import torch
from typing import Literal

class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class Normalize(torch.nn.Module):
    def __init__(self, norm_type: Literal['l1', 'l2'], axis=1):
        """
        Модуль, реализующий L1-, L2-нормализацию по заданной оси
        """
        super(Normalize, self).__init__()

        self.norm = 1 if norm_type == 'l1' else 2
        self.axis = axis

    def forward(self, x):
        # Вычисляем L1- или L2-норму
        norm = torch.norm(x, p=self.norm, dim=self.axis, keepdim=True)
        # Чуть-чуть "зажимаем" результат
        norm = norm.clamp(min=1e-8)

        # Нормализуем
        return x / norm
    