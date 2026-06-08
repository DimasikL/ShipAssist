import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# CHANGED: Добавлен импорт Callable
from typing import Optional, Tuple, Callable

class FocalLoss(nn.Module):
    """
    CHANGED: Перенесено из main_v3.
    Помогает при сильном дисбалансе классов (когда 'других слов' больше).
    """
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

class EMA:
    """
    CHANGED: Перенесено из main_v2.
    Экспоненциальное скользящее среднее весов для стабильности модели.
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self) -> None:
        """Обновление усредненных весов."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self) -> None:
        """Применить усредненные веса к модели (для валидации)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self) -> None:
        """Восстановить веса после валидации."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    CHANGED: Перенесено из main_v4.
    Регуляризация через смешивание аудио-признаков.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion: Callable, pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
    """Функция потерь для Mixup."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)