from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

class TransformerLRScheduler(LambdaLR):
    def __init__(self, 
                 optimizer: Optimizer, 
                 d_model: int, 
                 warmup_steps: int = 4000,
                 max_lr = None) -> None:
        self.d_model = d_model
        self.warmup_steps = warmup_steps

        self.scale = 1.0
        if max_lr:
            # Scales scheduler to maximum learning rate.
            optim_lr = optimizer.param_groups[0]['lr']
            prev_max_lr = optim_lr * d_model ** (-0.5) * warmup_steps ** (-0.5)
            self.scale = max_lr / prev_max_lr

        super().__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, step: int) -> float:
        step = step if step != 0 else 1
        return self.d_model ** (-0.5) * min(step ** (-0.5), step * self.warmup_steps ** (-1.5)) * self.scale
