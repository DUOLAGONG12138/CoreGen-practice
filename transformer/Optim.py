import numpy as np

class ScheduledOptim():
    """学习率调度的简单封装类"""
    def __init__(self, optimizer, n_warmup_steps, d_model) -> None:
        self._optimizer = optimizer
        self.n_warmup_step = n_warmup_steps
        self.n_current_steps = 0
        self.inti_lr = np.power(d_model, -0.5)
    
    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "内部优化器将梯度清零"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr