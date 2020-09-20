"""
Shared optimizer, the parameters in the optimizer will shared in the multiprocessors.
"""

import torch


class SharedRMSprop(torch.optim.RMSprop):
    def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False):
        super(SharedRMSprop, self).__init__(params, lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=0)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['square_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['grad_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                # share in memory
                state['square_avg'].share_memory_()
                state['momentum_buffer'].share_memory_()
                state['grad_avg'].share_memory_()

