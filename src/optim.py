import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

class Novograd(torch.optim.Optimizer):
    """
    Implements Novograd algorithm.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.95, 0))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        grad_averaging: gradient averaging
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    """

    def __init__(self, params, lr=1e-3, betas=(0.95, 0), eps=1e-8,
                 weight_decay=0, grad_averaging=False, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay,
                        grad_averaging=grad_averaging,
                        amsgrad=amsgrad)

        super(Novograd, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Novograd, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Sparse gradients are not supported.')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros([]).to(state['exp_avg'].device)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros([]).to(state['exp_avg'].device)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                norm = torch.sum(torch.pow(grad, 2))

                if exp_avg_sq == 0:
                    exp_avg_sq.copy_(norm)
                else:
                    exp_avg_sq.mul_(beta2).add_(1 - beta2, norm)

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                grad.div_(denom)
                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)
                if group['grad_averaging']:
                    grad.mul_(1 - beta1)
                exp_avg.mul_(beta1).add_(grad)

                p.data.add_(-group['lr'], exp_avg)

        return loss


class Optimizer():
    def __init__(self, parameters, optimizer, lr_scheduler, tf_start=1, tf_end=1, tf_step=1,
                 recon_init_weight=1.0, recon_decay=0.0, **kwargs):

        # Setup teacher forcing scheduler
        self.tf_rate = lambda step: max(tf_end, tf_start-(tf_start-tf_end)*step/tf_step)
        self.recon_sch = recon_init_weight!=1.0
        self.recon_rate = lambda step: max(1.0, recon_init_weight-(recon_init_weight-1.0)/max(recon_decay,1.0))

        # Setup torch optimizer
        self.tf_type = tf_end!=1
        self.opt_type = optimizer['type']
        init_lr = optimizer['lr']
        self.sch_type = lr_scheduler
        opt_type = optimizer.pop('type')
        if opt_type == 'novograd':
            opt = Novograd
        else:
            opt = getattr(torch.optim, opt_type)

        self.opt = opt(parameters,**optimizer)
        if lr_scheduler['type'] == 'warmup':
            warmup_step = 4000.0
            self.lr_scheduler = lambda step: init_lr * warmup_step **0.5 * np.minimum((step+1)*warmup_step**-1.5,(step+1)**-0.5 )
        elif lr_scheduler['type'] == 'decay':
            warmup_step = 1000.0
            self.lr_scheduler = lambda step: init_lr * warmup_step **0.5 * np.minimum((step+1)*warmup_step**-1.5,(step+1)**-0.5 )
        elif lr_scheduler['type'] == 'reduce_lr_on_plateau':
            lr_scheduler.pop('type')
            self.lr_scheduler = ReduceLROnPlateau(self.opt, **lr_scheduler)
        else:
            self.lr_scheduler = None

    def get_opt_state_dict(self):
        return self.opt.state_dict()

    def load_opt_state_dict(self, state_dict):
        self.opt.load_state_dict(state_dict)

    def pre_step(self, step, dev_loss=None):
        if self.lr_scheduler is not None:
            if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                if dev_loss:
                    self.lr_scheduler.step(dev_loss)
            else:
                cur_lr = self.lr_scheduler(step)
                for param_group in self.opt.param_groups:
                    param_group['lr'] = cur_lr
        self.opt.zero_grad()
        return self.tf_rate(step)

    def step(self):
        self.opt.step()

    def recon_rate(self,step):
        return self.recon_rate(step)

    def create_msg(self):
        return ['Optim.spec.| Algo. = {}\t| Lr/sampling/rec.loss scheduler = {}/{}/{}'\
                   .format(self.opt_type, self.sch_type, self.tf_type, self.recon_sch)]

