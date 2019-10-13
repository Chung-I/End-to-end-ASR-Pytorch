import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

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
        opt = getattr(torch.optim,optimizer.pop('type'))
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



