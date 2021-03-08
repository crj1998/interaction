
import os
import logging
from logging import handlers

import torch
import torch.nn.functional as F

class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }
    def __init__(self, filename, level='info', when='D', backCount=3, fmt='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S"):
        if os.path.exists(filename):
            os.remove(filename)
        format_str = logging.Formatter(fmt, datefmt)
        self.logger = logging.getLogger(filename)
        self.logger.setLevel(self.level_relations.get(level))
        console = logging.StreamHandler()
        console.setFormatter(format_str)
        fh = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backCount, encoding='utf-8')
        fh.setFormatter(format_str)
        self.logger.addHandler(console)
        self.logger.addHandler(fh)


def madrys(model, x_natural, y, device, step_size=0.021, epsilon=0.031, perturb_steps=3, distance='l_inf', isnormalize=True, mean=None, std=None, targeted=False):
    model.eval()
    if distance == 'l_inf':
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()    # random start
    elif distance == 'l_2':
        delta = torch.zeros_like(x_natural).to(device).detach()
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
        x_adv = x_natural.detach() + delta
    x_adv = torch.clamp(x_adv, min=0, max=1)     # x_adv is in [0,1]

    if targeted:  # targeted
        multiplier = -1
    else:          # non-targeted
        multiplier = 1

    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            if isnormalize:
                if (type(model(normalize(x_adv, mean, std))).__name__ == 'tuple') :
                    output, _ = model(normalize(x_adv, mean, std))
                else:
                    output = model(normalize(x_adv, mean, std))
            else:
                if (type(model(x_adv)).__name__ == 'tuple') :
                    output, _ = model(x_adv)
                else:
                    output = model(x_adv)
            loss_ce = F.cross_entropy(output, y)
            loss_ce = multiplier * loss_ce
        grad = torch.autograd.grad(loss_ce, [x_adv])[0]    # 是放入x_adv还是  normalize(x_adv)

        if distance == 'l_inf':
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        elif distance == 'l_2':
            g_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = grad / (g_norm + 1e-10)
            x_adv = x_adv.detach() + step_size * scaled_g
            delta = x_adv - x_natural
            delta = delta.renorm(p=2, dim=0, maxnorm=epsilon)
            x_adv = torch.clamp(x_natural + delta, 0.0, 1.0)

    return x_adv