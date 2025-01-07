from collections import OrderedDict
from copy import deepcopy

import torch
from torch.nn import functional as F

from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel


@MODEL_REGISTRY.register()
class MPRNetModel(SRModel):
    def get_optimizer(self, optim_type, params, lr, **kwargs):
        if optim_type == 'Adam':
            optimizer = torch.optim.Adam(params, lr, **kwargs)
        elif optim_type == 'AdamW':
            optimizer = torch.optim.AdamW(params, lr, **kwargs)
        elif optim_type == 'SGD':
            optimizer = torch.optim.SGD(params, lr, **kwargs)
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supperted yet.')
        return optimizer

    def test(self):
        # pad to multiplication of window_size
        window_size = 8
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(img)[-1]
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(img)[-1]
            self.net_g.train()
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss for 3 stages
        if self.cri_pix:
            for j in range(len(self.output)):
                l_pix = self.cri_pix(self.output[j], self.gt)
                l_total = l_total + l_pix
                loss_dict['l_pix'] = l_total
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
    # def load_network(self, net, load_path, strict=True, param_key='params'):
    #     """Load network.
    #
    #     Args:
    #         load_path (str): The path of networks to be loaded.
    #         net (nn.Module): Network.
    #         strict (bool): Whether strictly loaded.
    #         param_key (str): The parameter key of loaded network. If set to
    #             None, use the root 'path'.
    #             Default: 'params'.
    #     """
    #
    #     # logger = get_root_logger()
    #     net = self.get_bare_model(net)
    #     load_net = torch.load(load_path)
    #     # if param_key is not None:
    #     #     if param_key not in load_net and 'params' in load_net:
    #     #         param_key = 'params'
    #     #         logger.info('Loading: params_ema does not exist, use params.')
    #     #     load_net = load_net[param_key]
    #     # logger.info(f'Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}].')
    #     # remove unnecessary 'module.'
    #     load_net = load_net["state_dict"]
    #     for k, v in deepcopy(load_net).items():
    #         if k.startswith('module.'):
    #             load_net[k[7:]] = v
    #             load_net.pop(k)
    #     self._print_different_keys_loading(net, load_net, strict)
    #     net.load_state_dict(load_net, strict=strict)