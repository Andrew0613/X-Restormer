import torch
from torch.nn import functional as F

from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel
import math

@MODEL_REGISTRY.register()
class UformerModel(SRModel):
    def test(self):
        scale = self.opt.get('scale', 1)
        factor = 128
        original = self.lq
        rgb_noisy, mask = self.expand2square(self.lq, factor, scale)
        img = rgb_noisy
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(img)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(img)
            self.net_g.train()
        self.output = torch.masked_select(self.output, mask.bool()).reshape(1,3,original.shape[2]*scale,original.shape[3]*scale)
    def expand2square(self, timg, factor,scale):
        _, _, h, w = timg.size()

        X = int(math.ceil(max(h, w) / float(factor)) * factor)

        img = torch.zeros(1, 3, X, X).type_as(timg).to(self.device) # 3, h,w
        mask = torch.zeros(1, 1, X*scale, X*scale).type_as(timg).to(self.device)

        # print(img.size(),mask.size())
        # print((X - h)//2, (X - h)//2+h, (X - w)//2, (X - w)//2+w)
        img[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)] = timg
        mask[:, :, ((X - h)*scale // 2):((X - h)*scale // 2 + h*scale), ((X - w)*scale // 2):((X - w)*scale // 2 + w*scale)].fill_(1)

        return img, mask
