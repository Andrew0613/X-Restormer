import torch
from torch.nn import functional as F
from basicsr.models import lr_scheduler as lr_scheduler
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel
from basicsr.archs import build_network
from basicsr.metrics import calculate_metric
from basicsr.losses import build_loss
from basicsr.utils import get_root_logger, imwrite, tensor2img
from torch.optim.lr_scheduler import _LRScheduler
import math
from tqdm import tqdm
import os.path as osp
@MODEL_REGISTRY.register()
class RestormerModel(SRModel):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
    def setup_schedulers(self):
        """Set up schedulers."""
        train_opt = self.opt['train']
        scheduler_type = train_opt['scheduler'].pop('type')
        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.MultiStepRestartLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingRestartLR':
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.CosineAnnealingRestartLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingRestartCyclicLR':
            for optimizer in self.optimizers:
                self.schedulers.append(CosineAnnealingRestartCyclicLR(optimizer, **train_opt['scheduler']))
        else:
            raise NotImplementedError(f'Scheduler {scheduler_type} is not implemented yet.')
    def pre_process(self):
        # pad to multiplication of window_size
        padding_size = 64 if 'padding_size' not in self.opt else self.opt['padding_size']
        self.scale = self.opt.get('scale', 1)
        self.mod_pad_h, self.mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % padding_size != 0:
            self.mod_pad_h = padding_size - h % padding_size
        if w % padding_size != 0:
            self.mod_pad_w = padding_size - w % padding_size
        self.lq = F.pad(self.lq, (0, self.mod_pad_w, 0, self.mod_pad_h), 'reflect')
    def tile_process(self):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = self.lq.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        self.output = self.lq.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.opt['tile']['tile_size'])
        tiles_y = math.ceil(height / self.opt['tile']['tile_size'])

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.opt['tile']['tile_size']
                ofs_y = y * self.opt['tile']['tile_size']
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.opt['tile']['tile_size'], width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.opt['tile']['tile_size'], height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.opt['tile']['tile_pad'], 0)
                input_end_x_pad = min(input_end_x + self.opt['tile']['tile_pad'], width)
                input_start_y_pad = max(input_start_y - self.opt['tile']['tile_pad'], 0)
                input_end_y_pad = min(input_end_y + self.opt['tile']['tile_pad'], height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = self.lq[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                try:
                    if hasattr(self, 'net_g_ema'):
                        self.net_g_ema.eval()
                        with torch.no_grad():
                            output_tile = self.net_g_ema(input_tile)
                    else:
                        self.net_g.eval()
                        with torch.no_grad():
                            output_tile = self.net_g(input_tile)
                except RuntimeError as error:
                    print('Error', error)
                print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

                # output tile area on total image
                output_start_x = input_start_x * self.opt['scale']
                output_end_x = input_end_x * self.opt['scale']
                output_start_y = input_start_y * self.opt['scale']
                output_end_y = input_end_y * self.opt['scale']

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.opt['scale']
                output_end_x_tile = output_start_x_tile + input_tile_width * self.opt['scale']
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.opt['scale']
                output_end_y_tile = output_start_y_tile + input_tile_height * self.opt['scale']

                # put tile into output image
                self.output[:, :, output_start_y:output_end_y,
                            output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                                                       output_start_x_tile:output_end_x_tile]
    def post_process(self):
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - self.mod_pad_h * self.scale, 0:w - self.mod_pad_w * self.scale]

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)

            self.pre_process()

            if 'tile' in self.opt:
                self.tile_process()
            else:
                self.test()

            self.post_process()
            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        # save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                        #                          f'{img_name}_{self.opt["name"]}.png')
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

def get_position_from_periods(iteration, cumulative_period):
    """Get the position from a period list.

    It will return the index of the right-closest number in the period list.
    For example, the cumulative_period = [100, 200, 300, 400],
    if iteration == 50, return 0;
    if iteration == 210, return 2;
    if iteration == 300, return 2.

    Args:
        iteration (int): Current iteration.
        cumulative_period (list[int]): Cumulative period list.

    Returns:
        int: The position of the right-closest number in the period list.
    """
    for i, period in enumerate(cumulative_period):
        if iteration <= period:
            return i
class CosineAnnealingRestartCyclicLR(_LRScheduler):
    """ Cosine annealing with restarts learning rate scheme.
    An example of config:
    periods = [10, 10, 10, 10]
    restart_weights = [1, 0.5, 0.5, 0.5]
    eta_min=1e-7
    It has four cycles, each has 10 iterations. At 10th, 20th, 30th, the
    scheduler will restart with the weights in restart_weights.
    Args:
        optimizer (torch.nn.optimizer): Torch optimizer.
        periods (list): Period for each cosine anneling cycle.
        restart_weights (list): Restart weights at each restart iteration.
            Default: [1].
        eta_min (float): The mimimum lr. Default: 0.
        last_epoch (int): Used in _LRScheduler. Default: -1.
    """

    def __init__(self,
                 optimizer,
                 periods,
                 restart_weights=(1, ),
                 eta_mins=(0, ),
                 last_epoch=-1):
        self.periods = periods
        self.init_preiods = periods
        self.restart_weights = restart_weights
        self.eta_mins = eta_mins
        assert (len(self.periods) == len(self.restart_weights)
                ), 'periods and restart_weights should have the same length.'
        self.cumulative_period = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]
        super(CosineAnnealingRestartCyclicLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        idx = get_position_from_periods(self.last_epoch,
                                        self.cumulative_period)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        current_period = self.periods[idx]
        eta_min = self.eta_mins[idx]

        return [
            eta_min + current_weight * 0.5 * (base_lr - eta_min) *
            (1 + math.cos(math.pi * (
                (self.last_epoch - nearest_restart) / current_period)))
            for base_lr in self.base_lrs
        ]