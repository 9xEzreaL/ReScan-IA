import copy

import numpy as np
import torch
import tqdm

from core.base_model import BaseModel
from core.logger import LogTracker


class EMA:
    def __init__(self, beta=0.9999):
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            ma_params.data = self.update_average(ma_params.data, current_params.data)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class ReScanIA(BaseModel):
    def __init__(self, networks, losses, sample_num, task, optimizers, ema_scheduler=None, **kwargs):
        super(ReScanIA, self).__init__(**kwargs)

        self.loss_fn = losses[0]
        self.netG = networks[0]
        if ema_scheduler is not None:
            self.ema_scheduler = ema_scheduler
            self.netG_EMA = copy.deepcopy(self.netG)
            self.EMA = EMA(beta=self.ema_scheduler['ema_decay'])
        else:
            self.ema_scheduler = None

        self.netG = self.set_device(self.netG, distributed=self.opt['distributed'])
        if self.ema_scheduler is not None:
            self.netG_EMA = self.set_device(self.netG_EMA, distributed=self.opt['distributed'])
        self.load_networks()

        self.optG = torch.optim.Adam(
            list(filter(lambda p: p.requires_grad, self.netG.parameters())),
            **optimizers[0],
        )
        self.optimizers.append(self.optG)
        self.resume_training()

        net = self.netG.module if self.opt['distributed'] else self.netG
        net.set_loss(self.loss_fn)
        net.set_new_noise_schedule(phase=self.phase)

        self.train_metrics = LogTracker(*[m.__name__ for m in losses], phase='train')
        self.val_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='val')
        self.test_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='test')

        self.sample_num = sample_num
        self.task = task
        self.is_3d = bool(getattr(net, 'is_3d', False))

    def _net(self):
        return self.netG.module if self.opt['distributed'] else self.netG

    def set_input(self, data):
        self.cond_image = self.set_device(data.get('cond_image'))
        self.gt_image = self.set_device(data.get('gt_image'))
        self.mask = self.set_device(data.get('mask'))

        mask_image = data.get('mask_image')
        if mask_image is not None and not isinstance(mask_image, torch.Tensor):
            if isinstance(mask_image, np.ndarray):
                mask_image = torch.from_numpy(mask_image).float()
        self.mask_image = mask_image

        seg = data.get('seg')
        self.seg = self.set_device(seg) if seg is not None else None

        vessel_seg = data.get('vessel_seg')
        self.vessel_seg = self.set_device(vessel_seg) if vessel_seg is not None else None

        self.path = data['path']
        self.batch_size = len(data['path'])

    def _normalize_visual(self, tensor):
        tensor = tensor.detach().float().cpu()
        if self.is_3d:
            tensor = torch.clamp(tensor, -3.0, 3.0)
            return (tensor + 3.0) / 6.0
        return (tensor + 1.0) / 2.0

    def get_current_visuals(self, phase='train'):
        visuals = {
            'gt_image': self._normalize_visual(self.gt_image),
            'cond_image': self._normalize_visual(self.cond_image),
            'mask': self.mask.detach().float().cpu(),
            'mask_image': (
                self._normalize_visual(self.mask_image) if self.mask_image is not None else None
            ),
        }
        if phase != 'train':
            visuals['output'] = self._normalize_visual(self.output)
        if self.seg is not None:
            visuals['seg'] = self.seg.detach().float().cpu()
        if self.vessel_seg is not None:
            visuals['vessel_seg'] = self.vessel_seg.detach().float().cpu()
        return visuals

    def save_current_results(self):
        ret_path = []
        ret_result = []
        for idx in range(self.batch_size):
            ret_path.append('GT_{}'.format(self.path[idx]))
            ret_result.append(self.gt_image[idx].detach().float().cpu())

            if hasattr(self, 'visuals') and self.visuals is not None:
                ret_path.append('Process_{}'.format(self.path[idx]))
                ret_result.append(self.visuals[idx::self.batch_size].detach().float().cpu())

            if hasattr(self, 'output') and self.output is not None:
                ret_path.append('Out_{}'.format(self.path[idx]))
                ret_result.append(self.output[idx].detach().float().cpu())

            if self.mask_image is not None:
                ret_path.append('Mask_{}'.format(self.path[idx]))
                if isinstance(self.mask_image, torch.Tensor):
                    ret_result.append(self.mask_image[idx].detach().float().cpu())
                else:
                    ret_result.append(self.mask_image[idx])

            if self.seg is not None:
                ret_path.append('Seg_{}'.format(self.path[idx]))
                ret_result.append(self.seg[idx].detach().float().cpu())
            if self.vessel_seg is not None:
                ret_path.append('VesselSeg_{}'.format(self.path[idx]))
                ret_result.append(self.vessel_seg[idx].detach().float().cpu())

        self.results_dict = self.results_dict._replace(name=ret_path, result=ret_result)
        return self.results_dict._asdict()

    def _log_visuals(self, phase):
        for key, value in self.get_current_visuals(phase=phase).items():
            if value is None:
                continue
            if self.is_3d and value.dim() == 5:
                d_mid = value.shape[2] // 2
                value_2d = value[:, :, d_mid, :, :]
                if value_2d.shape[1] == 1:
                    value_2d = value_2d.repeat(1, 3, 1, 1)
                self.writer.add_images(key, value_2d)
            else:
                self.writer.add_images(key, value)

    def _run_restoration(self):
        return self._net().restoration(
            self.cond_image,
            y_0=self.gt_image,
            mask=self.mask,
            seg=self.seg,
            vessel_seg=self.vessel_seg,
            sample_num=self.sample_num,
        )

    def _eval_batch(self, phase, metrics_tracker, epoch_or_iter):
        self.output, self.visuals = self._run_restoration()
        self.iter += self.batch_size
        self.writer.set_iter(epoch_or_iter, self.iter, phase=phase)

        for met in self.metrics:
            key = met.__name__
            value = met(self.gt_image, self.output)
            metrics_tracker.update(key, value)
            self.writer.add_scalar(key, value)

        self._log_visuals(phase=phase)
        self.writer.save_images(self.save_current_results())

    def train_step(self):
        self.netG.train()
        self.train_metrics.reset()
        for train_data in tqdm.tqdm(self.phase_loader):
            self.set_input(train_data)
            self.optG.zero_grad()
            loss = self.netG(
                self.gt_image,
                self.cond_image,
                mask=self.mask,
                seg=self.seg,
                vessel_seg=self.vessel_seg,
            )
            loss.backward()
            self.optG.step()

            self.iter += self.batch_size
            self.writer.set_iter(self.epoch, self.iter, phase='train')
            self.train_metrics.update(self.loss_fn.__name__, loss.item())

            if self.iter % self.opt['train']['log_iter'] == 0:
                for key, value in self.train_metrics.result().items():
                    self.logger.info('{:5s}: {}\t'.format(str(key), value))
                    self.writer.add_scalar(key, value)
                self._log_visuals(phase='train')

            if self.ema_scheduler is not None:
                if (
                    self.iter > self.ema_scheduler['ema_start']
                    and self.iter % self.ema_scheduler['ema_iter'] == 0
                ):
                    self.EMA.update_model_average(self.netG_EMA, self.netG)

        for scheduler in self.schedulers:
            scheduler.step()
        return self.train_metrics.result()

    def val_step(self):
        self.netG.eval()
        self.val_metrics.reset()
        with torch.no_grad():
            for val_data in tqdm.tqdm(self.val_loader):
                self.set_input(val_data)
                self._eval_batch(phase='val', metrics_tracker=self.val_metrics, epoch_or_iter=self.epoch)
        return self.val_metrics.result()

    def test(self, iters=1):
        self.netG.eval()
        self.test_metrics.reset()
        with torch.no_grad():
            for it in range(iters):
                for phase_data in tqdm.tqdm(self.phase_loader):
                    self.set_input(phase_data)
                    self._eval_batch(
                        phase='test',
                        metrics_tracker=self.test_metrics,
                        epoch_or_iter=it,
                    )

                test_log = self.test_metrics.result()
                test_log.update({'epoch': self.epoch, 'iters': self.iter})
                for key, value in test_log.items():
                    self.logger.info('{:5s}: {}\t'.format(str(key), value))

    def load_networks(self):
        netG_label = self._net().__class__.__name__
        self.load_network(network=self.netG, network_label=netG_label, strict=False)
        if self.ema_scheduler is not None:
            self.load_network(
                network=self.netG_EMA,
                network_label=netG_label + '_ema',
                strict=False,
            )

    def save_everything(self):
        netG_label = self._net().__class__.__name__
        self.save_network(network=self.netG, network_label=netG_label)
        if self.ema_scheduler is not None:
            self.save_network(network=self.netG_EMA, network_label=netG_label + '_ema')
        self.save_training_state()
