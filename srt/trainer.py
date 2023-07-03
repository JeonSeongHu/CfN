import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

import srt.utils.visualize as vis
from srt.utils.common import mse2psnr, reduce_dict, gather_all
from srt.utils import nerf
from srt.utils.common import get_rank, get_world_size

import matplotlib.pyplot as plt
import os
import math
from collections import defaultdict
from random import randint

class SRTTrainer:
    def __init__(self, generator, discriminator, generator_optimizer, discriminator_optimizer, cfg, device, out_dir, render_kwargs):
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.config = cfg
        self.device = device
        self.show = False
        self.out_dir = out_dir
        self.render_kwargs = render_kwargs
        if 'num_coarse_samples' in cfg['training']:
            self.render_kwargs['num_coarse_samples'] = cfg['training']['num_coarse_samples']
        if 'num_fine_samples' in cfg['training']:
            self.render_kwargs['num_fine_samples'] = cfg['training']['num_fine_samples']

    def evaluate(self, val_loader, **kwargs):
        ''' Performs an evaluation.
        Args:
            val_loader (dataloader): pytorch dataloader
        '''
        self.generator.eval()
        eval_lists = defaultdict(list)

        loader = val_loader if get_rank() > 0 else tqdm(val_loader)
        sceneids = []

        for data in loader:
            sceneids.append(data['sceneid'])
            eval_step_dict = self.eval_step(data, **kwargs)

            for k, v in eval_step_dict.items():
                eval_lists[k].append(v)

        sceneids = torch.cat(sceneids, 0).cuda()
        sceneids = torch.cat(gather_all(sceneids), 0)

        print(f'Evaluated {len(torch.unique(sceneids))} unique scenes.')

        eval_dict = {k: torch.cat(v, 0) for k, v in eval_lists.items()}
        eval_dict = reduce_dict(eval_dict, average=True)  # Average across processes
        eval_dict = {k: v.mean().item() for k, v in eval_dict.items()}  # Average across batch_size
        print('Evaluation results:')
        return eval_dict

    def train_step(self, data, it): ## data = 1 batch를 받아 학습함.
        self.generator.train()
        self.discriminator.train()

        loss, d_loss, g_loss, loss_terms = self.compute_loss(data, it)
        loss = loss.mean(0)
        loss_terms = {k: v.mean(0).item() for k, v in loss_terms.items()}

        self.discriminator_optimizer.zero_grad()
        d_loss.backward()
        self.discriminator_optimizer.step()

        self.generator_optimizer.zero_grad()
        g_loss.backward()
        self.generator_optimizer.step()

        return loss.item(), loss_terms

    def compute_loss(self, data, it):
        # self.visualize(data, mode="helpme")
        device = self.device

        input_images = data.get('input_images').to(device)
        target_images = data.get('target_images').to(device).permute(0,1,3,4,2)
        input_camera_pos = data.get('input_camera_pos').to(device)
        input_rays = data.get('input_rays').to(device)
        target_camera_pos = data.get('target_camera_pos').to(device)
        target_rays = data.get('target_rays').to(device)
        # target_pixels = data.get('target_pixels').to(device)

        # print(input_rays.shape, target_rays.shape)


        z = self.generator.encoder(input_images, input_camera_pos, input_rays)

        loss = 0.
        loss_terms = dict()
        # pred_pixels, extras = self.generator.decoder(z, target_camera_pos, target_rays, **self.render_kwargs)
        # print(z.shape)
        # print(target_images.shape)
        # print(input_camera_pos.shape)

        # pred_imgs, extras = self.render_image_batch(z, target_camera_pos, target_rays, **self.render_kwargs)
        # logits_fake = self.discriminator(pred_imgs)
        # logits_real = self.discriminator(target_image)
        # loss + ((pred_imgs - input_images)**2).mean((1, 2))
        # loss_terms['mse'] = loss

        # i = randint(0, 4)
        #
        # target_image = target_images[:, i]
        # pred_img, extras = self.render_image(z, target_camera_pos[:, i], target_rays[:, i], **self.render_kwargs)
        # pred_img, extras = self.render_image_batch(z, target_camera_pos, target_rays, **self.render_kwargs)

        pred_imgs = torch.zeros_like(target_images)

        for i in range(pred_imgs.shape[1]):
            # print(f"z:{z.shape}\ntarget_camera_pos:{target_camera_pos[:, i].shape}\ntarget_rays:{target_rays[:, i].shape}")
            pred_imgs[:, i], extras = self.render_image(z=z, camera_pos=target_camera_pos[:, i], rays=target_rays[:, i], **self.render_kwargs)
            # temp, extras = self.render_image(z=z, camera_pos=target_camera_pos[:, i], rays=target_rays[:, i],**self.render_kwargs)
            # plt.imshow(temp[i].cpu().detach().numpy())
            # plt.show()
            # plt.imshow(target_images[:, i][i].cpu().detach().numpy())
            # plt.show()

        # print(pred_imgs.shape)


        criterion = nn.BCELoss()

        ### Train Generator
        # print(pred_imgs.shape, target_images.shape)
        mse = nn.L1Loss()(pred_imgs, target_images).mean()

        fake_logits = []

        for i in range(pred_imgs.shape[1]):
            fake_logits.append(self.discriminator(pred_imgs[:, i].permute(0, 3, 1, 2)))
        fake_logits = torch.vstack(fake_logits)
        gen_labels = torch.ones_like(fake_logits, requires_grad=False)

        a, b = 0.00005, 1
        g_loss = a * criterion(fake_logits, gen_labels) + b * mse

        ### Train Discriminator

        real_logits = self.discriminator(target_images.flatten(0,1).permute(0, 3, 1, 2).detach())
        fake_logits = self.discriminator(pred_imgs.flatten(0,1).permute(0, 3, 1, 2).detach())

        real_labels = torch.ones_like(real_logits, requires_grad=False)
        fake_labels = torch.zeros_like(fake_logits, requires_grad=False)

        d_loss_real = criterion(real_logits, real_labels)
        d_loss_fake = criterion(fake_logits, fake_labels)

        d_loss = d_loss_real + d_loss_fake

        loss_terms['g_loss'] = g_loss
        loss_terms['d_loss'] = d_loss
        loss_terms['mse'] = mse
        loss = d_loss + g_loss

        if self.show == True:
            num_images = 5  # Change this value to the number of images you want to display (up to 10)
            fig = plt.figure(figsize=(12, 6))
            rows, cols = num_images, 2

            for i in range(num_images):
                ax1 = fig.add_subplot(rows, cols, i * 2 + 1)
                # print(pred_imgs.shape)
                # print(pred_imgs[:, i].shape)
                ax1.imshow(pred_imgs[:, i][i].cpu().detach().numpy())
                ax1.set_axis_off()
                ax2 = fig.add_subplot(rows, cols, i * 2 + 2)
                ax2.imshow(target_images[:, i][i].cpu().detach().numpy())
                ax2.set_axis_off()

            plt.tight_layout()
            plt.show(block=False)
            plt.pause(4)
            plt.close()
            # print(pred_imgs[0], target_images[0])
            self.show = False
        return loss, d_loss, g_loss, loss_terms

    def eval_step(self, data, full_scale=False):
        with torch.no_grad():
            loss, d_loss, g_loss, loss_terms = self.compute_loss(data, 1000000)

        mse = loss_terms['mse']
        psnr = mse2psnr(mse)
        return {'psnr': psnr, 'mse': mse, **loss_terms}


    def render_image(self, z, camera_pos, rays, **render_kwargs):
        """
        Args:
            z [n, k, c]: set structured latent variables
            camera_pos [n, 3]: camera position
            rays [n, h, w, 3]: ray directions
            render_kwargs: kwargs passed on to decoder
        """
        batch_size, height, width = rays.shape[:3]
        rays = rays.flatten(1, 2)
        camera_pos = camera_pos.unsqueeze(1).repeat(1, rays.shape[1], 1)

        max_num_rays = self.config['data']['num_points'] * \
                self.config['training']['batch_size'] // (rays.shape[0] * get_world_size())

        num_rays = rays.shape[1]
        img = torch.zeros_like(rays)

        all_extras = []
        for i in range(0, num_rays, max_num_rays):
            img[:, i:i+max_num_rays], extras = self.generator.decoder(
                z=z, x=camera_pos[:, i:i+max_num_rays], rays=rays[:, i:i+max_num_rays],
                **render_kwargs)
            all_extras.append(extras)

        agg_extras = {}
        for key in all_extras[0]:
            agg_extras[key] = torch.cat([extras[key] for extras in all_extras], 1)
            agg_extras[key] = agg_extras[key].view(batch_size, height, width, -1)

        img = img.view(img.shape[0], height, width, 3)
        return img, agg_extras


    def visualize(self, data, it,mode='val'):
        device = "cuda"
        self.generator.eval()
        self.generator = self.generator.to(device)

        with torch.no_grad():
            input_images = data.get('input_images').to(device)
            input_camera_pos = data.get('input_camera_pos').to(device)
            input_rays = data.get('input_rays').to(device)

            camera_pos_base = input_camera_pos[:, 0]
            input_rays_base = input_rays[:, 0]
            if 'transform' in data:
                # If the data is transformed in some different coordinate system, where
                # rotating around the z axis doesn't make sense, we first undo this transform,
                # then rotate, and then reapply it.

                transform = data['transform'].to(device)
                inv_transform = torch.inverse(transform)
                camera_pos_base = nerf.transform_points_torch(camera_pos_base, inv_transform)
                input_rays_base = nerf.transform_points_torch(
                    input_rays_base, inv_transform.unsqueeze(1).unsqueeze(2), translate=False)
            else:
                transform = None

            input_images_np = np.transpose(input_images.cpu().numpy(), (0, 1, 3, 4, 2))

            z = self.generator.encoder(input_images, input_camera_pos, input_rays)

            batch_size, num_input_images, height, width, _ = input_rays.shape

            num_angles = 6

            columns = []
            for i in range(num_input_images):
                header = 'input' if num_input_images == 1 else f'input {i+1}'
                columns.append((header, input_images_np[:, i], 'image'))

            all_extras = []
            for i in range(num_angles):
                angle = i * (2 * math.pi / num_angles)
                angle_deg = (i * 360) // num_angles

                camera_pos_rot = nerf.rotate_around_z_axis_torch(camera_pos_base, angle)
                rays_rot = nerf.rotate_around_z_axis_torch(input_rays_base, angle)

                if transform is not None:
                    camera_pos_rot = nerf.transform_points_torch(camera_pos_rot, transform)
                    rays_rot = nerf.transform_points_torch(
                        rays_rot, transform.unsqueeze(1).unsqueeze(2), translate=False)
                # print(f"camera_pos_rot:{camera_pos_rot.shape}\nz:{z.shape}\nrays_rot:{rays_rot.shape}")
                img, extras = self.render_image(z, camera_pos_rot, rays_rot, **self.render_kwargs)
                all_extras.append(extras)
                columns.append((f'render {angle_deg}°', img.cpu().numpy(), 'image'))

            for i, extras in enumerate(all_extras):
                if 'depth' in extras:
                    depth_img = extras['depth'].unsqueeze(-1) / self.render_kwargs['max_dist']
                    depth_img = depth_img.view(batch_size, height, width, 1)
                    columns.append((f'depths {angle_deg}°', depth_img.cpu().numpy(), 'image'))

            output_img_path = os.path.join(self.out_dir, f'renders-{mode}-{it}')
            vis.draw_visualization_grid(columns, output_img_path)
#