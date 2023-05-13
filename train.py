"""Train pi-GAN. Supports distributed training."""

import argparse
import os
import numpy as np
import math

from collections import deque

import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image

from generators import generators
from discriminators import discriminators
from srt import data
from srt.model import SRT
from srt.trainer import SRTTrainer
from srt.checkpoint import Checkpoint
from srt.utils.common import init_ddp
from siren import siren
import fid_evaluation

import datasets
import curriculums
from tqdm import tqdm
from datetime import datetime
import copy
import yaml

from torch_ema import ExponentialMovingAverage

def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()

def load_images(images, curriculum, device):
    return_images = []
    head = 0
    for stage in curriculum['stages']:
        stage_images = images[head:head + stage['batch_size']]
        stage_images = F.interpolate(stage_images, size=stage['img_size'],  mode='bilinear', align_corners=True)
        return_images.append(stage_images)
        head += stage['batch_size']
    return return_images


def z_sampler(shape, device, dist):
    if dist == 'gaussian':
        z = torch.randn(shape, device=device)
    elif dist == 'uniform':
        z = torch.rand(shape, device=device) * 2 - 1
    return z


def train(rank, world_size, opt):
    torch.manual_seed(0)

    # cfg (config.yaml)
    with open(opt.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.CLoader)

    setup(rank, world_size, opt.port)
    device = torch.device(rank)

    #wandb?
    opt.wandb = opt.wandb and rank == 0  # Only log to wandb in main process

    #max iteration -> pi gan에선 필요 없을듯
    if opt.exit_after is not None:
        max_it = opt.exit_after
    elif 'max_it' in cfg['training']:
        max_it = cfg['training']['max_it']
    else:
        max_it = 1000000

    # -> pi gan에선 필요 없을듯 22
    exp_name = os.path.basename(os.path.dirname(opt.config))
    if opt.rtpt is not None:
        from rtpt import RTPT
        rtpt = RTPT(name_initials=opt.rtpt, experiment_name=exp_name, max_iterations=max_it)

    # out_dir = os.path.dirname(args.config) -> opt.output_dir

    # config.yaml 쓸 거면 metadata['batch_size'] 말고 이거 써야할 듯
    batch_size = cfg['training']['batch_size'] // world_size

    model_selection_metric = cfg['training']['model_selection_metric']
    if cfg['training']['model_selection_mode'] == 'maximize':
        model_selection_sign = 1
    elif cfg['training']['model_selection_mode'] == 'minimize':
        model_selection_sign = -1
    else:
        raise ValueError('model_selection_mode must be either maximize or minimize.')


    curriculum = getattr(curriculums, opt.curriculum)
    metadata = curriculums.extract_metadata(curriculum, 0)

    fixed_z = z_sampler((25, 256), device='cpu', dist=metadata['z_dist'])

    SIREN = getattr(siren, metadata['model'])

    CHANNELS = 3

    scaler = torch.cuda.amp.GradScaler()

    if opt.load_dir != '':
        generator = torch.load(os.path.join(opt.load_dir, 'generator.pth'), map_location=device)
        discriminator = torch.load(os.path.join(opt.load_dir, 'discriminator.pth'), map_location=device)
        ema = torch.load(os.path.join(opt.load_dir, 'ema.pth'), map_location=device)
        ema2 = torch.load(os.path.join(opt.load_dir, 'ema2.pth'), map_location=device)
    else:
        #generator = getattr(generators, metadata['generator'])(SIREN, metadata['latent_dim']).to(device)
        generator = SRT(cfg['model']).train().to(device)
        discriminator = getattr(discriminators, metadata['discriminator'])().to(device)
        ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)
        ema2 = ExponentialMovingAverage(generator.parameters(), decay=0.9999)

    if world_size > 1:
        generator.encoder = DDP(generator.encoder, device_ids=[rank], output_device=rank)
        generator.decoder = DDP(generator.decoder, device_ids=[rank], output_device=rank)
        encoder_module = generator.encoder.module
        decoder_module = generator.decoder.module
    else:
        encoder_module = generator.encoder
        decoder_module = generator.decoder

    generator_ddp = DDP(generator, device_ids=[rank], find_unused_parameters=True)
    discriminator_ddp = DDP(discriminator, device_ids=[rank], find_unused_parameters=True, broadcast_buffers=False)
    generator = generator_ddp.module
    discriminator = discriminator_ddp.module

    if metadata.get('unique_lr', False):
        mapping_network_param_names = [name for name, _ in generator_ddp.module.siren.mapping_network.named_parameters()]
        mapping_network_parameters = [p for n, p in generator_ddp.named_parameters() if n in mapping_network_param_names]
        generator_parameters = [p for n, p in generator_ddp.named_parameters() if n not in mapping_network_param_names]
        optimizer_G = torch.optim.Adam([{'params': generator_parameters, 'name': 'generator'},
                                        {'params': mapping_network_parameters, 'name': 'mapping_network', 'lr':metadata['gen_lr']*5e-2}],
                                       lr=metadata['gen_lr'], betas=metadata['betas'], weight_decay=metadata['weight_decay'])
    else:
        optimizer_G = torch.optim.Adam(generator_ddp.parameters(), lr=metadata['gen_lr'], betas=metadata['betas'], weight_decay=metadata['weight_decay'])

    optimizer_D = torch.optim.Adam(discriminator_ddp.parameters(), lr=metadata['disc_lr'], betas=metadata['betas'], weight_decay=metadata['weight_decay'])

    if opt.load_dir != '':
        optimizer_G.load_state_dict(torch.load(os.path.join(opt.load_dir, 'optimizer_G.pth')))
        optimizer_D.load_state_dict(torch.load(os.path.join(opt.load_dir, 'optimizer_D.pth')))
        if not metadata.get('disable_scaler', False):
            scaler.load_state_dict(torch.load(os.path.join(opt.load_dir, 'scaler.pth')))

    generator_losses = []
    discriminator_losses = []

    if opt.set_step != None:
        generator.step = opt.set_step
        discriminator.step = opt.set_step

    if metadata.get('disable_scaler', False):
        scaler = torch.cuda.amp.GradScaler(enabled=False)

    #generator.set_device(device)

    # ----------
    #  Training
    # ----------

    with open(os.path.join(opt.output_dir, 'options.txt'), 'w') as f:
        f.write(str(opt))
        f.write('\n\n')
        f.write(str(generator))
        f.write('\n\n')
        f.write(str(discriminator))
        f.write('\n\n')
        f.write(str(curriculum))

    # Initialize datasets
    print('Loading training set...')
    train_dataset = data.get_dataset('train', cfg['data'])
    eval_split = 'test' if opt.test else 'val'
    print(f'Loading {eval_split} set...')
    eval_dataset = data.get_dataset(eval_split, cfg['data'],
                                    max_len=opt.max_eval, full_scale=opt.full_scale)

    num_workers = cfg['training']['num_workers'] if 'num_workers' in cfg['training'] else 1
    print(f'Using {num_workers} workers per process for data loading.')

    # Initialize data loaders
    train_sampler = val_sampler = None
    shuffle = False
    if isinstance(train_dataset, torch.utils.data.IterableDataset):
        assert num_workers == 1, "Our MSN dataset is implemented as Tensorflow iterable, and does not currently support multiple PyTorch workers per process. Is also shouldn't need any, since Tensorflow uses multiple workers internally."
    else:
        if world_size >= 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, shuffle=True, drop_last=False)
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                eval_dataset, shuffle=True, drop_last=False)
        else:
            shuffle = True

    torch.manual_seed(rank)
    dataloader = None

    # loader for SRT
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
        sampler=train_sampler, shuffle=shuffle,
        worker_init_fn=data.worker_init_fn, persistent_workers=True)

    val_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=max(1, batch_size // 8), num_workers=1, 
        sampler=val_sampler, shuffle=shuffle,
        pin_memory=False, worker_init_fn=data.worker_init_fn, persistent_workers=True)
    
    dataloader = train_loader

    # Loaders for visualization scenes
    vis_loader_val = torch.utils.data.DataLoader(
        eval_dataset, batch_size=12, shuffle=shuffle, worker_init_fn=data.worker_init_fn)
    vis_loader_train = torch.utils.data.DataLoader(
        train_dataset, batch_size=12, shuffle=shuffle, worker_init_fn=data.worker_init_fn)
    print('Data loaders initialized.')

    data_vis_val = next(iter(vis_loader_val))  # Validation set data for visualization
    train_dataset.mode = 'val'  # Get validation info from training set just this once
    data_vis_train = next(iter(vis_loader_train))  # Validation set data for visualization
    train_dataset.mode = 'train'
    print('Visualization data loaded.')
    # ~

    total_progress_bar = tqdm(total = opt.n_epochs, desc = "Total progress", dynamic_ncols=True)
    total_progress_bar.update(discriminator.epoch)
    interior_step_bar = tqdm(dynamic_ncols=True)
    #
    epoch_it = 1
    train_sampler.set_epoch(epoch_it)

    print_every = cfg['training']['print_every']
    checkpoint_every = cfg['training']['checkpoint_every']
    validate_every = cfg['training']['validate_every']
    visualize_every = cfg['training']['visualize_every']
    backup_every = cfg['training']['backup_every']

    for _ in range (opt.n_epochs):
        total_progress_bar.update(1)

        metadata = curriculums.extract_metadata(curriculum, discriminator.step)

        # Set learning rates
        for param_group in optimizer_G.param_groups:
            if param_group.get('name', None) == 'mapping_network':
                param_group['lr'] = metadata['gen_lr'] * 5e-2
            else:
                param_group['lr'] = metadata['gen_lr']
            param_group['betas'] = metadata['betas']
            param_group['weight_decay'] = metadata['weight_decay']
        for param_group in optimizer_D.param_groups:
            param_group['lr'] = metadata['disc_lr']
            param_group['betas'] = metadata['betas']
            param_group['weight_decay'] = metadata['weight_decay']

        '''
        if not dataloader or dataloader.batch_size != metadata['batch_size']:
            dataloader, CHANNELS = datasets.get_dataset_distributed(metadata['dataset'],
                                        world_size,
                                        rank,
                                        **metadata)

            step_next_upsample = curriculums.next_upsample_step(curriculum, discriminator.step)
            step_last_upsample = curriculums.last_upsample_step(curriculum, discriminator.step)

        
            interior_step_bar.reset(total=(step_next_upsample - step_last_upsample))
            interior_step_bar.set_description(f"Progress to next stage")
            interior_step_bar.update((discriminator.step - step_last_upsample))
        '''
        step_next_upsample = 0
        step_last_upsample = 0
        if not dataloader or dataloader.batch_size != batch_size:
            step_next_upsample = curriculums.next_upsample_step(curriculum, discriminator.step)
            step_last_upsample = curriculums.last_upsample_step(curriculum, discriminator.step)


            interior_step_bar.reset(total=(step_next_upsample - step_last_upsample))
            interior_step_bar.set_description(f"Progress to next stage")
            interior_step_bar.update((discriminator.step - step_last_upsample))

        # input_images, input_camera_pos, input_rays, 
        # target_pixels, target_camera_pos, target_rays, scendid, transform
        for i, (imgs, _, _, _, _, _, _, _) in enumerate(dataloader):
            if discriminator.step % opt.model_save_interval == 0 and rank == 0:
                now = datetime.now()
                now = now.strftime("%d--%H:%M--")
                #torch.save(ema, os.path.join(opt.output_dir, now + 'ema.pth'))
                #torch.save(ema2, os.path.join(opt.output_dir, now + 'ema2.pth'))
                #torch.save(generator_ddp.module, os.path.join(opt.output_dir, now + 'generator.pth'))
                #torch.save(discriminator_ddp.module, os.path.join(opt.output_dir, now + 'discriminator.pth'))
                #torch.save(optimizer_G.state_dict(), os.path.join(opt.output_dir, now + 'optimizer_G.pth'))
                #torch.save(optimizer_D.state_dict(), os.path.join(opt.output_dir, now + 'optimizer_D.pth'))
                #torch.save(scaler.state_dict(), os.path.join(opt.output_dir, now + 'scaler.pth'))
            metadata = curriculums.extract_metadata(curriculum, discriminator.step)

            #if dataloader.batch_size != metadata['batch_size']: break
            if dataloader.batch_size != batch_size: break

            if scaler.get_scale() < 1:
                scaler.update(1.)

            generator_ddp.train()
            discriminator_ddp.train()

            alpha = min(1, (discriminator.step - step_last_upsample) / (metadata['fade_steps']))

            real_imgs = imgs.to(device, non_blocking=True)

            metadata['nerf_noise'] = max(0, 1. - discriminator.step/5000.)

            # TRAIN DISCRIMINATOR
            with torch.cuda.amp.autocast():
                # Generate images for discriminator training
                with torch.no_grad():
                    z = z_sampler((real_imgs.shape[0], metadata['latent_dim']), device=device, dist=metadata['z_dist'])
                    split_batch_size = z.shape[0] // metadata['batch_split']
                    gen_imgs = []
                    gen_positions = []
                    for split in range(metadata['batch_split']):
                        subset_z = z[split * split_batch_size:(split+1) * split_batch_size]
                        #g_imgs, g_pos = generator_ddp(subset_z, **metadata)            
                        g_imgs, g_pos = trainer.visualize(data_vis_train, mode='train')

                        gen_imgs.append(g_imgs)
                        gen_positions.append(g_pos)

                    gen_imgs = torch.cat(gen_imgs, axis=0)
                    gen_positions = torch.cat(gen_positions, axis=0)

                real_imgs.requires_grad = True
                r_preds, _, _ = discriminator_ddp(real_imgs, alpha, **metadata)

            if metadata['r1_lambda'] > 0:
                # Gradient penalty
                grad_real = torch.autograd.grad(outputs=scaler.scale(r_preds.sum()), inputs=real_imgs, create_graph=True)
                inv_scale = 1./scaler.get_scale()
                grad_real = [p * inv_scale for p in grad_real][0]
            with torch.cuda.amp.autocast():
                if metadata['r1_lambda'] > 0:
                    grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                    grad_penalty = 0.5 * metadata['r1_lambda'] * grad_penalty
                else:
                    grad_penalty = 0

                g_preds, g_pred_latent, g_pred_position = discriminator_ddp(gen_imgs, alpha, **metadata)
                if metadata['z_lambda'] > 0 or metadata['pos_lambda'] > 0:
                    latent_penalty = torch.nn.MSELoss()(g_pred_latent, z) * metadata['z_lambda']
                    #position_penalty = torch.nn.MSELoss()(g_pred_position, gen_positions) * metadata['pos_lambda']
                    identity_penalty = latent_penalty #+ position_penalty
                else:
                    identity_penalty=0

                d_loss = torch.nn.functional.softplus(g_preds).mean() + torch.nn.functional.softplus(-r_preds).mean() + grad_penalty + identity_penalty
                discriminator_losses.append(d_loss.item())

            optimizer_D.zero_grad()
            scaler.scale(d_loss).backward()
            scaler.unscale_(optimizer_D)
            torch.nn.utils.clip_grad_norm_(discriminator_ddp.parameters(), metadata['grad_clip'])
            scaler.step(optimizer_D)


            # TRAIN GENERATOR
            z = z_sampler((imgs.shape[0], metadata['latent_dim']), device=device, dist=metadata['z_dist'])

            split_batch_size = z.shape[0] // metadata['batch_split']
            trainer = SRTTrainer(generator, optimizer_G, cfg, device, opt.output_dir, train_dataset.render_kwargs)

            for split in range(metadata['batch_split']):
                with torch.cuda.amp.autocast():
                    subset_z = z[split * split_batch_size:(split+1) * split_batch_size]
                    #gen_imgs, gen_positions = generator_ddp(subset_z, **metadata)
                    gen_imgs, gen_positions = trainer.visualize(data_vis_train, mode='train')
                    
                    g_preds, g_pred_latent, g_pred_position = discriminator_ddp(gen_imgs, alpha, **metadata)

                    topk_percentage = max(0.99 ** (discriminator.step/metadata['topk_interval']), metadata['topk_v']) if 'topk_interval' in metadata and 'topk_v' in metadata else 1
                    topk_num = math.ceil(topk_percentage * g_preds.shape[0])

                    g_preds = torch.topk(g_preds, topk_num, dim=0).values

                    if metadata['z_lambda'] > 0 or metadata['pos_lambda'] > 0:
                        latent_penalty = torch.nn.MSELoss()(g_pred_latent, subset_z) * metadata['z_lambda']
                        #position_penalty = torch.nn.MSELoss()(g_pred_position, gen_positions) * metadata['pos_lambda']
                        identity_penalty = latent_penalty #+ position_penalty
                    else:
                        identity_penalty = 0

                    g_loss = torch.nn.functional.softplus(-g_preds).mean() + identity_penalty
                    generator_losses.append(g_loss.item())

                scaler.scale(g_loss).backward()

            scaler.unscale_(optimizer_G)
            torch.nn.utils.clip_grad_norm_(generator_ddp.parameters(), metadata.get('grad_clip', 0.3))
            scaler.step(optimizer_G)
            scaler.update()
            optimizer_G.zero_grad()
            ema.update(generator_ddp.parameters())
            ema2.update(generator_ddp.parameters())


            if rank == 0:
                interior_step_bar.update(1)
                if i%10 == 0:
                    tqdm.write(f"[Experiment: {opt.output_dir}] [GPU: {os.environ['CUDA_VISIBLE_DEVICES']}] [Epoch: {discriminator.epoch}/{opt.n_epochs}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}] [Step: {discriminator.step}] [Alpha: {alpha:.2f}] [Img Size: {metadata['img_size']}] [Batch Size: {metadata['batch_size']}] [TopK: {topk_num}] [Scale: {scaler.get_scale()}]")

                if discriminator.step % opt.sample_interval == 0:
                    generator_ddp.eval()
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            copied_metadata['img_size'] = 128
                            gen_imgs = generator_ddp.module.staged_forward(fixed_z.to(device),  **copied_metadata)[0]
                    save_image(gen_imgs[:25], os.path.join(opt.output_dir, f"{discriminator.step}_fixed.png"), nrow=5, normalize=True)

                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            copied_metadata['h_mean'] += 0.5
                            copied_metadata['img_size'] = 128
                            gen_imgs = generator_ddp.module.staged_forward(fixed_z.to(device),  **copied_metadata)[0]
                    save_image(gen_imgs[:25], os.path.join(opt.output_dir, f"{discriminator.step}_tilted.png"), nrow=5, normalize=True)

                    ema.store(generator_ddp.parameters())
                    ema.copy_to(generator_ddp.parameters())
                    generator_ddp.eval()
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            copied_metadata['img_size'] = 128
                            gen_imgs = generator_ddp.module.staged_forward(fixed_z.to(device),  **copied_metadata)[0]
                    save_image(gen_imgs[:25], os.path.join(opt.output_dir, f"{discriminator.step}_fixed_ema.png"), nrow=5, normalize=True)

                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            copied_metadata['h_mean'] += 0.5
                            copied_metadata['img_size'] = 128
                            gen_imgs = generator_ddp.module.staged_forward(fixed_z.to(device),  **copied_metadata)[0]
                    save_image(gen_imgs[:25], os.path.join(opt.output_dir, f"{discriminator.step}_tilted_ema.png"), nrow=5, normalize=True)

                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['img_size'] = 128
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            copied_metadata['psi'] = 0.7
                            gen_imgs = generator_ddp.module.staged_forward(torch.randn_like(fixed_z).to(device),  **copied_metadata)[0]
                    save_image(gen_imgs[:25], os.path.join(opt.output_dir, f"{discriminator.step}_random.png"), nrow=5, normalize=True)

                    ema.restore(generator_ddp.parameters())

                if discriminator.step % opt.sample_interval == 0:
                    torch.save(ema, os.path.join(opt.output_dir, 'ema.pth'))
                    torch.save(ema2, os.path.join(opt.output_dir, 'ema2.pth'))
                    torch.save(generator_ddp.module, os.path.join(opt.output_dir, 'generator.pth'))
                    torch.save(discriminator_ddp.module, os.path.join(opt.output_dir, 'discriminator.pth'))
                    torch.save(optimizer_G.state_dict(), os.path.join(opt.output_dir, 'optimizer_G.pth'))
                    torch.save(optimizer_D.state_dict(), os.path.join(opt.output_dir, 'optimizer_D.pth'))
                    torch.save(scaler.state_dict(), os.path.join(opt.output_dir, 'scaler.pth'))
                    torch.save(generator_losses, os.path.join(opt.output_dir, 'generator.losses'))
                    torch.save(discriminator_losses, os.path.join(opt.output_dir, 'discriminator.losses'))

                # Visualize Output
                #if opt.visnow or (i > 0 and visualize_every > 0 and (i % visualize_every) == 0):
                #    print('Visualizing...')
                    #trainer.visualize(data_vis_val, mode='val')
                #    trainer.visualize(data_vis_train, mode='train')

            if opt.eval_freq > 0 and (discriminator.step + 1) % opt.eval_freq == 0:
                generated_dir = os.path.join(opt.output_dir, 'evaluation/generated')

                if rank == 0:
                    fid_evaluation.setup_evaluation(metadata['dataset'], generated_dir, target_size=128)
                dist.barrier()
                ema.store(generator_ddp.parameters())
                ema.copy_to(generator_ddp.parameters())
                generator_ddp.eval()
                fid_evaluation.output_images(generator_ddp, metadata, rank, world_size, generated_dir)
                ema.restore(generator_ddp.parameters())
                dist.barrier()
                if rank == 0:
                    fid = fid_evaluation.calculate_fid(metadata['dataset'], generated_dir, target_size=128)
                    with open(os.path.join(opt.output_dir, f'fid.txt'), 'a') as f:
                        f.write(f'\n{discriminator.step}:{fid}')

                torch.cuda.empty_cache()

            discriminator.step += 1
            generator.step += 1
        discriminator.epoch += 1
        #generator.epoch += 1
        epoch_it += 1

    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=3000, help="number of epochs of training")
    parser.add_argument("--sample_interval", type=int, default=200, help="interval between image sampling")
    parser.add_argument('--output_dir', type=str, default='debug')
    parser.add_argument('--load_dir', type=str, default='')
    parser.add_argument('--curriculum', type=str)
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--port', type=str, default='12355')
    parser.add_argument('--set_step', type=int, default=None)
    parser.add_argument('--model_save_interval', type=int, default=5000)

    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--exit-after', type=int, help='Exit after this many training iterations.')
    parser.add_argument('--test', action='store_true', help='When evaluating, use test instead of validation split.')
    parser.add_argument('--evalnow', action='store_true', help='Run evaluation on startup.')
    parser.add_argument('--visnow', action='store_true', help='Run visualization on startup.')
    parser.add_argument('--wandb', action='store_true', help='Log run to Weights and Biases.')
    parser.add_argument('--max-eval', type=int, help='Limit the number of scenes in the evaluation set.')
    parser.add_argument('--full-scale', action='store_true', help='Evaluate on full images.')
    parser.add_argument('--print-model', action='store_true', help='Print model and parameters on startup.')
    parser.add_argument('--rtpt', type=str, help='Use rtpt to set process name with given initials.')

    opt = parser.parse_args()
    print(opt)
    os.makedirs(opt.output_dir, exist_ok=True)
    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    mp.spawn(train, args=(num_gpus, opt), nprocs=num_gpus, join=True)
