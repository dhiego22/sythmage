import os
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.image.fid import FrechetInceptionDistance
from utils import r1_penalty, compute_fid_3D, mse, plot_metrics_over_epochs, normalize
from datasets import PairedMRI3DPatches
from models import (UNetGenerator3D, PatchDiscriminator3D)


def train(
   dir_base,
   dir_target,
   epochs=10,
   batch_size=4,
   lr=2e-4,
   lambda_r1=10,
   patch_size = (64,64,64),
   patches_per_volume = 32,
   save_dir="checkpoints"
):
   #Setup
   random.seed(42)
   np.random.seed(42)
   torch.manual_seed(42)
   torch.cuda.manual_seed_all(42)
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
   # Get dataset
   train_ds = PairedMRI3DPatches(dir_base=dir_base, dir_target=dir_target, patch_size=patch_size, patches_per_volume=patches_per_volume) 
   train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
   
   # Get models and parameters
   netG = UNetGenerator3D(in_ch=1, out_ch=1).to(device)
   netD = PatchDiscriminator3D(in_ch=2).to(device)
   adv_criterion = nn.MSELoss()
   l1_criterion  = nn.L1Loss() #  pixel-wise reconstruction (encourages fidelity and reduces artifacts)
   optG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
   optD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
   scaler = torch.amp.GradScaler('cuda', enabled=True) # GradScaler to safely scale gradients during mixed-precision training (only enabled on CUDA)
      
   # Initialize Losses vectors 
   generator_loss_list = []
   discriminator_loss_list = []
   
   # Initialize metrics
   mse_list = []
   psnr_list = []
   ssim_list = []
   fid_list = []
   
   # Start training loop
   for epoch in range(1, epochs+1):
       netG.train()
       netD.train()
       g_loss_sum = 0.0; d_loss_sum = 0.0
   
       # epoch metric accumulators
       mse_sum = 0.0
       psnr_sum = 0.0
       ssim_sum = 0.0
       
       for i, (src3t, tgt7t) in enumerate(train_loader):
           src3t = src3t.to(device, non_blocking=True)
           tgt7t = tgt7t.to(device, non_blocking=True) 
   
           # Train Discriminator
           optD.zero_grad(set_to_none=True)
           with torch.amp.autocast('cuda', enabled=True):
               real_input = torch.cat([src3t, tgt7t], dim=1)
               real_input.requires_grad_(True)
               pred_real = netD(real_input)
               r1 = r1_penalty(pred_real, real_input)
               d_loss_real = adv_criterion(pred_real, torch.ones_like(pred_real))
               fake_7t = netG(src3t).detach()
               fake_input = torch.cat([src3t, fake_7t], dim=1)
               pred_fake = netD(fake_input)
               d_loss_fake = adv_criterion(pred_fake, torch.zeros_like(pred_fake))
               d_loss = 0.5*(d_loss_real + d_loss_fake) + 0.5*lambda_r1*r1 # d_loss = 0.5 * (d_loss_real + d_loss_fake)
           scaler.scale(d_loss).backward(); scaler.step(optD)
          
           # Train Generator
           optG.zero_grad(set_to_none=True)
           with torch.amp.autocast('cuda', enabled=True):
               gen_7t = netG(src3t)
               pred_gen = netD(torch.cat([src3t, gen_7t], dim=1))
               g_adv = adv_criterion(pred_gen, torch.ones_like(pred_gen))
               g_l1  = l1_criterion(gen_7t, tgt7t) * lambda_r1
               g_loss = g_adv + g_l1
           scaler.scale(g_loss).backward(); scaler.step(optG); scaler.update()
           g_loss_sum += float(g_loss.item()); d_loss_sum += float(d_loss.item())
   
       ssim = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
       psnr = PeakSignalNoiseRatio(data_range=2.0).to(device)
   
       with torch.no_grad():
           mse_sum += mse(gen_7t, tgt7t).item()
           psnr_sum += psnr(gen_7t, tgt7t).item()
           ssim_sum += ssim(gen_7t, tgt7t).item()
                       
       # epoch averages
       avg_mse = mse_sum / len(train_loader)
       avg_psnr = psnr_sum / len(train_loader)
       avg_ssim = ssim_sum / len(train_loader)
       
       mse_list.append(avg_mse)
       psnr_list.append(avg_psnr)
       ssim_list.append(avg_ssim)
       
       avg_d = g_loss_sum / max(1, len(train_loader))
       avg_g = d_loss_sum / max(1, len(train_loader))
       
       # Append losses
       discriminator_loss_list.append(avg_d)
       generator_loss_list.append(avg_g)
       
       # Compute FID   
       fid_metric = FrechetInceptionDistance(feature=64).to(device)
       fid = compute_fid_3D(netG, train_loader, fid_metric, device, num_slices=8)
       fid_list.append(fid)
   
       best_epoch = epoch
       best_netG = netG.state_dict()
       best_netD = netD.state_dict()
       print(f'Epoch={epoch} - avg_g={avg_g:.4f} - avg_d={avg_d:.4f} - fid={fid:.3f} - mse={avg_mse:.3f} - psnr={avg_psnr:.3f} - ssim={avg_ssim:.3f}')
   
      if epoch % 10 == 0: 
         print('saving weights!') 
         checkpoint_path = os.path.join(save_dir, f'checkpoint_{epoch}.pt')
         torch.save({
             'epoch': best_epoch,
             'netG': best_netG,
             'netD': best_netD,
             'optG': optG.state_dict(),
             'optD': optD.state_dict(),
         }, checkpoint_path)

   def normalize_metrics(x):
       mean_x = sum(x) / len(x)
       variance = sum((v - mean_x) ** 2 for v in x) / len(x)
       std_x = variance ** 0.5
       # avoid division by zero if all values are identical
       if std_x == 0:
           return [0.0 for _ in x]
       return [(v - mean_x) / std_x for v in x]
   plot_metrics_over_epochs(normalize_metrics(generator_loss_list), 
                           normalize_metrics(discriminator_loss_list),
                           normalize_metrics(mse_list),
                           normalize_metrics(psnr_list),
                           normalize_metrics(ssim_list),
                           normalize_metrics(fid_list),
                           labels=['Generator Loss', 'Discriminator Loss', 'MSE', 'PSNR', 'SSIM', 'FID'], 
                           title="Training Metrics Over Epochs (Normalized)",
                           y_label="Metrics Scores", 
                           start_epoch=1, 
                           epoch_step=1,
                           styles=None, 
                           save_path='training_metrics.png', show=False) 
