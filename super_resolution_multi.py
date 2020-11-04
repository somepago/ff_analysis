"""
python super_resolution_multi.py  --max_images 4 --latent_size 256 --mapping_size 128 --iters 10000 --reg_lambda 1e-4 --exp base_multi
"""

import os
import numpy as np
import imageio

import torch
import torch.nn as nn
import wandb
import math


def create_grid(h, w):
    grid_y, grid_x = torch.meshgrid([torch.linspace(0, 1, steps=h),
                                     torch.linspace(0, 1, steps=w)])
    grid = torch.stack([grid_y, grid_x], dim=-1)
    return grid


def input_mapping(x, B):
    if B is None:
        return x
    else:
        x_proj = (2. * np.pi * x) @ B.t()
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=30, is_first=False, is_last=False):
        super().__init__()
        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last
        self.init_weights()

    def init_weights(self):
        b = 1 / self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        x = self.linear(x)
        return x if self.is_last else torch.sin(self.w0 * x)

    


def make_model(num_layers, input_dim, hidden_dim, layerstyle = 'relu'):
    if layerstyle == 'sine':
        layers = [SirenLayer(input_dim, hidden_dim, is_first=True)]
        for i in range(1, num_layers - 1):
            layers.append(SirenLayer(hidden_dim, hidden_dim))
        layers.append(SirenLayer(hidden_dim, 3, is_last=True))
    else:
        layers = [nn.Linear(input_dim, hidden_dim),nn.ReLU()]
        for i in range(1, num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 3))
        layers.append(nn.Sigmoid())

    return nn.Sequential(*layers)


def train_model(network_size, learning_rate, iters, B, latent_params,
                train_data, test_data, layerstyle = 'relu', device=None):
    # import ipdb
    # ipdb.set_trace()
    model = make_model(*network_size,layerstyle=layerstyle).to(device)

    num_images = len(train_data[0])
    latent_size, latent_bound, reg_lambda = latent_params
    if latent_size > 0:
        image_latents = torch.nn.Embedding(num_images, latent_size,
                                           max_norm=latent_bound).to(device)
        torch.nn.init.normal_(
            image_latents.weight.data,
            0.0,
            1.0 / math.sqrt(latent_size),
        )
        optim = torch.optim.Adam(list(model.parameters()) + list(image_latents.parameters()),
                                 lr=learning_rate)
    else:
        optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    for i in range(iters):
        model.train()
        optim.zero_grad()

        x = input_mapping(train_data[0], B)
        if latent_size > 0:
            img_dim = x[0].shape[:2]
            latents = image_latents.weight.unsqueeze(1).unsqueeze(1)
            latents = latents.expand(-1, *img_dim, -1)
            x = torch.cat([x, latents], dim=3)

        t_o = model(x)
        t_loss = .5 * loss_fn(t_o, train_data[1])

        if reg_lambda > 0 and latent_size > 0:
            l2_size_loss = torch.sum(torch.norm(image_latents.weight, dim=1))
            reg_loss = (reg_lambda * min(1, iters / 100) * l2_size_loss) / num_images
            t_loss = t_loss + reg_loss

        t_loss.backward()
        optim.step()

        if i % 1000 == 0:
            psnr = - 10 * torch.log10(2 * t_loss).item()
            print(f"[steps:{i:4d}]: train loss: {t_loss.item():.6f} psnr: {psnr:.6f}")

            model.eval()
            with torch.no_grad():
                x = input_mapping(test_data[0], B)
                if latent_size > 0:
                    img_dim = x[0].shape[:2]
                    latents = image_latents.weight.unsqueeze(1).unsqueeze(1)
                    latents = latents.expand(-1, *img_dim, -1)
                    x = torch.cat([x, latents], dim=3)

                v_o = model(x)
                v_loss = loss_fn(v_o, test_data[1])

            v_psnr = - 10 * torch.log10(2 * v_loss).item()
            print(f"[steps:{i:4d}]: valid loss: {v_loss.item():.6f} psnr: {v_psnr:.6f}")

            wandb.log({
                'loss/train': t_loss.item(),
                'loss/valid': v_loss.item(),
                'psnr/train': psnr,
                'psnr/valid': v_psnr,
                'prediction': [wandb.Image(img) for img in v_o.data.cpu().numpy()]
            })


def get_job_id():
    from datetime import datetime
    job_id = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if 'SLURM_JOB_NAME' in os.environ and 'SLURM_JOB_ID' in os.environ:
        # running with sbatch and not srun
        if os.environ['SLURM_JOB_NAME'] != 'bash':
            job_id = os.environ['SLURM_JOB_ID']

    return job_id


if __name__ == "__main__":
    import argparse
    from glob import glob
    parser = argparse.ArgumentParser(description='Fourier Feature Networks')

    parser.add_argument("--data", default='./data')
    parser.add_argument("--exp", default=None)

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--iters', type=int, default=250)
    parser.add_argument('--max_images', type=int, default=4)

    parser.add_argument('--encoding', default='gauss')
    parser.add_argument('--scale', type=int, default=10.)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--mapping_size', type=int, default=256)
    parser.add_argument('--latent_size', type=int, default=0)
    parser.add_argument('--reg_lambda', type=float, default=0.)
    parser.add_argument('--latent_bound', type=float, default=1.0)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--activations', default='relu')
#     parser.add

    args = parser.parse_args()

    # Boilerplate stuff
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    job_id = get_job_id() if args.exp is None else args.exp
    print(f"Job Id: {job_id}")
    wandb.init(project="fourier-networks", name=job_id)
    wandb.config.update(args)

    # Download and center crop 512x512 image
    paths = glob(os.path.join(args.data, '*jpg'))
    imgs = [imageio.imread(path)[..., :3] / 255. for path in paths]
    num_images = min(len(imgs), args.max_images)
    imgs = imgs[0:num_images]

    # Create input pixel coordinates in the unit square
    img_dim = imgs[0].shape[:2]
    grid = create_grid(*img_dim[::-1])
    grid = grid.unsqueeze(0).to(device).expand(num_images, -1, -1, -1)
    imgs = torch.stack([torch.tensor(img).float() for img in imgs]).to(device)

    test_data = (grid, imgs)
    train_data = (grid[:, ::2, ::2], imgs[:, ::2, :: 2])

    # Multiple configurations
    if args.encoding == 'none':
        B = None
        input_dim = 2 + args.latent_size
    elif args.encoding == 'basic':
        B = torch.eye(2).to(device)
        input_dim = 4 + args.latent_size
    else:
        B = torch.randn((args.mapping_size, 2)).to(device) * args.scale
        input_dim = 2 * args.mapping_size + args.latent_size

    network_size = (args.num_layers, input_dim, args.hidden_dim)
    latent_params = (args.latent_size, args.latent_bound, args.reg_lambda)
    # log inputs
    wandb.log({
        'groundtruth': [wandb.Image(img.data.cpu().numpy()) for img in imgs]
    })

    # Collect outputs
    train_model(network_size, args.lr, args.iters, B, latent_params,
                train_data, test_data, layerstyle=args.activations, device=device)
