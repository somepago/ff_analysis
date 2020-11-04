import os
import numpy as np
from tqdm import tqdm
import imageio

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class Edges(nn.Module):
    def __init__(self, nc=3):
        super(Edges, self).__init__()
        edge_weights = torch.Tensor([[0., 1., 0.],
                                     [1., -4, 1.],
                                     [0., 1., 0.]])
        self.edge_kernel = nn.Conv2d(nc, nc, 3, 1, 1, groups=nc, bias=False)
        self.edge_kernel.weight.data = edge_weights.reshape(1, 1, 3, 3).repeat(nc, 1, 1, 1)

        if torch.cuda.is_available():
            self.edge_kernel = self.edge_kernel.to('cuda')
            
    def forward(self, x):

        # Get edges
        edges = torch.abs(self.edge_kernel(x))
#         edges = edges/torch.median(edges)
        return edges.detach()



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


def make_model(num_layers, input_dim, hidden_dim, layerstyle = 'regular'):
    if layerstyle == 'Siren':
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


def train_model(network_size, learning_rate, iters, B, train_data, test_data, device=None):
    num_layers, input_dim, hidden_dim = network_size
    input_dim = 2 if B is None else B.shape[0] * 2
    model = make_model(num_layers, input_dim, hidden_dim).to(device)
    
    optim = torch.optim.Adam(list(model.parameters()) + [B], lr=learning_rate)
    loss_fn = torch.nn.MSELoss()
#     loss_fn = torch.nn.L1Loss()

    train_psnrs = []
    test_psnrs = []
    pred_imgs = []
    xs = []
    
#     edge_ = Edges()
#     im_torch = train_data[1].permute(0,3,1,2)
#     edge_weights = edge_(im_torch).permute(0,2,3,1)
#     edge_weights = (edge_weights/torch.sum(edge_weights))*256*256
#     edge_weights = torch.clamp(edge_weights, min=0.1)
#     edge_weights = edge_weights.detach()
    
    for i in tqdm(range(iters), desc='train iter', leave=False):
        model.train()
        optim.zero_grad()

        t_o = model(input_mapping(train_data[0], B))
        t_loss = .5 * loss_fn(t_o, train_data[1])
        t_loss.backward(retain_graph=True)
        optim.step()

        if i % 25 == 0:
            train_psnrs.append(- 10 * torch.log10(2 * t_loss).item())
            model.eval()
            with torch.no_grad():
                v_o = model(input_mapping(test_data[0], B))
                v_loss = loss_fn(v_o, test_data[1])
                v_psnrs = - 10 * torch.log10(2 * v_loss).item()
                test_psnrs.append(v_psnrs)
                xs.append(i)
                pred_imgs.append(v_o)

    return {
        'state': model.state_dict(),
        'train_psnrs': train_psnrs,
        'test_psnrs': test_psnrs,
        'pred_imgs': torch.stack(pred_imgs).data.cpu().numpy(),
        'xs': xs,
    }


if __name__ == "__main__":
    import argparse
    from glob import glob
    parser = argparse.ArgumentParser(description='Fourier Feature Networks - Single image')

    parser.add_argument("--data", default='./data')
    parser.add_argument("--exp", default=None)

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--iters', type=int, default=250)

    parser.add_argument('--encoding', default='gauss')
    parser.add_argument('--scale', type=int, default=10.)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--mapping_size', type=int, default=256)
    parser.add_argument('--latent_size', type=int, default=0)
    parser.add_argument('--reg_lambda', type=float, default=0.)
    parser.add_argument('--latent_bound', type=float, default=1.0)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--train_data_scaling', type=int, default=2)
    

    args = parser.parse_args()

    # Download and center crop 512x512 image
    image_url = args.data
    img = imageio.imread(image_url)[..., :3] / 255.
    c = [img.shape[0]//2, img.shape[1]//2]
    r = 256
    img = img[c[0]-r:c[0]+r, c[1]-r:c[1]+r]

    # torch device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Create input pixel coordinates in the unit square
    img_dim = img.shape[:2]
    grid = create_grid(*img_dim[::-1])
    img = torch.tensor(img).float()

    grid = grid.unsqueeze(0).to(device)
    img = img.unsqueeze(0).to(device)

    test_data = (grid, img)
    #new - Added the scaling to traindata to check if it still works
    sf = args.train_data_scaling
    train_data = (grid[:, ::sf, ::sf], img[:, ::sf, :: sf])

    network_size = (args.num_layers, 512, 256)
    learning_rate = args.lr
    iters = args.iters
    mapping_size = args.mapping_size

    # Multiple configurations
    B_dict = {
#         'none': None,
        'basic': torch.eye(2).to(device)
    }
    B_gauss = torch.randn((mapping_size, 2)).to(device)
    for scale in [1.,10.,100.]:
        B_dict[f'gauss_{scale}'] = nn.Parameter(B_gauss * scale, requires_grad=True)

    # Collect outputs
    outputs = {}
    for k in tqdm(B_dict):
        outputs[k] = train_model(network_size, learning_rate, iters, B_dict[k],
                                 train_data=train_data, test_data=test_data, device=device)

    # Output images
    plt.figure(figsize=(24, 4))
    N = len(outputs)
    for i, k in enumerate(outputs):
        plt.subplot(1, N + 1, i + 1)
        plt.imshow(outputs[k]['pred_imgs'][-1][0])
        plt.title(f"{k} ({outputs[k]['test_psnrs'][-1]:.3f})")
        plt.axis('off')
    plt.subplot(1, N + 1, N + 1)
    plt.imshow(img[0].data.cpu().numpy())
    plt.title('GT')
    plt.axis('off')
    plt.savefig("output_img.png", bbox_inches='tight')

    # Train/test plots
    plt.figure(figsize=(16, 6))

    plt.subplot(121)
    for i, k in enumerate(outputs):
        plt.plot(outputs[k]['xs'], outputs[k]['train_psnrs'], label=k)
    plt.title('Train error')
    plt.ylabel('PSNR')
    plt.xlabel('Training iter')
    plt.legend()

    plt.subplot(122)
    for i, k in enumerate(outputs):
        plt.plot(outputs[k]['xs'], outputs[k]['test_psnrs'], label=k)
    plt.title('Test error')
    plt.ylabel('PSNR')
    plt.xlabel('Training iter')
    plt.legend()

    plt.savefig("output_loss.png", bbox_inches='tight')

#     # Animation
#     all_preds = np.concatenate([outputs[n]['pred_imgs'][:, 0] for n in outputs], axis=-2)
#     data8 = (255 * np.clip(all_preds, 0, 1)).astype(np.uint8)
#     f = os.path.join('output_convergence.mp4')
#     imageio.mimwrite(f, data8, fps=20)

#     # Display video inline
#     from IPython.display import HTML
#     from base64 import b64encode

#     mp4 = open(f, 'rb').read()
#     data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
#     HTML(f'''
#     <video width=1000 controls autoplay loop>
#           <source src="{data_url}" type="video/mp4">
#     </video>
#     <table width="1000" cellspacing="0" cellpadding="0">
#       <tr>{''.join(N * [f'<td width="{1000 // len(outputs)}"></td>'])}</tr>
#       <tr>{''.join(N * ['<td style="text-align:center">{}</td>'])}</tr>
#     </table>
#     '''.format(*list(outputs.keys())))
