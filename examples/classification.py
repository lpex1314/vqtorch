# mnist VQ experiment with various settings.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

from vqtorch.nn import VectorQuant
from tqdm.auto import trange
from torch.utils.data import DataLoader
from torchvision import transforms

lr = 3e-4
train_iter = 1000
num_codes = 256
seed = 1234
feature_size = 32


class SimpleVQClassifier(nn.Module):
    def __init__(self, **vq_kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VectorQuant(feature_size=feature_size, **vq_kwargs),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 10),
        ])
        return

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, VectorQuant):
                x, vq_dict = layer(x)
            else:
                x = layer(x)
        return x, vq_dict


def train(model, train_loader, train_iterations=1000, alpha=10, ignore_commitment_loss=False):
    def iterate_dataset(data_loader):
        data_iter = iter(data_loader)
        while True:
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                x, y = next(data_iter)
            yield x.cuda(), y.cuda()

    criterion = nn.CrossEntropyLoss()

    for _ in (pbar := trange(train_iterations)):
        opt.zero_grad()
        x, y = next(iterate_dataset(train_loader))
        out, vq_out = model(x)
        sce_loss = criterion(out, y)
        cmt_loss = vq_out['loss']
        acc = (out.argmax(dim=1) == y).float().mean()
        if ignore_commitment_loss:
            sce_loss.backward()
        else:
            (sce_loss + alpha * cmt_loss).backward()

        opt.step()
        pbar.set_description(f'sce loss: {sce_loss.item():.3f} | ' + \
                             f'cmt loss: {cmt_loss.item():.3f} | ' + \
                             f'acc: {acc.item() * 100:.1f} | ' + \
                             f'active %: {vq_out["q"].unique().numel() / num_codes * 100:.3f}')
    return


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = DataLoader(datasets.MNIST(root='/root/autodl-tmp/mnist/', train=True, download=True,
                                          transform=transform), batch_size=256, shuffle=True)

print('baseline')
torch.random.manual_seed(seed)
model = SimpleVQClassifier(num_codes=num_codes).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=lr)
train(model, train_dataset, train_iterations=train_iter)

print('+ kmeans init')
torch.random.manual_seed(seed)
model = SimpleVQClassifier(num_codes=num_codes, kmeans_init=True).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=lr)
train(model, train_dataset, train_iterations=train_iter)

print('+ synchronized update')
torch.random.manual_seed(seed)
model = SimpleVQClassifier(num_codes=num_codes, kmeans_init=True, sync_nu=1.0).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=lr)
train(model, train_dataset, train_iterations=train_iter)

print('+ affine parameterization')
torch.random.manual_seed(seed)
model = SimpleVQClassifier(num_codes=num_codes, kmeans_init=True, sync_nu=1.0, affine_lr=10).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=lr)
train(model, train_dataset, train_iterations=train_iter)

print('+ soft discretization')
torch.random.manual_seed(seed)
model = SimpleVQClassifier(num_codes=num_codes, kmeans_init=True, soft_discretization=True, gamma=0.2,
                           gamma_lr=0.002).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=lr)
train(model, train_dataset, train_iterations=train_iter)

print('+ soft cluster assignment')
torch.random.manual_seed(seed)
model = SimpleVQClassifier(num_codes=num_codes, kmeans_init=True, soft_discretization_enabled=False, gamma=0.2,
                           gamma_lr=0.002, soft_cluster_assignment=True, delta=0.2, delta_lr=0.002).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=lr)
train(model, train_dataset, train_iterations=train_iter)
