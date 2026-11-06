import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

BGR = torch.tensor([104.008, 116.669, 122.675]).view(1, 3, 1, 1)


class DeepLabV2(nn.Module):
    def __init__(self, k, weights):
        super().__init__()
        net = torchvision.models.resnet101(weights=None, replace_stride_with_dilation=[False, True, True])
        net.maxpool.ceil_mode = True
        net.layer2[0].conv1.stride, net.layer2[0].conv2.stride = (2, 2), (1, 1)
        for layer, d in ((net.layer3, 2), (net.layer4, 4)):
            layer[0].conv2.dilation, layer[0].conv2.padding = (d, d), (d, d)
        self.backbone = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3,
                                      net.layer4)
        self.aspp = nn.ModuleList([nn.Conv2d(2048, k, 3, padding=d, dilation=d) for d in (6, 12, 18, 24)])
        for c in self.aspp:
            nn.init.normal_(c.weight, 0, 0.01)
            nn.init.zeros_(c.bias)
        self.load(weights)
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)

    def load(self, path):
        sd = torch.load(path, map_location="cpu", weights_only=True)
        sd = sd.get("state_dict", sd)
        names = {"conv1": "0", "bn1": "1", "layer1": "4", "layer2": "5", "layer3": "6", "layer4": "7"}
        new = {}
        for key, v in sd.items():
            parts = key.split(".")
            if parts[0] in ("Scale", "module"):
                parts = parts[1:]
            if parts[0] in names:
                new[".".join([names[parts[0]]] + parts[1:])] = v
        missing, unexpected = self.backbone.load_state_dict(new, strict=False)
        missing = [m for m in missing if not m.endswith("num_batches_tracked")]
        if missing or unexpected:
            raise RuntimeError(f"{path}: {len(missing)} weights missing, {len(unexpected)} unexpected")

    def train(self, mode=True):
        super().train(mode)
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def prep(self, x):
        return x.flip(1) * 255 - BGR.to(x)

    def forward(self, x):
        f = self.backbone(self.prep(x))
        out = sum(c(f) for c in self.aspp)
        return F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)


class ResBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, cin), nn.Conv2d(cin, cout, 3, 1, 1)
        self.n2, self.c2 = nn.GroupNorm(8, cout), nn.Conv2d(cout, cout, 3, 1, 1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        return self.c2(F.silu(self.n2(h))) + self.skip(x)


class VAE(nn.Module):
    def __init__(self, size, latent=64):
        super().__init__()
        self.hw = (size[0] // 32, size[1] // 32)
        n = self.hw[0] * self.hw[1]
        c, e = (32, 64, 128, 256, 512, 512), (16, 32, 64, 64, 64, 64)
        enc = [nn.Conv2d(3, c[0], 3, 1, 1)]
        for i in range(5):
            enc += [ResBlock(c[i], c[i]), nn.Conv2d(c[i], c[i + 1], 3, 2, 1)]
        enc += [ResBlock(c[5], c[5]), nn.GroupNorm(8, c[5]), nn.SiLU(), nn.Conv2d(c[5], 64, 1), nn.Flatten(),
                nn.Linear(64 * n, 1024), nn.SiLU()]
        self.enc = nn.Sequential(*enc)
        self.mu, self.logvar = nn.Linear(1024, latent), nn.Linear(1024, latent)
        edge = [nn.Sequential(nn.Conv2d(2, e[0], 3, 1, 1), nn.SiLU())]
        for i in range(5):
            edge.append(nn.Sequential(nn.Conv2d(e[i], e[i + 1], 3, 2, 1), nn.GroupNorm(8, e[i + 1]), nn.SiLU()))
        self.edge = nn.ModuleList(edge)
        self.dec_fc = nn.Sequential(nn.Linear(latent, 1024), nn.SiLU(), nn.Linear(1024, 128 * n), nn.SiLU())
        self.dec_in = nn.Conv2d(128, c[5], 1)
        self.blocks = nn.ModuleList([ResBlock(c[i] + e[i], c[i]) for i in range(6)])
        self.ups = nn.ModuleList([nn.Conv2d(c[i + 1], c[i], 3, 1, 1) for i in range(5)])
        self.out = nn.Sequential(nn.GroupNorm(8, c[0]), nn.SiLU(), nn.Conv2d(c[0], 3, 3, 1, 1))

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h).clamp(-10, 10)

    def decode(self, edges, z):
        es = []
        h = edges
        for layer in self.edge:
            h = layer(h)
            es.append(h)
        h = self.dec_in(self.dec_fc(z).view(z.size(0), 128, *self.hw))
        h = self.blocks[5](torch.cat([h, es[5]], 1))
        for i in range(4, -1, -1):
            h = self.ups[i](F.interpolate(h, scale_factor=2, mode="nearest"))
            h = self.blocks[i](torch.cat([h, es[i]], 1))
        return torch.tanh(self.out(h))
