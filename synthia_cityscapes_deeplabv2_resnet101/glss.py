import argparse
import copy
import os
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from data import CLASSES, THINGS, THIRTEEN, Images, Synthia, Target, Val, cityscapes, sampling_weights
from model import VAE, DeepLabV2

warnings.filterwarnings("ignore")
K = len(CLASSES)
SOURCE_ITERS, GEN_ITERS, SEARCH_ITERS, ITERS = 40000, 20000, 150, 80000
GEN_SIZE, SEARCH_BATCH, SEED = (384, 768), 32, 0
YIQ = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]])
T0 = time.time()


def say(msg):
    print(f"[{(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)


def progress(step, done, total, start, t):
    sec = (time.time() - t) / max(done - start, 1) * (total - done)
    say(f"{step}/4 {done}/{total} [{sec / 3600:.1f} h left]" if sec >= 3600 else
        f"{step}/4 {done}/{total} [{sec / 60:.0f} min left]")


def amp():
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def save(obj, path):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def forever(ds, batch, workers, seed, weights=None):
    g = torch.Generator().manual_seed(seed)
    while True:
        if weights is None:
            loader = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers, drop_last=True, generator=g,
                                pin_memory=True)
        else:
            sampler = WeightedRandomSampler(weights, num_samples=len(ds), replacement=True, generator=g)
            loader = DataLoader(ds, batch_size=batch, sampler=sampler, num_workers=workers, drop_last=True,
                                pin_memory=True)
        for b in loader:
            yield b


def uniform(b, lo, hi, dev):
    return torch.empty(b, 1, 1, 1, device=dev).uniform_(lo, hi)


def blur(x, sigma):
    b, c, h, w = x.shape
    r = torch.arange(-4, 5, device=x.device, dtype=x.dtype)
    k = torch.exp(-r[None] ** 2 / (2 * sigma[:, None] ** 2))
    k = (k / k.sum(1, keepdim=True)).repeat_interleave(c, 0)
    y = F.pad(x.reshape(1, b * c, h, w), (4, 4, 4, 4), mode="reflect")
    y = F.conv2d(y, k[:, None, :, None], groups=b * c)
    y = F.conv2d(y, k[:, None, None, :], groups=b * c)
    return y.reshape(b, c, h, w)


def camera(x):
    b, dev = x.size(0), x.device
    x = x * torch.exp(torch.randn(b, 3, 1, 1, device=dev) * 0.08)
    x = x * uniform(b, 0.75, 1.25, dev)
    m = x.mean((1, 2, 3), keepdim=True)
    x = (x - m) * uniform(b, 0.7, 1.3, dev) + m
    g = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    x = g + (x - g) * uniform(b, 0.4, 1.6, dev)
    a = torch.empty(b, device=dev).uniform_(-0.35, 0.35)
    rot = torch.zeros(b, 3, 3, device=dev)
    rot[:, 0, 0] = 1
    rot[:, 1, 1], rot[:, 1, 2], rot[:, 2, 1], rot[:, 2, 2] = a.cos(), -a.sin(), a.sin(), a.cos()
    yiq = YIQ.to(dev)
    x = torch.einsum("bij,bjhw->bihw", torch.linalg.inv(yiq) @ rot @ yiq, x)
    x = x.clamp(0, 1) ** uniform(b, 0.7, 1.4, dev)
    x = x + uniform(b, -1.0, 1.5, dev) * (x - blur(x, torch.empty(b, device=dev).uniform_(0.5, 2.0)))
    x = x + torch.randn_like(x) * uniform(b, 0.0, 0.03, dev)
    g = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]).expand_as(x)
    x = torch.where(torch.rand(b, 1, 1, 1, device=dev) < 0.05, g, x)
    return x.clamp(0, 1)


def gray(x):
    return (0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3] + 1) * 0.5


def edges(x):
    ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=x.device, dtype=x.dtype)
    g = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
    e = F.conv2d(F.pad(g, (1, 1, 1, 1), mode="reflect"), torch.stack([ky, ky.t()]).unsqueeze(1))
    return e / (e.flatten(1).std(1)[:, None, None, None] + 1e-3)


def ssim(a, b):
    g = torch.arange(11, device=a.device, dtype=a.dtype) - 5
    g = torch.exp(-g ** 2 / 4.5)
    g = g / g.sum()
    w = (g[:, None] * g[None, :])[None, None]
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ma, mb = F.conv2d(a, w), F.conv2d(b, w)
    saa = F.conv2d(a * a, w) - ma * ma
    sbb = F.conv2d(b * b, w) - mb * mb
    sab = F.conv2d(a * b, w) - ma * mb
    s = (2 * ma * mb + c1) / (ma * ma + mb * mb + c1) * (2 * sab + c2) / (saa + sbb + c2)
    return s.mean((1, 2, 3))


def features(model, x):
    b = model.backbone
    h = b[4](b[3](b[2](b[1](b[0](model.prep(x))))))
    f2 = b[5](h)
    f3 = b[6](f2)
    f4 = b[7](f3)
    return [f2, f3, f4], sum(c(f4) for c in model.aspp)


def perceptual(fa, fb):
    return sum(((F.normalize(a.float(), dim=1) - F.normalize(b.float(), dim=1)) ** 2).sum(1).mean()
               for a, b in zip(fa, fb))


def source_model(args, dev):
    path = os.path.join(args.out, "source.pt")
    torch.manual_seed(SEED)
    model = DeepLabV2(K, args.init).to(dev).to(memory_format=torch.channels_last)
    backbone = [p for p in model.backbone.parameters() if p.requires_grad]
    opt = torch.optim.SGD([{"params": backbone, "lr0": 2.5e-4}, {"params": model.aspp.parameters(), "lr0": 2.5e-3}],
                          lr=2.5e-4, momentum=0.9, weight_decay=5e-4)
    start = 0
    if os.path.exists(path):
        c = torch.load(path, map_location=dev, weights_only=True)
        model.load_state_dict(c["model"])
        opt.load_state_dict(c["opt"])
        start = c["iter"]
    if start >= SOURCE_ITERS:
        return model.eval()
    ds = Synthia(args.synthia, (512, 1024))
    batches = forever(ds, 2, args.workers, SEED * 1000 + start, sampling_weights(ds, args.counts, args.workers))
    model.train()
    t = time.time()
    for it in range(start, SOURCE_ITERS):
        for g in opt.param_groups:
            g["lr"] = g["lr0"] * (1 - it / SOURCE_ITERS) ** 0.9
        x, y = next(batches)
        x, y = x.to(dev, non_blocking=True).float().div_(255), y.to(dev, non_blocking=True).long()
        x = camera(x)
        with amp():
            logits = model(x.contiguous(memory_format=torch.channels_last))
        loss = F.cross_entropy(logits.float(), y, ignore_index=255)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (it + 1) % 500 == 0:
            progress(1, it + 1, SOURCE_ITERS, start, t)
        if (it + 1) % 2000 == 0 or it + 1 == SOURCE_ITERS:
            save({"model": model.state_dict(), "opt": opt.state_dict(), "iter": it + 1}, path)
    return model.eval()


def generator(seg, args, dev):
    path = os.path.join(args.out, "generator.pt")
    torch.manual_seed(SEED)
    gen = VAE(GEN_SIZE).to(dev)
    opt = torch.optim.Adam(gen.parameters(), lr=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, GEN_ITERS)
    start = 0
    if os.path.exists(path):
        c = torch.load(path, map_location=dev, weights_only=True)
        gen.load_state_dict(c["gen"])
        opt.load_state_dict(c["opt"])
        sched.load_state_dict(c["sched"])
        start = c["iter"]
    if start >= GEN_ITERS:
        return gen.eval()
    h, w = GEN_SIZE
    ds = Synthia(args.synthia, (h, w), scale=(1.0, 1.0), size=(int(round(w * 760 / 1280)), w))
    batches = forever(ds, 4, args.workers, SEED * 1000 + 7 + start)
    seg.eval()
    seg.requires_grad_(False)
    gen.train()
    t = time.time()
    for it in range(start, GEN_ITERS):
        x = next(batches)[0].to(dev, non_blocking=True).float().div_(127.5).sub_(1)
        with amp():
            mu, logvar = gen.encode(x)
            mu, logvar = mu.float(), logvar.float()
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
            out = gen.decode(edges(x), z).float()
        kl = torch.clamp(-0.5 * (1 + logvar - mu ** 2 - logvar.exp()).mean(0), min=0.5).mean()
        with amp():
            with torch.no_grad():
                fr, lr = features(seg, (x + 1) * 0.5)
            fo, lo = features(seg, (out + 1) * 0.5)
        sem = F.kl_div(F.log_softmax(lo.float(), 1), F.softmax(lr.float(), 1), reduction="none").sum(1).mean()
        loss = F.mse_loss(out, x) + perceptual(fo, fr) + sem + 0.05 * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), 5.0)
        opt.step()
        sched.step()
        if (it + 1) % 500 == 0:
            progress(2, it + 1, GEN_ITERS, start, t)
        if (it + 1) % 2000 == 0 or it + 1 == GEN_ITERS:
            save({"gen": gen.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "iter": it + 1}, path)
    return gen.eval()


def clones(gen, paths, path, args, dev):
    gen.requires_grad_(False)
    h, w = GEN_SIZE
    out = np.lib.format.open_memmap(path + ".part", mode="w+", dtype=np.uint8, shape=(len(paths), 3, h, w))
    noise = torch.Generator(device=dev).manual_seed(SEED)
    done, t = 0, time.time()
    for imgs in DataLoader(Images(paths, (h, w)), batch_size=SEARCH_BATCH, num_workers=args.workers):
        x = imgs.to(dev).float().div_(127.5).sub_(1)
        e, target = edges(x), gray(x)
        z = torch.randn(x.size(0), gen.mu.out_features, generator=noise, device=dev).requires_grad_(True)
        opt = torch.optim.Adam([z], lr=0.1, betas=(0.9, 0.99))
        best = torch.full((x.size(0),), float("inf"), device=dev)
        best_z = z.detach().clone()
        for it in range(SEARCH_ITERS + 1):
            with amp():
                y = gen.decode(e, z)
            loss = 1 - ssim(gray(y.float()), target)
            with torch.no_grad():
                better = loss < best
                best = torch.where(better, loss, best)
                best_z[better] = z.detach()[better]
            if it == SEARCH_ITERS:
                break
            z.grad, = torch.autograd.grad(loss.sum(), z)
            opt.step()
        with torch.no_grad(), amp():
            c = gen.decode(e, best_z).float()
        out[done:done + len(c)] = ((c + 1) * 127.5).round().clamp(0, 255).byte().cpu().numpy()
        done += len(c)
        if done // 100 > (done - len(c)) // 100 or done == len(paths):
            progress(3, done, len(paths), 0, t)
    out.flush()
    del out
    os.replace(path + ".part", path)


def final_model(paths, clones_path, args, dev):
    path = os.path.join(args.out, "model.pt")
    torch.manual_seed(SEED)
    student = DeepLabV2(K, args.init).to(dev).to(memory_format=torch.channels_last)
    teacher = copy.deepcopy(student)
    teacher.requires_grad_(False)
    ref = DeepLabV2(K, args.init).to(dev).to(memory_format=torch.channels_last).eval()
    ref.requires_grad_(False)
    backbone = [p for p in student.backbone.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": backbone, "lr0": 6e-5}, {"params": student.aspp.parameters(), "lr0": 6e-4}],
                            lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01)
    start = 0
    if os.path.exists(path):
        c = torch.load(path, map_location=dev, weights_only=True)
        student.load_state_dict(c["student"])
        teacher.load_state_dict(c["teacher"])
        opt.load_state_dict(c["opt"])
        start = c["iter"]
    if start >= ITERS:
        return student.eval()
    torch.manual_seed(SEED + start)
    ds = Synthia(args.synthia, (512, 512))
    src = forever(ds, 2, args.workers // 2, SEED * 1000 + 19 + start, sampling_weights(ds, args.counts, args.workers))
    tgt = forever(Target(paths, clones_path), 2, args.workers // 2, SEED * 1000 + 23 + start)
    things = torch.tensor(THINGS, device=dev)
    student.train()
    teacher.eval()
    t = time.time()
    for it in range(start, ITERS):
        for g in opt.param_groups:
            g["lr"] = g["lr0"] * min(1.0, (it + 1) / 1500) * (1 - it / ITERS)
        (xs, ys), (xt, ct) = next(src), next(tgt)
        xs = xs.to(dev, non_blocking=True).float().div_(255)
        ys = ys.to(dev, non_blocking=True).long()
        xt = xt.to(dev, non_blocking=True).float().div_(255)
        ct = ct.to(dev, non_blocking=True).float().div_(255)
        with torch.no_grad(), amp():
            pt = 0.5 * (teacher(xt.contiguous(memory_format=torch.channels_last)).float().softmax(1)
                        + teacher(ct.contiguous(memory_format=torch.channels_last)).float().softmax(1))
        conf, pl = pt.float().max(1)
        wt = (conf > 0.968).float().mean((1, 2), keepdim=True).expand_as(conf)
        mask = torch.zeros_like(ys, dtype=torch.bool)
        for b in range(ys.size(0)):
            present = ys[b].unique()
            present = present[present != 255]
            if len(present):
                chosen = present[torch.randperm(len(present), device=dev)[:(len(present) + 1) // 2]]
                mask[b] = torch.isin(ys[b], chosen)
        m = mask[:, None].float()
        xm = camera(m * xs + (1 - m) * xt)
        ym = torch.where(mask, ys, pl)
        wm = torch.where(mask, torch.ones_like(wt), wt)
        xa = camera(xs)
        with amp():
            f = student.backbone(student.prep(xa.contiguous(memory_format=torch.channels_last)))
            ls = F.interpolate(sum(c(f) for c in student.aspp), size=xa.shape[-2:], mode="bilinear",
                               align_corners=False)
        loss = F.cross_entropy(ls.float(), ys, ignore_index=255)
        with torch.no_grad(), amp():
            fr = ref.backbone(ref.prep(xa.contiguous(memory_format=torch.channels_last)))
        thing = F.adaptive_max_pool2d(torch.isin(ys, things)[:, None].float(), f.shape[-2:])[:, 0] > 0
        if thing.any():
            loss = loss + 0.005 * (f.float() - fr.float()).norm(dim=1)[thing].mean()
        with amp():
            lm = student(xm.contiguous(memory_format=torch.channels_last))
        ce = F.cross_entropy(lm.float(), ym, ignore_index=255, reduction="none")
        loss = loss + (ce * wm).sum() / (ym != 255).sum().clamp(min=1)
        keep = torch.rand(xt.size(0), 1, -(-xt.size(2) // 64), -(-xt.size(3) // 64), device=dev) > 0.7
        keep = keep.float().repeat_interleave(64, 2).repeat_interleave(64, 3)[:, :, :xt.size(2), :xt.size(3)]
        with amp():
            lk = student((camera(xt) * keep).contiguous(memory_format=torch.channels_last))
        loss = loss + (F.cross_entropy(lk.float(), pl, reduction="none") * wt).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            alpha = min(1 - 1 / (it + 1), 0.99)
            for p_t, p_s in zip(teacher.parameters(), student.parameters()):
                p_t.mul_(alpha).add_(p_s.detach(), alpha=1 - alpha)
        if (it + 1) % 500 == 0:
            progress(4, it + 1, ITERS, start, t)
        if (it + 1) % 2000 == 0 or it + 1 == ITERS:
            save({"student": student.state_dict(), "teacher": teacher.state_dict(), "opt": opt.state_dict(),
                  "iter": it + 1}, path)
    return student.eval()


@torch.no_grad()
def evaluate(model, val, args, dev):
    model.eval()
    conf = torch.zeros(K * K, dtype=torch.long, device=dev)
    for x, y in DataLoader(val, batch_size=1, num_workers=args.workers):
        x, y = x.to(dev).float().div_(255), y.to(dev).long()
        with amp():
            p = model(x.contiguous(memory_format=torch.channels_last)).float().softmax(1)
        pred = F.interpolate(p, size=y.shape[-2:], mode="bilinear", align_corners=False).argmax(1)
        valid = y != 255
        conf += torch.bincount(y[valid] * K + pred[valid], minlength=K * K)
    c = conf.view(K, K).double()
    return (c.diag() / (c.sum(0) + c.sum(1) - c.diag()).clamp(min=1) * 100).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".")
    ap.add_argument("--init", default="DeepLab_resnet_pretrained_init-f81d91e8.pth")
    ap.add_argument("--out", default="out")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    args.synthia, args.cityscapes = os.path.join(args.data, "synthia"), os.path.join(args.data, "cityscapes")
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    args.counts = os.path.join(args.out, "class_pixels.npy")
    paths, val = cityscapes(args.cityscapes, "train"), Val(args.cityscapes)
    say(f"loaded SYNTHIA {len(Synthia(args.synthia, (512, 1024)))} images, Cityscapes train {len(paths)} images, "
        f"val {len(val)} images")
    clones_path = os.path.join(args.out, "clones.npy")
    if not os.path.exists(clones_path):
        seg = source_model(args, dev)
        gen = generator(seg, args, dev)
        del seg
        clones(gen, paths, clones_path, args, dev)
        del gen
        torch.cuda.empty_cache()
    iou = evaluate(final_model(paths, clones_path, args, dev), val, args, dev)
    lines = ["SYNTHIA -> Cityscapes, DeepLabv2 ResNet-101", f"{'Class':<14}{'IoU':>8}"]
    lines += [f"{c:<14}{v:>8.2f}" for c, v in zip(CLASSES, iou)]
    lines += [f"{'mIoU (16)':<14}{sum(iou) / len(iou):>8.2f}",
              f"{'mIoU (13)':<14}{sum(iou[i] for i in THIRTEEN) / len(THIRTEEN):>8.2f}"]
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
