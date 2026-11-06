import glob
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

CLASSES = ["road", "sidewalk", "building", "wall", "fence", "pole", "light", "sign", "vegetation", "sky", "person",
           "rider", "car", "bus", "motorcycle", "bicycle"]
THIRTEEN = [i for i, c in enumerate(CLASSES) if c not in ("wall", "fence", "pole")]
THINGS = [CLASSES.index(c) for c in ("light", "sign", "person", "rider", "car", "bus", "motorcycle", "bicycle")]
SYNTHIA = {3: 0, 4: 1, 2: 2, 21: 3, 5: 4, 7: 5, 15: 6, 9: 7, 6: 8, 1: 9, 10: 10, 17: 11, 8: 12, 19: 13, 12: 14, 11: 15}
CITYSCAPES = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 23: 9, 24: 10, 25: 11, 26: 12, 28: 13,
              32: 14, 33: 15}


def table(mapping):
    t = np.full(65536, 255, np.uint8)
    for k, v in mapping.items():
        t[k] = v
    return t


def read_rgb(path):
    return cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def cityscapes(root, split):
    paths = sorted(glob.glob(os.path.join(root, "leftImg8bit", split, "*", "*_leftImg8bit.png")))
    if not paths:
        raise FileNotFoundError(f"no images in {root}/leftImg8bit/{split}")
    return paths


class Synthia(Dataset):
    def __init__(self, root, crop, scale=(0.75, 1.25), size=(760, 1280)):
        self.images = sorted(glob.glob(os.path.join(root, "RAND_CITYSCAPES", "RGB", "*.png")))
        if not self.images:
            raise FileNotFoundError(f"no images in {root}/RAND_CITYSCAPES/RGB")
        self.lut, self.crop, self.scale, self.size = table(SYNTHIA), crop, scale, size

    def __len__(self):
        return len(self.images)

    def label(self, i):
        p = self.images[i].replace(os.sep + "RGB" + os.sep, os.sep + os.path.join("GT", "LABELS") + os.sep)
        lab = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        lab = lab[:, :, 2] if lab.ndim == 3 else lab
        return self.lut[lab.astype(np.int64)]

    def __getitem__(self, i):
        img, lab = read_rgb(self.images[i]), self.label(i)
        h, w = self.size
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            lab = cv2.resize(lab, (w, h), interpolation=cv2.INTER_NEAREST)
        s = float(torch.empty(1).uniform_(*self.scale))
        h, w = int(h * s + 0.5), int(w * s + 0.5)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        lab = cv2.resize(lab, (w, h), interpolation=cv2.INTER_NEAREST)
        ch, cw = self.crop
        if h < ch or w < cw:
            img = cv2.copyMakeBorder(img, 0, max(0, ch - h), 0, max(0, cw - w), cv2.BORDER_CONSTANT, value=0)
            lab = cv2.copyMakeBorder(lab, 0, max(0, ch - h), 0, max(0, cw - w), cv2.BORDER_CONSTANT, value=255)
            h, w = lab.shape
        y, x = int(torch.randint(0, h - ch + 1, (1,))), int(torch.randint(0, w - cw + 1, (1,)))
        img, lab = img[y:y + ch, x:x + cw], lab[y:y + ch, x:x + cw]
        if torch.rand(1) < 0.5:
            img, lab = img[:, ::-1], lab[:, ::-1]
        return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1), torch.from_numpy(np.ascontiguousarray(lab))


class Val(Dataset):
    def __init__(self, root, size=(512, 1024)):
        self.images, self.lut, self.size = cityscapes(root, "val"), table(CITYSCAPES), size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = cv2.resize(read_rgb(self.images[i]), (self.size[1], self.size[0]), interpolation=cv2.INTER_LINEAR)
        p = self.images[i].replace("leftImg8bit", "gtFine", 1).replace("_leftImg8bit.png", "_gtFine_labelIds.png")
        lab = self.lut[cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.int64)]
        return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1), torch.from_numpy(lab)


class Images(Dataset):
    def __init__(self, paths, size):
        self.paths, self.size = paths, size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.resize(read_rgb(self.paths[i]), (self.size[1], self.size[0]), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)


class Target(Dataset):
    def __init__(self, paths, clones_path, size=(512, 1024), crop=(512, 512)):
        self.paths, self.clones_path, self.size, self.crop = paths, clones_path, size, crop
        self.clones = None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        if self.clones is None:
            self.clones = np.load(self.clones_path, mmap_mode="r")
        h, w = self.size
        img = cv2.resize(read_rgb(self.paths[i]), (w, h), interpolation=cv2.INTER_AREA)
        cl = cv2.resize(np.ascontiguousarray(np.asarray(self.clones[i]).transpose(1, 2, 0)), (w, h),
                        interpolation=cv2.INTER_LINEAR)
        ch, cw = self.crop
        y, x = int(torch.randint(0, h - ch + 1, (1,))), int(torch.randint(0, w - cw + 1, (1,)))
        img, cl = img[y:y + ch, x:x + cw], cl[y:y + ch, x:x + cw]
        if torch.rand(1) < 0.5:
            img, cl = img[:, ::-1], cl[:, ::-1]
        return (torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1),
                torch.from_numpy(np.ascontiguousarray(cl)).permute(2, 0, 1))


class Counts(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        lab = torch.from_numpy(self.ds.label(i).astype(np.int64))
        return torch.bincount(lab.flatten(), minlength=256)[:len(CLASSES)]


def sampling_weights(ds, cache, workers, temperature=0.01, min_pixels=3000):
    if os.path.exists(cache):
        counts = np.load(cache)
    else:
        counts = torch.cat([c for c in DataLoader(Counts(ds), batch_size=32, num_workers=workers)]).numpy()
        np.save(cache, counts)
    freq = counts.sum(0) / max(1, counts.sum())
    p = np.exp((1 - freq) / temperature - ((1 - freq) / temperature).max())
    p = p / p.sum()
    has = counts >= min_pixels
    n = np.maximum(has.sum(0), 1)
    w = (has * (p / n)[None]).sum(1)
    w = np.where(w > 0, w, w[w > 0].min() if (w > 0).any() else 1.0)
    return torch.from_numpy(w / w.sum())
