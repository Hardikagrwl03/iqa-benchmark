#!/usr/bin/env python3
"""Run arniqa_resnet on cropped parts of images and plot quality vs ISO, Shutter, Focal.

Usage: python arniqa_test.py --dir data/captures_20260517_033855
"""
import os
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sympy import re
import torch
from arniqa_resnet import DEVICE, ARNIQA_resnet

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def crop_center(img, w=512, h=512):
    iw, ih = img.size
    x = max(0, (iw - w) // 2)
    y = max(0, (ih - h) // 2)
    return img.crop((x, y, x + w, y + h))


def get_meta_from_name(path):
    base = os.path.basename(path)
    m = re.search(r"exp_(\d+)ns_iso_(\d+)_focus_([0-9.]+)", base)
    if not m:
        return {'iso': None, 'shutter': None, 'focal': None}
    exposure_ns = int(m.group(1))
    iso = float(m.group(2))
    focus = float(m.group(3))
    # Convert exposure ns to seconds for plotting
    shutter_s = exposure_ns / 1_000_000_000.0
    return {'iso': iso, 'shutter': shutter_s, 'focal': focus}


def score_image(model, img):
    # model can be a module; try common function names
    out, _, _ = model.forward(torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0)
    distance = torch.norm(out, dim=1).item()
    return distance


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', required=True)
    p.add_argument('--crop-size', type=int, default=224)
    args = p.parse_args()

    model = ARNIQA_resnet(device='cuda' if torch.cuda.is_available() else 'cpu')
    files = [os.path.join(args.dir, f) for f in os.listdir(args.dir)
             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
    arniqa_resnet_scores = []
    isos = []
    shutters = []
    focals = []

    for fp in sorted(files):
        try:
            img = Image.open(fp).convert('RGB')
            imgc = crop_center(img, args.crop_size, args.crop_size)
            meta = get_meta_from_name(fp)
            try:
                s = score_image(model, imgc)
            except Exception:
                # if scoring fails, store nan
                s = float('nan')
            arniqa_resnet_scores.append(s)
            isos.append(meta.get('iso'))
            shutters.append(meta.get('shutter'))
            focals.append(meta.get('focal'))
            print(fp, 'score=', s, 'iso=', meta.get('iso'), 'shutter=', meta.get('shutter'), 'focal=', meta.get('focal'))
        except Exception as e:
            print('skipping', fp, 'error', e)

    # plotting
    def safe_plot(x, y, xlabel, fname):
        xarr = np.array([v if v is not None else np.nan for v in x], dtype=float)
        yarr = np.array(y, dtype=float)
        mask = ~np.isnan(xarr) & ~np.isnan(yarr)
        plt.figure()
        plt.scatter(xarr[mask], yarr[mask])
        plt.xlabel(xlabel)
        plt.ylabel('arniqa score')
        plt.grid(True)
        plt.savefig(fname)
        plt.close()

    safe_plot(isos, arniqa_resnet_scores, 'ISO', 'score_vs_iso.png')
    safe_plot(shutters, arniqa_resnet_scores, 'Shutter', 'score_vs_shutter.png')
    safe_plot(focals, arniqa_resnet_scores, 'FocalLength', 'score_vs_focal.png')

    print('plots saved: score_vs_iso.png, score_vs_shutter.png, score_vs_focal.png')


if __name__ == '__main__':
    main()
