#!/usr/bin/env python3
"""Run arniqa_resnet on cropped parts of images and plot quality vs ISO, Shutter, Focal.

Usage: python arniqa_test.py --dir data/captures_20260517_033855
"""
import os
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import re
import torch
from arniqa_resnet import DEVICE, ARNIQA_resnet
from sklearn.manifold import TSNE

from arniqa_mobilenet import ARNIQA_mobilenet

EXPOSURE_NS = [
	166_666,         # 1/6000 s
    500_000,         # 1/2000 s
    1_333_333,       # 1/750 s 
    4_000_000,       # 1/250 s
    11_111_111,      # 1/90 s
    33_333_333,      # 1/30 s
]
exp_ns = np.array(EXPOSURE_NS)/1_000_000_000.0  # convert to seconds

ISO_VALUES = [50, 100, 200, 400, 800]
iso_values = np.array(ISO_VALUES)

FOCUS_DIOPTERS = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0]
focal_diopters = np.array(FOCUS_DIOPTERS)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def crop_center(img, w=512, h=512):
    iw, ih = img.size
    x = max(0, (iw - w) // 2)
    y = max(0, (ih - h) // 2)
    return img.crop((x, y, x + w, y + h))


def get_meta_from_name(path):
    base = os.path.basename(path)
    m = re.search(r"exp_(\d+)ns_iso_(\d+)_focus_([0-9]+(?:\.[0-9]+)?)", base)
    if not m:
        print(f"WARNING: could not parse meta from filename {base}")
        return {'iso': None, 'shutter': None, 'focal': None}
    exposure_ns = int(m.group(1))
    iso = float(m.group(2))
    focus = float(m.group(3))
    print(f"parsed from filename: exposure_ns={exposure_ns}, iso={iso}, focus={focus}")
    # Convert exposure ns to seconds for plotting
    shutter_s = exposure_ns / 1_000_000_000.0
    return {'iso': iso, 'shutter': shutter_s, 'focal': focus}


def arniqa_resnet_score_image(model, img):
    # model can be a module; try common function names
    out, _, _ = model.forward(torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0)
    out_1d = out.reshape(out.size(0), -1).squeeze(0)
    print("model output features shape:", out.shape, "->", out_1d.shape)
    return out_1d.cpu().detach().numpy()
    # distance = torch.norm(out_1d).item()
    # return distance

def arniqa_mobilenet_score_image(model, img):
    # model can be a module; try common function names
    out, _ = model.forward(torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0)
    out_1d = out.reshape(out.size(0), -1).squeeze(0)
    print("model output features shape:", out.shape, "->", out_1d.shape)
    return out_1d.cpu().detach().numpy()
    # out_1d = out.reshape(out.size(0), -1).squeeze(0)
    # print("model output features shape:", out.shape, "->", out_1d.shape)
    # return out_1d.cpu().detach().numpy()
    # distance = torch.norm(out_1d).item()
    # return distance

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', required=True)
    p.add_argument('--crop-size', type=int, default=224)
    args = p.parse_args()

    model_resnet = ARNIQA_resnet(device=DEVICE)
    model_mobilenet = ARNIQA_mobilenet(device=DEVICE)
    files = [os.path.join(args.dir, f) for f in os.listdir(args.dir)
             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
    arniqa_resnet_scores = []
    arniqa_mobilenet_scores = []
    isos = []
    shutters = []
    focals = []

    for fp in sorted(files):
        try:
            img = Image.open(fp).convert('RGB')
            imgc = crop_center(img, args.crop_size, args.crop_size)
            meta = get_meta_from_name(fp)
            # if meta['iso'] == 200 and meta['shutter'] == 0.004:
            resnet_s = arniqa_resnet_score_image(model_resnet, imgc)
            mobilenet_s = arniqa_mobilenet_score_image(model_mobilenet, imgc)
            arniqa_mobilenet_scores.append(mobilenet_s)
            arniqa_resnet_scores.append(resnet_s)
            isos.append(meta.get('iso'))
            shutters.append(meta.get('shutter'))
            focals.append(meta.get('focal'))
            print(fp, 'score=', resnet_s, 'iso=', meta.get('iso'), 'shutter=', meta.get('shutter'), 'focal=', meta.get('focal'))
        except Exception as e:
            print('skipping', fp, 'error', e)

    arniqa_resnet_latents = np.array(arniqa_resnet_scores)
    print("all features shape:", arniqa_resnet_latents.shape)
    arniqa_mobilenet_latents = np.array(arniqa_mobilenet_scores)
    print("all features shape:", arniqa_mobilenet_latents.shape)
    tsne = TSNE(
        n_components=2,
        perplexity=120,
        learning_rate='auto',
        init='pca',
        random_state=42
    )
    latent_2d = tsne.fit_transform(arniqa_resnet_latents)
    latent_2d_mobilenet = tsne.fit_transform(arniqa_mobilenet_latents)

    focal_mapping = {   
        val:i for i,val in enumerate(FOCUS_DIOPTERS)
    }
    focal_label_ids = np.array([
        focal_mapping[v] for v in focals
    ])
    # Distinct categorical colors
    focal_cmap = plt.get_cmap('tab20', len(FOCUS_DIOPTERS))

    iso_mapping = {
        val:i for i,val in enumerate(ISO_VALUES)
    }
    iso_label_ids = np.array([
        iso_mapping[v] for v in isos
    ])
    iso_cmap = plt.get_cmap('tab10', len(ISO_VALUES))

    shutters_mapping = {
        val:i for i,val in enumerate(exp_ns)
    }
    shutter_label_ids = np.array([
        shutters_mapping[v] for v in shutters
    ])
    shutter_cmap = plt.get_cmap('tab10', len(EXPOSURE_NS))

    def plot_tsne(ax, latent, label_ids, cmap, labels, title, cbar_label):
        scatter = ax.scatter(
            latent[:, 0],
            latent[:, 1],
            c=label_ids,
            cmap=cmap,
            s=40
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_ticks(range(len(labels)))
        cbar.set_ticklabels(labels)
        cbar.set_label(cbar_label)
        ax.set_title(title)
        ax.set_xlabel("t-SNE Dim 1")
        ax.set_ylabel("t-SNE Dim 2")

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    plot_tsne(
        axes[0, 0],
        latent_2d,
        focal_label_ids,
        focal_cmap,
        FOCUS_DIOPTERS,
        "Focus (ResNet)",
        "Focus Diopters"
    )

    plot_tsne(
        axes[0, 1],
        latent_2d,
        iso_label_ids,
        iso_cmap,
        ISO_VALUES,
        "ISO (ResNet)",
        "ISO"
    )

    plot_tsne(
        axes[0, 2],
        latent_2d,
        shutter_label_ids,
        shutter_cmap,
        exp_ns,
        "Exposure (ResNet)",
        "Exposure (s)"
    )

    plot_tsne(
        axes[1, 0],
        latent_2d_mobilenet,
        focal_label_ids,
        focal_cmap,
        FOCUS_DIOPTERS,
        "Focus (MobileNet)",
        "Focus Diopters"
    )

    plot_tsne(
        axes[1, 1],
        latent_2d_mobilenet,
        iso_label_ids,
        iso_cmap,
        ISO_VALUES,
        "ISO (MobileNet)",
        "ISO"
    )

    plot_tsne(
        axes[1, 2],
        latent_2d_mobilenet,
        shutter_label_ids,
        shutter_cmap,
        exp_ns,
        "Exposure (MobileNet)",
        "Exposure (s)"
    )

    fig.tight_layout()
    plt.show()

    
    # plotting
    # def safe_plot(x, y, xlabel, fname):
    #     xarr = np.array([v if v is not None else np.nan for v in x], dtype=float)
    #     yarr = np.array(y, dtype=float)
    #     mask = ~np.isnan(xarr) & ~np.isnan(yarr)
    #     plt.figure()
    #     plt.scatter(xarr[mask], yarr[mask])
    #     plt.xlabel(xlabel)
    #     plt.ylabel('arniqa score')
    #     plt.grid(True)
    #     plt.savefig(fname)
    #     plt.close()

    # safe_plot(isos, arniqa_resnet_scores, 'ISO', 'score_vs_iso.png')
    # safe_plot(shutters, arniqa_resnet_scores, 'Shutter', 'score_vs_shutter.png')
    # safe_plot(focals, arniqa_resnet_scores, 'FocalLength', 'score_vs_focal.png')

    # print('plots saved: score_vs_iso.png, score_vs_shutter.png, score_vs_focal.png')


if __name__ == '__main__':
    main()
