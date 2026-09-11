"""Convert a diffusion-generated image dataset into the layout this repo
expects.

Training on diffusion-generated images alongside the real dataset is part of
the published LiResNet recipe (`use_ddpm: True` in the configs), but the
generated data is not distributed with this repository. It comes from the
robustness literature that the LiResNet papers build on -- see the README for
pointers.

Those releases ship a variety of array layouts and key names. This script
normalises whatever you downloaded into `{out_dir}/c{num_classes}_ddpm.npz`
with the keys `image` and `label`, which is what `DDPM_dataset` in
`tools/dataset/tinyimgnet.py` reads.

Example:
    python tools/dataset/make_ddpm_npz.py 1m.npz --num_classes 10
    python tools/dataset/make_ddpm_npz.py gen.npz --image_key xs --label_key ys
"""
import argparse
import os

import numpy as np


def get_args():
    parser = argparse.ArgumentParser(
        'Build c{N}_ddpm.npz from a diffusion-generated dataset')
    parser.add_argument('source',
                        type=str,
                        help='path to the downloaded .npz (or .npy for '
                        'images, paired with --label_file)')
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--out_dir', default='./data/', type=str)
    parser.add_argument('--image_key',
                        default=None,
                        type=str,
                        help='key holding the images; auto-detected if unset')
    parser.add_argument('--label_key',
                        default=None,
                        type=str,
                        help='key holding the labels; auto-detected if unset')
    parser.add_argument('--label_file',
                        default=None,
                        type=str,
                        help='separate .npy of labels, if not in `source`')
    return parser.parse_args()


IMAGE_KEYS = ('image', 'images', 'x', 'xs', 'data', 'arr_0')
LABEL_KEYS = ('label', 'labels', 'y', 'ys', 'targets', 'arr_1')


def _pick(data, explicit, candidates, what):
    if explicit is not None:
        if explicit not in data:
            raise KeyError(f'`{explicit}` not in the source. '
                           f'Available keys: {list(data.keys())}')
        return data[explicit]
    for key in candidates:
        if key in data:
            print(f'Using `{key}` as the {what} array.')
            return data[key]
    raise KeyError(f'Could not find the {what} array. Available keys: '
                   f'{list(data.keys())}. Pass --{what}_key explicitly.')


def to_uint8_nhwc(images):
    """`simple_dataset` calls `Image.fromarray`: images must be uint8 HWC."""
    if images.ndim != 4:
        raise ValueError(f'Expected 4D images, got shape {images.shape}')

    # NCHW -> NHWC. A 3-channel axis at position 1 means channels-first.
    if images.shape[1] in (1, 3) and images.shape[-1] not in (1, 3):
        print(f'Transposing NCHW {images.shape} -> NHWC')
        images = images.transpose(0, 2, 3, 1)

    if images.shape[-1] not in (1, 3):
        raise ValueError(f'Expected 1 or 3 channels last, got {images.shape}')

    if images.dtype != np.uint8:
        lo, hi = float(images.min()), float(images.max())
        print(f'Converting {images.dtype} images (range [{lo:.3f}, {hi:.3f}]) '
              'to uint8')
        if hi <= 1.0 + 1e-6 and lo >= -1e-6:
            images = images * 255.0          # [0, 1]
        elif lo < -1e-6:
            images = (images + 1.0) * 127.5  # [-1, 1]
        images = np.clip(np.rint(images), 0, 255).astype(np.uint8)

    return np.ascontiguousarray(images)


def main():
    args = get_args()

    if args.source.endswith('.npy'):
        if args.label_file is None:
            raise ValueError('A .npy source needs --label_file too.')
        images = np.load(args.source)
        labels = np.load(args.label_file)
    else:
        data = np.load(args.source)
        images = _pick(data, args.image_key, IMAGE_KEYS, 'image')
        if args.label_file is not None:
            labels = np.load(args.label_file)
        else:
            labels = _pick(data, args.label_key, LABEL_KEYS, 'label')

    images = to_uint8_nhwc(np.asarray(images))
    labels = np.asarray(labels).reshape(-1).astype(np.int64)

    if len(images) != len(labels):
        raise ValueError(f'{len(images)} images but {len(labels)} labels.')

    lo, hi = int(labels.min()), int(labels.max())
    if lo < 0 or hi >= args.num_classes:
        raise ValueError(f'Labels span [{lo}, {hi}], which does not fit '
                         f'--num_classes {args.num_classes}.')

    size = images.shape[1]
    expected = {10: 32, 100: 32, 200: 64}.get(args.num_classes)
    if expected is not None and size != expected:
        print(f'WARNING: images are {size}x{size} but num_classes='
              f'{args.num_classes} usually pairs with {expected}x{expected}. '
              'DDPM_dataset crops to the size implied by num_classes.')

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f'c{args.num_classes}_ddpm.npz')
    np.savez(out, image=images, label=labels)
    print(f'Wrote {out}: {images.shape} {images.dtype}, '
          f'{len(labels)} labels in [{lo}, {hi}]')


if __name__ == '__main__':
    main()
