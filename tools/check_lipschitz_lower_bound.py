"""Empirical lower bound on the Lipschitz constant of the sub-network.

`GloroNet.sub_lipschitz()` computes an *upper* bound, which is what the
robustness certificate rests on. This script searches for input pairs that
maximize ||f(x) - f(x')|| / ||x - x'||, giving a *lower* bound. The gap
between the two is how loose the certificate is -- so this is the tool for
checking whether a change to the Lipschitz computation is both sound (lower
bound stays below it) and tight (the gap shrinks).

Run from the repository root:

    python tools/check_lipschitz_lower_bound.py \\
        --checkpoint pretrained/cifar10_small.pth --device cpu --steps 2000
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.getcwd())

import tools  # noqa: E402


def get_args():
    parser = argparse.ArgumentParser(
        'Check the lipschitz lower bound for Globally-Robust Neural Networks')

    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config',
                        type=str,
                        default='',
                        help='override the config stored in the checkpoint')
    parser.add_argument('--device', default='auto', type=str)
    parser.add_argument('--batch_size',
                        default=32,
                        type=int,
                        help='split in half into pairs, so 32 gives 16 pairs')
    parser.add_argument('--steps', default=500, type=int)
    parser.add_argument('--lr',
                        default=1e-2,
                        type=float,
                        help='the original 1e-4 is far too small: the search '
                        'is still climbing linearly after thousands of steps '
                        'and badly understates the lower bound')
    parser.add_argument('--num_lc_iter', default=1000, type=int)
    parser.add_argument('--save_curve',
                        default='',
                        type=str,
                        help='optional path to save the search curve')

    return parser.parse_args()


def main():
    args = get_args()

    model, cfg, _ = tools.load_gloronet(args.checkpoint, args.config)
    dataset_cfg = cfg['dataset']

    device = tools.get_device(args.device)
    model = model.to(device).eval()
    model.set_num_lc_iter(args.num_lc_iter)

    subL = model.sub_lipschitz().item()
    print(f'Computed sub_lipschitz (upper bound) is {subL:.4f}')

    # `forward(return_feat=True)` returns the features before the head, which
    # is exactly the sub-network sub_lipschitz() bounds.
    input_size = dataset_cfg['input_size']
    in_channels = dataset_cfg.get('in_channels', 3)
    half = args.batch_size // 2

    # the model expects inputs in [0, 1]; it applies its own .sub(.5)
    inputs = torch.rand(args.batch_size, in_channels, input_size, input_size,
                        device=device)
    inputs.requires_grad = True

    optimizer = torch.optim.Adam([inputs], lr=args.lr)

    curve = []
    best = 0.
    for k in range(args.steps):
        optimizer.zero_grad()
        outputs = model(inputs, return_feat=True)
        diff = (outputs[:half] - outputs[half:]).pow(2).sum(1).sqrt()
        input_diff = (inputs[:half] -
                      inputs[half:]).pow(2).sum((1, 2, 3)).sqrt()
        ratio = diff / input_diff.clamp_min(1e-9)
        (-ratio.mean()).backward()
        optimizer.step()

        best = max(best, ratio.max().item())
        curve.append(best)
        if k % max(args.steps // 10, 1) == 0:
            print(f'  step {k:>6}: best lower bound {best:.4f}')

    print(f'Computed sub_lipschitz (upper bound) is {subL:.4f}')
    print(f'Lower bound of sub_lipschitz is        {best:.4f}')
    if subL > 0:
        print(f'Ratio (lower / upper) is               {best / subL:.4f}')
    if best > subL:
        print('WARNING: the empirical lower bound EXCEEDS the computed upper '
              'bound. The upper bound is unsound -- the certificate does not '
              'hold.')

    if args.save_curve:
        torch.save(curve, args.save_curve)
        print(f'Saved search curve to {args.save_curve}')


if __name__ == '__main__':
    main()
