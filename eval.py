"""Evaluate a trained GloroNet: Lipschitz bound, clean accuracy and VRA.

None of this requires training, and the Lipschitz parts do not even require
data -- `sub_lipschitz()` is a pure function of the model weights. That makes
this the fast way to iterate on how the Lipschitz constant is computed:

    # just the bound, sub-second on CPU, no dataset touched
    python eval.py --checkpoint pretrained/cifar10_small.pth --skip_data

    # how the power method converges
    python eval.py --checkpoint pretrained/cifar10_small.pth \
        --lc_sweep 1,5,10,50,100,500

    # accuracy and VRA on a subset, seconds on CPU
    python eval.py --checkpoint pretrained/cifar10_small.pth --num_test 1000
"""
import argparse
import time

import torch

import tools


def get_args():
    parser = argparse.ArgumentParser(
        'Evaluate Globally-Robust Neural Networks')

    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config',
                        type=str,
                        default='',
                        help='override the config stored in the checkpoint')
    parser.add_argument('--device', default='auto', type=str)
    parser.add_argument('--data_root', default='./data/', type=str)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='0 is fastest here: evaluation is short enough '
                        'that worker startup dominates')
    parser.add_argument('--num_test',
                        default=0,
                        type=int,
                        help='evaluate only the first N test images '
                        '(0 = all). The test loader is not shuffled, so this '
                        'is a deterministic prefix of the test set.')
    parser.add_argument('--num_lc_iter',
                        default=500,
                        type=int,
                        help='power iterations for the Lipschitz bound; '
                        'training uses 500 at validation time')
    parser.add_argument('--eps',
                        default='',
                        type=str,
                        help='comma-separated radii for VRA; defaults to the '
                        "config's eps and 2x/3x it")
    parser.add_argument('--lc_sweep',
                        default='',
                        type=str,
                        help='comma-separated iteration counts; report the '
                        'Lipschitz bound at each and exit')
    parser.add_argument('--skip_data',
                        action='store_true',
                        help='report only the Lipschitz bound, no dataset')

    return parser.parse_args()


COMPONENTS = ('stem', 'conv', 'neck', 'linear')


def lipschitz_breakdown(net):
    """Per-component factors that `GloroNet.sub_lipschitz` combines.

    As shipped, `sub_lipschitz` is the product of these four; reporting them
    separately shows which part of the network a change to the bound moved.

    Returns `(parts, warnings)`. A component that has been removed, renamed,
    or left without a `lipschitz` method is reported as a warning rather than
    dropped -- a silently missing factor would make the breakdown look
    complete when it is not.
    """
    parts, warnings = {}, []
    for name in COMPONENTS:
        module = getattr(net, name, None)
        if module is None:
            warnings.append(f'component `{name}` does not exist on the model')
            continue
        if not hasattr(module, 'lipschitz'):
            warnings.append(
                f'component `{name}` ({type(module).__name__}) has no '
                '`lipschitz()` method')
            continue
        value = module.lipschitz()
        # some components return a float, others a tensor still carrying grad
        parts[name] = (float(value.detach())
                       if torch.is_tensor(value) else float(value))
    return parts, warnings


@torch.no_grad()
def evaluate(net, loader, device, sub_lipschitz, eps_list, num_test=0):
    """Clean accuracy and VRA at each radius, from a single forward pass.

    The certified-prediction margin is the same computation as
    `models.trades_loss` (models/margin_layer.py:31-37), lifted out so that
    several radii share one forward pass instead of one each.
    """
    correct = total = 0
    correct_vra = [0] * len(eps_list)

    for inputs, targets in loader:
        if num_test and total >= num_test:
            break
        inputs = inputs.to(device)
        targets = targets.to(device)
        if num_test:
            keep = min(inputs.shape[0], num_test - total)
            inputs, targets = inputs[:keep], targets[:keep]

        head = net.head.get_weight()
        y = net(inputs)
        pred = y.argmax(1)

        # distance from the predicted class's head vector to every other one
        head_j = head[pred].unsqueeze(1)
        head_ji = (head_j - head.unsqueeze(0)).norm(dim=-1)

        correct += pred.eq(targets).sum().item()
        total += targets.size(0)

        for i, eps in enumerate(eps_list):
            y_ = y + sub_lipschitz * eps * head_ji
            y_ = y_.scatter(1, pred.view(-1, 1), -10.**10)
            y_ = y_.max(1)[0].reshape(-1, 1)
            y_ = torch.cat([y, y_], dim=1)
            correct_vra[i] += y_.argmax(1).eq(targets).sum().item()

    return correct / total, [c / total for c in correct_vra], total


def main():
    args = get_args()

    net, cfg, ckpt = tools.load_gloronet(args.checkpoint, args.config)
    dataset_cfg, gloro_cfg = cfg['dataset'], cfg['gloro']

    device = tools.get_device(args.device)
    net = net.to(device).eval()

    print(f'Checkpoint : {args.checkpoint}')
    print(f'Device     : {device}')
    if 'training_logs' in ckpt and ckpt['training_logs']:
        print(f"Trained    : {ckpt['training_logs'][-1]}")
    print()

    # -- the Lipschitz bound: no data required --------------------------
    if args.lc_sweep:
        iters = [int(n) for n in args.lc_sweep.split(',')]
        print('Lipschitz bound vs power iterations:')
        print(f'{"iters":>8}  {"sub_lipschitz":>14}  {"seconds":>8}')
        for n in iters:
            net.set_num_lc_iter(n)
            t = time.time()
            lc = net.sub_lipschitz().item()
            print(f'{n:>8}  {lc:>14.4f}  {time.time() - t:>8.3f}')
        return

    net.set_num_lc_iter(args.num_lc_iter)
    t = time.time()
    parts, warnings = lipschitz_breakdown(net)
    sub_lipschitz = net.sub_lipschitz().item()
    lc_time = time.time() - t

    print(f'Lipschitz bound ({args.num_lc_iter} power iterations, '
          f'{lc_time:.2f}s):')
    if parts:
        print('  ' + '  '.join(f'{k}={v:.4f}' for k, v in parts.items()))
    print(f'  sub_lipschitz() = {sub_lipschitz:.4f}')

    for w in warnings:
        print(f'  WARNING: {w}; the breakdown above is incomplete.')

    # The breakdown is only meaningful if it actually explains the bound VRA
    # is computed from. Say so when it does not, rather than implying it does.
    if parts and not warnings:
        product = 1.0
        for value in parts.values():
            product *= value
        if abs(product - sub_lipschitz) > 1e-3 * max(abs(sub_lipschitz), 1.0):
            print(f'  WARNING: the per-component product is {product:.4f}, '
                  f'which differs from sub_lipschitz() = {sub_lipschitz:.4f}. '
                  'The components above do not explain the bound; VRA below '
                  'uses sub_lipschitz().')

    if args.skip_data:
        return

    # -- accuracy and VRA ------------------------------------------------
    if args.eps:
        eps_list = [float(e) for e in args.eps.split(',')]
    else:
        eps = gloro_cfg['eps']
        eps_list = [eps, eps * 2, eps * 3]

    _, _, val_loader, _ = tools.data_loader(
        data_name=dataset_cfg['name'],
        num_classes=dataset_cfg['num_classes'],
        batch_size=args.batch_size,
        data_root=args.data_root,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
        seed=dataset_cfg.get('seed', 2023))

    t = time.time()
    acc, vras, total = evaluate(net, val_loader, device, sub_lipschitz,
                                eps_list, args.num_test)

    print(f'\nEvaluated {total} test images in {time.time() - t:.2f}s:')
    print(f'  clean accuracy = {100 * acc:.2f}%')
    for eps, vra in zip(eps_list, vras):
        print(f'  VRA @ {eps:.4f} ({eps * 255:.0f}/255) = {100 * vra:.2f}%')


if __name__ == '__main__':
    main()
