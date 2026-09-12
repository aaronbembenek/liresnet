import torch
import yaml

import models


def load_config(ckpt: dict, config_path: str = '') -> dict:
    """Config for a checkpoint: an explicit file, else the stored one.

    `train.py` saves the config it trained with into the checkpoint, so the
    usual case needs no config file at all.
    """
    if config_path:
        with open(config_path, 'r') as f:
            return yaml.load(f, Loader=yaml.Loader)
    if 'configs' in ckpt:
        return ckpt['configs']
    raise ValueError('The checkpoint stores no config; pass --config.')


def load_gloronet(checkpoint_path: str, config_path: str = ''):
    """Build a `GloroNet` from a checkpoint and load its weights.

    Returns `(net, cfg, ckpt)`. The net is left in whatever mode `GloroNet`
    starts in; callers are expected to `.to(device).eval()`.

    Missing and unexpected keys are treated differently on purpose, because
    only one of them can corrupt a result:

    - A **missing** key means the model needs state the checkpoint does not
      have, which would be left at its random initialization. Every number
      downstream would then be meaningless while still looking plausible, so
      this is a hard error with no override.
    - An **unexpected** key means the checkpoint carries state the model no
      longer uses. Nothing is left uninitialized, so this only warns.
    """
    ckpt = torch.load(checkpoint_path, 'cpu', weights_only=False)
    cfg = load_config(ckpt, config_path)

    net = models.GloroNet(**cfg['model'], **cfg['dataset'])
    missing, unexpected = net.load_state_dict(ckpt['backbone'], strict=False)

    if missing:
        raise SystemExit(
            f'Cannot load {checkpoint_path}: the model expects state that the '
            f'checkpoint does not contain.\n\n  missing: '
            f'{", ".join(missing)}\n\n'
            'Loading anyway would leave those tensors randomly initialized '
            'and silently invalidate every number that follows, so this is '
            'refused.\n\n'
            'If you added scratch state for a new Lipschitz computation (a '
            'cached power-iteration vector, say), register it as '
            '`register_buffer(..., persistent=False)` so it stays out of the '
            'checkpoint and is rebuilt on each run. If you added something '
            'that genuinely needs to be learned, the model has to be '
            'retrained.')

    if unexpected:
        print(f'WARNING: {checkpoint_path} contains state this model no '
              f'longer uses: {", ".join(unexpected)}. It has been ignored.')

    return net, cfg, ckpt
