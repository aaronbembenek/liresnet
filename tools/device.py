import torch


def get_device(name: str = 'auto') -> torch.device:
    """Resolve a device name into a torch.device.

    Args:
        name (str): one of `auto`, `cuda`, `mps` or `cpu`. `auto` prefers
            CUDA, then Apple Silicon (MPS), then CPU.
    """
    name = name.lower()

    if name == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    if name.startswith('cuda'):
        if not torch.cuda.is_available():
            raise ValueError('CUDA is not available on this machine.')
        return torch.device(name)

    if name == 'mps':
        if not torch.backends.mps.is_available():
            raise ValueError('MPS is not available on this machine. It needs '
                             'Apple Silicon and a torch build with MPS.')
        return torch.device('mps')

    if name == 'cpu':
        return torch.device('cpu')

    raise ValueError(f'Unsupported device `{name}`. Use one of '
                     '`auto`, `cuda`, `mps`, `cpu`.')
