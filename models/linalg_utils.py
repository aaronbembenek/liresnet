import torch
from torch import Tensor


def _gram_spectral_norm(W: Tensor) -> Tensor:
    """Largest singular value via the symmetric eigensolver on W's gram matrix.

    `sqrt(lambda_max(W W^T))` equals `sigma_max(W)`. Uses whichever of
    `W W^T` / `W^T W` is smaller. Forming the gram matrix squares the
    condition number, which is harmless here: the matrices this is used on
    are near-orthogonal, so cond(W) ~ 1.
    """
    gram = W @ W.mT if W.shape[-2] <= W.shape[-1] else W.mT @ W
    return torch.linalg.eigvalsh(gram).clamp_min(0).amax(-1).sqrt()


def spectral_norm(W: Tensor) -> Tensor:
    """Largest singular value of `W`, batched over any leading dimensions.

    Two robustness measures over a plain `torch.linalg.matrix_norm(W, ord=2)`:

    1. On MPS the computation is moved to the CPU. PyTorch's MPS SVD kernel
       declines to stage matrices this large in threadgroup memory and falls
       back on its own anyway, and it is the less robust of the two.
    2. If the SVD fails to converge, fall back to the gram-matrix eigensolver.
       These matrices come out of `CholeskyOrth` and are near-orthogonal, so
       essentially all of their singular values sit at ~1 -- the case LAPACK's
       divide-and-conquer SVD (`gesdd`) reports as "too many repeated singular
       values". Whether it converges turns on the last few decimal places, so
       this fires unpredictably rather than never. The eigensolver is stable
       on this structure and agrees with the SVD to ~1e-6 relative.

    Only the Lipschitz bound calls this, and only in eval mode, so the CPU
    transfer costs a validation pass rather than a training step.
    """
    src = W.device
    if src.type == 'mps':
        W = W.cpu()

    try:
        s = torch.linalg.matrix_norm(W, ord=2)
    except torch.linalg.LinAlgError:
        s = _gram_spectral_norm(W)

    return s.to(src)
