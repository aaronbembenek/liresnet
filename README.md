# 🚀 LiResNet: An Network Architecture for Training Certifiable Robust Models


This repository provides the implementation of our cutting-edge research on certifiable robust models. We proudly present the LiResNet model introduced at NeurIPS 2023 and its subsequent improvements now available as a preprint on arXiv. Our works are based on the training and certification schemes in [GloRo Nets](https://arxiv.org/abs/2102.08452). 

- [NeurIPS 2023] [Unlocking Deterministic Robustness Certification on ImageNet](https://arxiv.org/abs/2301.12549)
- [ICLR 2024] [Effectively Leveraging Capacity for Improved Deterministic Robustness Certification](https://openreview.net/forum?id=qz3mcn99cu)


## 🚀 Getting Started:
- For training a model, check out our `run.sh` as a starting point.
- Dive into our [configs](/configs) for additional dataset configurations.

## 🛠️ Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync          # creates .venv and installs everything
```

Run anything in the project with `uv run` from the repository root:

```bash
uv run python train.py --config configs/cifar10.yaml
```

### Choosing a device

`train.py` takes `--device auto|cuda|mps|cpu` (default `auto`, which prefers
CUDA, then Apple Silicon's MPS, then CPU). Single-process training needs no
launcher; `--launcher` defaults to `none`. Multi-GPU training is unchanged --
see `run.sh`, which passes `--launcher=pytorch` to `torchrun`.

```bash
# Apple Silicon GPU
uv run python train.py --config configs/cifar10_small.yaml --device mps --num_workers 2

# CPU
uv run python train.py --config configs/cifar10_small.yaml --device cpu
```

No `PYTORCH_ENABLE_MPS_FALLBACK` is required: every linear-algebra op this
model uses has a native MPS kernel. On the evaluation path PyTorch's own MPS
SVD kernel may print a `matrix too large to stage in MPS threadgroup memory`
warning and fall back to the CPU internally; this is harmless and costs only a
little evaluation speed.

On this hardware `--num_workers 2` is faster than the default of 4 -- the
augmentation pipeline contends with training for cores.

### A quick config

[`configs/cifar10_small.yaml`](configs/cifar10_small.yaml) trains a much
smaller model (0.8M parameters) for 30 epochs. It is meant for experimenting
with the certification machinery, **not** for reproducing the table below.

Measured on an M-series Mac with `--device mps --num_workers 2`:

| | wall time | clean accuracy | VRA@36/255 |
|:--|:--:|:--:|:--:|
| `cifar10_small.yaml`, 30 epochs | 9.9 min | 58.6% | 50.8% |

The same config on `--device cpu` is roughly six times slower per epoch
(~2.7 min/epoch), so budget a bit over an hour for the full 30.

The model size is configurable under `model:`. Besides the conv trunk's
`depth` and `width`, `out_dim` and `mlp_depth` size the MLP head -- which
dominates both parameter count and training cost, since every forward pass
orthogonalizes a weight stack of shape `(mlp_depth, out_dim, out_dim)`. Two
constraints are enforced: `out_dim` must be even, and `out_dim` must not
exceed `width * (feature_size // 4) ** 2`.

### Experimenting with the Lipschitz bound

The certificate rests on an *upper* bound on the network's Lipschitz constant,
computed by `GloroNet.sub_lipschitz()`. Experimenting with how that bound is
computed needs no training: the bound is a pure function of the weights, and
accuracy and VRA need only a forward pass over a trained checkpoint.

A trained checkpoint for `configs/cifar10_small.yaml` ships in
[`pretrained/`](pretrained), so you can start immediately:

```bash
# the bound alone -- no dataset touched, ~2s on CPU including startup
uv run python eval.py --checkpoint pretrained/cifar10_small.pth \
    --device cpu --skip_data

# accuracy and VRA on the full test set
uv run python eval.py --checkpoint pretrained/cifar10_small.pth --device cpu
```

which reports a per-component breakdown of the bound alongside the results:

```
Lipschitz bound (500 power iterations, 0.96s):
  stem=2.3007  conv=1.2003  neck=0.9998  linear=0.9997
  sub_lipschitz (product) = 2.7608

Evaluated 10000 test images in 22.27s:
  clean accuracy = 59.98%
  VRA @ 0.1412 (36/255) = 50.97%
  VRA @ 0.2824 (72/255) = 42.94%
  VRA @ 0.4235 (108/255) = 34.88%
```

Use `--num_test 1000` for a fast subset (about a second on CPU), and
`--eps` to pick the radii.

The `lipschitz()` implementations you would edit are
[`Conv2d`](models/layers/conv.py) (stem, power iteration),
[`LiResConv`](models/blocks.py) (conv trunk, power iteration),
[`LiResMLP`](models/blocks.py) (spectral norm), and
[`Map2Vec`](models/model.py) (neck). `spectral_norm()` in
[models/linalg_utils.py](models/linalg_utils.py) sits in that path.

Changing how the bound is computed does not invalidate the shipped
checkpoint: the weights do not depend on it, so a new bound can be evaluated
against a trained model without retraining. Two things are checked for you:

- If your model needs state the checkpoint does not have, loading is
  **refused** rather than leaving tensors randomly initialized -- that would
  invalidate every number downstream while still looking plausible. If you
  added scratch state for a new Lipschitz computation, such as a cached
  power-iteration vector, register it with
  `register_buffer(..., persistent=False)` so it is rebuilt on each run
  instead of being expected in the checkpoint.
- If you restructure `sub_lipschitz()` so it is no longer the product of the
  four components, `eval.py` says so rather than presenting a breakdown that
  does not add up.

**A smaller bound always looks like an improvement.** VRA is computed from
whatever `sub_lipschitz()` returns, and a smaller bound certifies a larger
radius -- so *any* change that shrinks the bound raises VRA, whether or not
the bound is still valid. Halving it on the shipped checkpoint lifts
VRA@36/255 from 50.9% to 55.4% while clean accuracy does not move at all,
because the network itself is unchanged. **Rising VRA is therefore not
evidence that a change is correct; it is equally the signature of an unsound
bound.** Check soundness separately, with the lower-bound search below.

**Checking that a bound is sound and tight.** The power method converges to
the spectral norm *from below*, so too few iterations silently produces an
*under*estimate -- an optimistic bound that the certificate does not actually
support. `--lc_sweep` makes that visible:

```bash
uv run python eval.py --checkpoint pretrained/cifar10_small.pth \
    --device cpu --lc_sweep 1,5,10,50,100,500
```

To check the bound from the other side, `tools/check_lipschitz_lower_bound.py`
searches for input pairs maximizing `||f(x) - f(x')|| / ||x - x'||`, giving an
empirical *lower* bound. A sound upper bound must exceed it, and the gap is
how loose the bound is:

```bash
uv run python tools/check_lipschitz_lower_bound.py \
    --checkpoint pretrained/cifar10_small.pth --device cpu
```

For the shipped checkpoint this reaches a lower bound of 2.52 against the
computed upper bound of 2.76 -- within about 9%. The search is sensitive to
`--lr`; too small a value leaves it still climbing and understates the bound.

### Results for cifar10_small_stride1

The `cifar10_small` configuration, with all convolutional strides set to 1.

Ran this command:

```
uv run python eval.py --checkpoint pretrained/cifar10_small_stride1.pth --device mps --num_workers 2
```

and got this:

```
Checkpoint : pretrained/cifar10_small_stride1.pth
Device     : mps
Trained    : Epoch 29: Train acc 58.26%, 39.74%; val acc 62.75%, 53.51%. sub_lipschitz: 3.78. Time: 1.19 mins.

Lipschitz bound (500 power iterations, 0.71s):
  stem=4.0648  conv=0.9313  neck=0.9998  linear=0.9998
  sub_lipschitz() = 3.7858

Evaluated 10000 test images in 15.22s:
  clean accuracy = 62.75%
  VRA @ 0.1412 (36/255) = 53.50%
  VRA @ 0.2824 (72/255) = 44.65%
  VRA @ 0.4235 (108/255) = 35.98%
```

### Diffusion-generated training data (`use_ddpm`)

The published recipe trains on the real dataset *plus* a large set of
diffusion-generated images, which is where the numbers in the table come from.
Each config has a `use_ddpm` flag under `training:`; when it is on, training
loads `data/c{num_classes}_ddpm.npz`.

**That file is not distributed with this repository and there is no code here
that downloads it.** With `use_ddpm: True` and no such file, training stops at
startup with a `FileNotFoundError`. Set `use_ddpm: False` to train on the real
data alone (as `configs/cifar10_small.yaml` does), which trains fine but will
not reproduce the published accuracy.

To use the real recipe, obtain a generated-image set from the robustness
literature these papers build on -- the DDPM data of [Rebuffi et
al. (2021)](https://arxiv.org/abs/2103.01946) or the EDM data of [Wang et
al. (2023)](https://arxiv.org/abs/2302.04638) -- then convert whatever layout
it ships in:

```bash
uv run python tools/dataset/make_ddpm_npz.py <downloaded.npz> --num_classes 10
```

The converter normalizes the arrays to the `uint8` NHWC images and `image` /
`label` keys that the loader expects, and validates shapes and label ranges.

## 📈 Main Results:
| dataset       | clean accuracy | VRA@36/255 | VRA@72/255 | VRA@108/255 |
|:-------------:|:--------------:|:----------:|:----------:|:-----------:|
| CIFAR-10      | 87.0%          | 78.1%      | 66.6%      | 53.5%       |
| CIFAR-100     | 62.1%          | 50.1%      | 38.5%      | 29.0%       |
| Tiny-ImageNet | 48.4%          | 37.0%      | 26.8%      | 18.6%       |
| ImageNet      | 49.0%          | 38.3%      | -          | -           |

## 🤝 Support:
- Encountering issues? Submit an Issue.
- For specific inquiries, 📧 drop an email to `kaihu@cmu.edu`.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citations
If you find this repository useful, consider to use the following citations

```
@INPROCEEDINGS{hu2023scaling,
    title={Unlocking Deterministic Robustness Certification on ImageNet},
    author={Kai Hu and Andy Zou and Zifan Wang and Klas Leino and Matt Fredrikson},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
    url={https://openreview.net/forum?id=SHyVaWGTO4}
}

@misc{hu2023recipe,
    title={A Recipe for Improved Certifiable Robustness: Capacity and Data}, 
    author={Kai Hu and Klas Leino and Zifan Wang and Matt Fredrikson},
    year={2023},
    eprint={2310.02513},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}

@INPROCEEDINGS{leino21gloro,
    title = {Globally-Robust Neural Networks},
    author = {Klas Leino and Zifan Wang and Matt Fredrikson},
    booktitle = {International Conference on Machine Learning (ICML)},
    year = {2021}
}
```
