# Installation

This project uses `pixi` for environment management.

## Prerequisites

Install Pixi first:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Restart your shell after installation, or load Pixi into the current shell if needed.

## Environment Overview

- `default`: CPU-oriented local development environment.
- `cpu`: explicit CPU environment for local users who want a named target.
- `cuda`: GPU training environment using a broadly available PyTorch CUDA runtime on `linux-64`.

## Local CPU Setup

Clone or enter the project directory on a `linux-64` machine:

```bash
cd deep-hedger
```

Install the default CPU environment:

```bash
pixi install
```

Enter the default environment:

```bash
pixi shell
```

Or enter the explicit `cpu` environment:

```bash
pixi shell -e cpu
```

Run a Python version check:

```bash
python --version
```

Check that CPU PyTorch and core packages import correctly:

```bash
python -c "import torch, numpy, scipy, pandas, matplotlib, rich; print(torch.__version__); print(torch.cuda.is_available()); print('CPU environment ready')"
```

For CPU users, `torch.cuda.is_available()` should normally print `False`.

## Cluster CUDA Setup

Move into the project directory:

```bash
cd /path/to/deep-hedger
```

This project is configured only for `linux-64`. Use the `cuda` environment on Linux GPU nodes, not on non-Linux local machines.

Install the CUDA-enabled environment:

```bash
pixi install -e cuda
```

Enter the CUDA environment interactively only if your cluster allows it:

```bash
pixi shell -e cuda
```

For managed clusters, prefer non-interactive `pixi run -e cuda ...` execution over relying on an activated interactive shell.

Verify PyTorch and CUDA visibility:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

This environment uses `pytorch-cuda = "12.4.*"` on `linux-64`, which is a practical cluster-friendly target for broad PyTorch compatibility.

If `torch.cuda.is_available()` returns `False`, confirm that:

- NVIDIA drivers are available on the node.
- The node exposes GPUs to the job.
- The cluster driver is compatible with the CUDA runtime required by the installed PyTorch build.

## Cluster Compatibility Notes

- Keep the `cuda` environment restricted to `linux-64` GPU nodes.
- Install and solve environments on a login node or shared filesystem before running workloads on compute nodes.
- Prefer `pixi run -e cuda ...` for batch execution because it avoids shell activation edge cases.
- Keep the project directory on storage visible to both login and compute nodes.
- If the cluster blocks internet access on compute nodes, complete `pixi install -e cuda` before job submission.
- If your cluster requires site-specific driver or CUDA modules, load those outside the project configuration rather than baking them into project dependencies.

## Explicit CPU Environment Setup

Install the named CPU environment directly:

```bash
pixi install -e cpu
```

Run a one-off CPU PyTorch check:

```bash
pixi run -e cpu python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Non-Interactive Commands

Run a one-off command in the default environment:

```bash
pixi run python --version
```

Run a one-off command in the CUDA environment:

```bash
pixi run -e cuda python -c "import torch; print(torch.cuda.is_available())"
```

## Updating Environments

After editing `pixi.toml`, refresh environments with:

```bash
pixi install
pixi install -e cpu
pixi install -e cuda
```

## Notes

- The `default` and `cpu` environments are intended for local CPU development with PyTorch on `linux-64`.
- The `cuda` environment is intended for GPU-backed training on Linux GPU nodes with a compatible NVIDIA driver stack.
- Current repository scaffolding is documentation-first; training and inference scripts are not implemented yet.
