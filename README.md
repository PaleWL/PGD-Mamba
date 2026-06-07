# PGD-Mamba

This repository provides a PyTorch implementation of PGD-Mamba, a prior-guided dual-domain Mamba network for accelerated MRI reconstruction. The model combines image-domain Mamba reconstruction, k-space/frequency-domain modeling, frozen CLIP prior extraction, and LCMA-based feature modulation with data consistency.

## Environment

```bash
conda env create -f environment.yml
conda activate mamba_recon_env
```

If the local CUDA extensions are not installed automatically, install them from the bundled folders:

```bash
cd causal-conv1d
pip install .

cd ../mamba
pip install .
```

The CLIP prior uses `open_clip_torch`. If it is missing:

```bash
pip install open_clip_torch
```

## Datasets

Download the public datasets from:

- IXI: https://brain-development.org/ixi-dataset/
- FastMRI: https://fastmri.org/

Place the processed files under:

```text
code/datasets/ixi/
code/datasets/fastmri/
```

Expected examples:

```text
code/datasets/ixi/ixi_train.h5
code/datasets/ixi/ixi_val.h5
code/datasets/fastmri/fastmri_train.h5
code/datasets/fastmri/fastmri_val.h5
```

Dataset files are not included in this repository.

## Training Example

```bash
cd code
python train.py \
  --dataset fastmri \
  --model mamba_unrolled \
  --exp pgd_mamba_fastmri_8x \
  --batch_size 2 \
  --gpu_id 0 \
  --max_iterations 10000 \
  --acceleration 8 \
  --use_fourier 1 \
  --window_size 0 \
  --opts MODEL.USE_CLIP_PRIOR True
```

## Citation

If this project is useful for your research, please cite the related work on MambaRecon, CLIP, and accelerated MRI reconstruction. The PGD-Mamba citation will be added after publication.
