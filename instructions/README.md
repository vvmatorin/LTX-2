# Better Training

## Installation

1. Install `uv` following the [instructions](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)
2. `git clone https://github.com/vvmatorin/LTX-2 --branch feature/better-training`
3. Move into the library: `cd LTX-2/`
4. Run `uv sync` to update the packages

## Training

0. The `config_i2v.yaml` imitates training schedule from the [ai-toolkit](https://github.com/ostris/ai-toolkit)
1. Activate the environment: `source .venv/bin/activate`
2. Copy `config_i2v.yaml` to `packages/ltx-trainer/configs`
3. Replace all of `/path/to/...` placeholders with correct values
4. Based on your configuration run either:

```
# MULTI-GPU TRAINING (2 GPUs)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1 \
    accelerate launch --config_file configs/accelerate/ddp.yaml scripts/train.py configs/config_i2v.yaml

# SINGLE-GPU TRAINING
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
    python scripts/train.py configs/config_i2v.yaml
```
