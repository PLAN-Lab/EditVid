<div align="center">

# 🎬 One Editor, Many Edits: A Unified Training-Free Framework for Diverse Video Editing

**[Adheesh Sunil Juvekar](https://plan-lab.github.io/member.html?id=adheesh-juvekar)** · **[Onkar Kishor Susladkar](https://plan-lab.github.io/member.html?id=onkar-susladkar)** · **[Kiet A. Nguyen](https://plan-lab.github.io/member.html?id=kiet-nguyen)** · **[Muntasir Wahed](https://plan-lab.github.io/member.html?id=muntasir-wahed)** · **[Nabeel Bashir](https://plan-lab.github.io/member.html?id=nabeel-bashir)** · **[Xiaona Zhou](https://plan-lab.github.io/member.html?id=xiaona-zhou)** · **[Tianjiao Yu](https://plan-lab.github.io/member.html?id=tianjiao%28joey%29-yu)** · **[Vedant Shah](https://plan-lab.github.io/member.html?id=vedant-shah)** · **[Ismini Lourentzou](https://plan-lab.github.io/member.html?id=ismini-lourentzou)**

University of Illinois Urbana-Champaign

[![Project page](https://img.shields.io/badge/Project-Page-2563eb?logo=googlechrome&logoColor=white)](https://plan-lab.github.io/editvid)
[![arXiv](https://img.shields.io/badge/arXiv-2609.04190-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2609.04190)

</div>

## ✨ Overview

![EditVid video editing comparison](assets/editvid-elephant-comparisons.webp)

- 🎯 We introduce **EditVid**, a training-free framework that supports both instruction-guided and subject-guided video editing, leveraging MM-DiT-based image editors as strong priors across diverse editing settings.

- 🧠 We identify temporal context as a key factor in MM-DiT representation reuse and develop a RoPE-aware local–global design that uses adjacent-frame key–value memory for short-range coherence and confidence- and cycle-consistent token transfer for long-range preservation.

- 📊 Comprehensive quantitative, human, robustness, and cross-backbone evaluations demonstrate **EditVid**'s strong performance in temporally consistent video editing without video-specific training.

Official implementation of **EditVid**, a training-free framework for diverse video editing, supporting various instruction-guided and reference-guided edits in one framework, including style transfer, attribute modification, object insertion, part-level editing, and subject replacement.

## 🛠️ Installation

The code is tested with Python 3.11, PyTorch 2.7.0, CUDA 12.8.

Access to the gated `black-forest-labs/FLUX.2-klein-9B` model is required.


```bash
git clone https://github.com/PLAN-Lab/editvid.git
cd editvid
conda create -n editvid python=3.11 pip -y
conda activate editvid
conda install ffmpeg -y
python -m pip install -r requirements.txt
```

Next, choose one of the following Diffusers installation methods. Both use the exact tested Diffusers commit `769a1f3a120dd2a483b0e99bd6b13464b8ee62fb` and leave its source code unmodified.

### Option A: installation helper

```bash
python scripts/install_diffusers.py
```

The helper fetches the pinned commit, confirms that its checkout is clean, and installs Diffusers and the local `editvid` package.

### Option B: manual installation

```bash
mkdir -p .vendor
git clone https://github.com/huggingface/diffusers.git .vendor/diffusers
git -C .vendor/diffusers checkout --detach 769a1f3a120dd2a483b0e99bd6b13464b8ee62fb
python -m pip install --no-deps -e .vendor/diffusers
python -m pip install --no-deps -e .
```

Authenticate for the gated model and inspect the installed versions:

```bash
hf auth login
python --version
python -c "import torch; print('torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'GPU available:', torch.cuda.is_available())"
python -c "import diffusers; print('diffusers:', diffusers.__version__, diffusers.__file__)"
python -c "from editvid import EditVidPipeline, EditVidTransformer2DModel; print(EditVidPipeline.__module__, EditVidTransformer2DModel.__module__)"
git -C .vendor/diffusers rev-parse HEAD
git -C .vendor/diffusers status --short
```

## 🎞️ Run the examples

The bundled manifest contains six full-length source videos from the project-page showcase. `run_editvid.py` passes each `edit_prompt` to the model exactly as written in the manifest.

### Test the setup with one chunk

Use one source video and cap it at 16 frames with `--max-frames`, `--chunk-size`, this performs exactly one generation chunk (if memory allows):

```bash
CUDA_VISIBLE_DEVICES=0 python run_editvid.py \
  --manifest-json examples/manifest.json \
  --source-videos-dir examples/source_videos \
  --rows 0 \
  --max-frames 16 \
  --chunk-size 16 \
  --fallback-chunk-sizes 8,4 \
  --output-root outputs \
  --run-name setup-test
```

The equivalent convenience command is:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_example.sh
```

### Generate the full videos
Omit `--max-frames` to process every frame of every bundled source video. Use this version to visualize complete results or produce outputs for evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python run_editvid.py \
  --manifest-json examples/manifest.json \
  --source-videos-dir examples/source_videos \
  --chunk-size 32 \
  --fallback-chunk-sizes 16,8,4 \
  --output-root outputs \
  --run-name examples-full
```

Results are written under `outputs/<run-name>/`. For custom experiments, edit or replace `examples/manifest.json`; run `python run_editvid.py --help` for the remaining options.

## 📝 Citation

If you find EditVid useful for your research, please cite our paper:

```bibtex
@article{juvekar2026editvid,
  title   = {One Editor, Many Edits: A Unified Training-Free Framework for Diverse Video Editing},
  author  = {Adheesh Sunil Juvekar and Onkar Kishor Susladkar and Kiet A. Nguyen and Muntasir Wahed and Nabeel Bashir and Xiaona Zhou and Tianjiao Yu and Vedant Shah and Ismini Lourentzou},
  journal = {arXiv preprint arXiv:2609.04190},
  year    = {2026},
}
```
