# HSR-Det: Open-Vocabulary Food Detection with VLM Name Prompts and CFCR


## Introduction

This repository contains the code release of **HSR-Det**, an open-vocabulary food detection project built on top of **Object-Centric OVD**, **Detectron2**, and a customized zero-shot classification pipeline for the **ZSFooD2** dataset.

Compared with the original upstream OVD codebase, this version is adapted for food-category open-vocabulary detection and includes:

- ZSFooD2 dataset registration and zero-shot splits
- VLM-generated class/name prompt features
- PIS-based training on top of the OVD framework
- paper source packages and qualitative visualization assets

## Highlights

- Base framework: Object-Centric Open-Vocabulary Detection
- Detection backbone: Detectron2-style OVD training pipeline
- Dataset: `ZSFooD2`
- Number of categories: `228`
- Main training config used in this repo: `configs/coco/COCO_OVD_Base_PIS.yaml`

## Method Overview

This project follows the Object-Centric OVD training paradigm and further adapts it to the food detection setting.

The current repository version combines the following ingredients:

- zero-shot classifier weights generated from category prompts
- pseudo image-level supervision (PIS)
- VLM description features for train/eval stages
- confusion-aware fine-grained refinement related components
- custom food-category open-vocabulary split on ZSFooD2

## Paper Figures

### Qualitative Comparison

<p align="center">
  <img src="paper/comparison_4x5_grid.png" alt="qualitative comparison" width="100%">
</p>


## Repository Structure

```text
.
├── configs/                      # training and evaluation configs
├── ovd/                          # core model, dataset, evaluation, roi head logic
├── preprocessing/                # confusion mining, description generation, analysis scripts
├── tools/                        # dataset and feature preprocessing scripts
├── utils/                        # auxiliary utilities
├── paper/                        # paper source packages and visualization figures
├── output/                       # local training outputs and checkpoints
├── detectron2/                   # local detectron2 source
└── train_net.py                  # main training / evaluation entry
```

## Environment Setup

The upstream codebase was tested with **PyTorch 1.10.0** and **CUDA 11.3**. For this repository, a practical setup is:

```bash
conda create -n dpdn python=3.12 -y
conda activate dpdn

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

cd detectron2
pip install -e .
cd ..
```

## Dataset Preparation
The registered food dataset layout is:

```text
/.../datasets/
└── ZSFooD2/
    ├── train2017/
    ├── val2017/
    ├── annotations/
    │   ├── instances_train2017.json
    │   ├── instances_val2017.json
    │   ├── captions_train2017_tags_allcaps_pis.json
    │   └── ...
    └── zero-shot/
        ├── instances_train2017_seen_2_oriorder.json
        ├── instances_val2017_all_2_oriorder.json
        └── instances_train2017_seen_2_oriorder_cat_info.json
```
Additional data required for training: https://huggingface.co/datasets/LTaiQin/ZSFooD_vlm_des

Before training on another machine, you should update the absolute paths in the config files and dataset registration code.

## Main Training Config

The experiment in this repository was trained with:

```text
configs/coco/COCO_OVD_Base_PIS.yaml
```

Important path fields that usually need to be edited before running:

- `MODEL.PIS_PROP_PATH`
- `MODEL.VLM_TRAIN_DES_PATH`
- `MODEL.VLM_EVAL_DES_PATH`
- `MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH`
- `MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH`
- dataset root defined in `ovd/datasets/coco_zeroshot.py`

## Training

```bash
python train_net.py \
  --num-gpus 8 \
  --config-file configs/coco/COCO_OVD_Base_PIS.yaml
```

## Evaluation

Example evaluation command for the current local checkpoint:

```bash
python train_net.py \
  --num-gpus 8 \
  --config-file configs/coco/COCO_OVD_Base_PIS.yaml \
  --eval-only \
  MODEL.WEIGHTS output/coco_ovd_PIS_vlm_name_prompt_cfcr/model_best.pth
```
The model weights are open source at https://huggingface.co/LTaiQin/DPDN-Det

## Acknowledgement

This codebase is adapted from:

- [Object-Centric OVD](https://github.com/hanoonaR/object-centric-ovd)
- [Detectron2](https://github.com/facebookresearch/detectron2)

If your final paper also depends on additional VLM or proposal-generation projects, you can add them here before publishing.
