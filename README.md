# KSPR-Det

**Knowledge-Constrained Semantic Prototype Reconstruction for Fine-Grained Open-Vocabulary Object Detection**

This repository provides the detector-side implementation and released resources for **KSPR-Det**, a semantic-to-visibility relation transfer framework for fine-grained open-vocabulary object detection (FG-OVD). The current `zsfood` branch targets **ZSFooD2** and is built on [Object-Centric OVD](https://github.com/hanoonaR/object-centric-ovd), [Detectron2](https://github.com/facebookresearch/detectron2), and [CLIP](https://github.com/openai/CLIP).

## Method

KSPR-Det improves region classification from the class and image sides.

- **Knowledge-Constrained Semantic Prototype Reconstruction (KSPR)** estimates semantic-to-visibility relations from base-category semantics and crop-grounded descriptions. A Correlation Atlas organizes the resulting cue-attribute associations and guides the reconstruction of cached semantic prototypes.
- **Context-Conditioned Feature Refinement (CFCR)** uses current-image appearance context to refine region-of-interest (RoI) features. It combines fine-level and group-level context through DBSCAN grouping, attention, and gated residual fusion.

The detector consumes the offline semantic cache through `MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH` and precomputed image-context features through `MODEL.VLM_TRAIN_DES_PATH` and `MODEL.VLM_EVAL_DES_PATH`. The complete offline Atlas-construction and candidate-generation pipeline is not included in this branch and is not invoked by `train_net.py`.

## Results

The following values are the manuscript means over detector seeds **9, 99, and 999**. AP50-N, AP50-B, and AP50-All denote AP50 on novel, base, and all valid categories, respectively.

| Dataset | Setting | Online LVLM | AP50-N | AP50-B | AP50-All |
|:--|:--|:--:|--:|--:|--:|
| ZSFooD | Class-name baseline | No | 10.40 | 90.15 | 71.78 |
| ZSFooD | KSPR reconstruction | No | 23.43 | 90.62 | **75.14** |
| ZSFooD | KSPR-Det | Yes | **28.05** | 89.03 | 74.98 |
| UEC FOOD 256 | Class-name baseline | No | 9.50 | 13.35 | 11.97 |
| UEC FOOD 256 | KSPR reconstruction | No | 27.31 | 21.52 | 23.60 |
| UEC FOOD 256 | KSPR-Det | Yes | **30.01** | **24.14** | **26.25** |

This branch currently provides the ZSFooD2 detector pipeline. The UEC FOOD 256 values are included to summarize the corresponding manuscript result.

## Released Resources

| Resource | Link |
|:--|:--|
| ZSFooD2 appearance features | [Hugging Face](https://huggingface.co/datasets/LTaiQin/ZSFooD_vlm_des) |
| Released detector checkpoint (legacy repository name) | [Hugging Face](https://huggingface.co/LTaiQin/DPDN-Det) |

## Repository Structure

```text
.
├── configs/                         # training and evaluation configurations
├── datasets/zeroshot_weights/       # bundled class-name prototype baselines
├── ovd/datasets/                    # ZSFooD2 registration and split metadata
├── ovd/evaluation/                  # base/novel COCO-style evaluation
├── ovd/modeling/                    # detector and refinement modules
│   └── roi_heads/DDFM.py            # fine/group context fusion used by CFCR
├── ovd/transforms/                  # loaders for precomputed appearance features
├── paper/                           # qualitative visualization
├── tools/                           # dataset and feature preparation utilities
├── train_net.py                     # training and evaluation entry point
└── requirements.txt
```

## Installation

The current code was checked with the following environment:

- Python 3.9
- PyTorch 1.12.1
- torchvision 0.13.1
- CUDA 11.3
- Detectron2 commit `9604f5995cc628619f0e4fd913453b4d7d61db3f`

Create the environment and install the pinned Detectron2 revision:

```bash
git clone --branch zsfood https://github.com/LTaiQin/KSPR_Det.git
cd KSPR_Det

conda create -n ksprdet python=3.9 -y
conda activate ksprdet
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch -y

python -m pip install -r requirements.txt

git clone https://github.com/facebookresearch/detectron2.git detectron2-src
git -C detectron2-src checkout 9604f5995cc628619f0e4fd913453b4d7d61db3f
python -m pip install -e detectron2-src
```

## Dataset Preparation

The registered ZSFooD2 split contains **184 base categories** and **44 novel categories**. Arrange the dataset as follows:

```text
/path/to/datasets/
└── ZSFooD2/
    ├── train2017/
    ├── val2017/
    ├── annotations/
    │   ├── instances_train2017.json
    │   ├── instances_val2017.json
    │   └── captions_train2017_tags_allcaps_pis.json
    └── zero-shot/
        ├── instances_train2017_seen_2_oriorder.json
        ├── instances_val2017_all_2_oriorder.json
        └── instances_train2017_seen_2_oriorder_cat_info.json
```

Download the released appearance features from [Hugging Face](https://huggingface.co/datasets/LTaiQin/ZSFooD_vlm_des). The train and validation feature directories must contain one `.pkl` file per image basename.

Before running the code, set the dataset root in `ovd/datasets/coco_zeroshot.py` and replace the machine-specific paths in the configuration files.

## Configuration

The primary configuration for this branch is:

```text
configs/coco/COCO_OVD_Base_PIS.yaml
```

Update these fields before training or evaluation:

| Field | Expected value |
|:--|:--|
| `MODEL.WEIGHTS` | initialization or evaluation checkpoint |
| `MODEL.PIS_PROP_PATH` | directory containing pseudo proposal `.pkl` files |
| `MODEL.VLM_TRAIN_DES_PATH` | per-image training appearance features |
| `MODEL.VLM_EVAL_DES_PATH` | per-image validation appearance features |
| `MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH` | semantic prototype cache in NumPy format |
| `MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH` | ZSFooD2 base-category metadata JSON |
| `OUTPUT_DIR` | experiment output directory |

The bundled configuration retains absolute paths from the original experiment. Replace all of them before execution.

## Training

The manuscript uses a 90k-iteration schedule with a total batch size of 24. An eight-GPU run can be launched with:

```bash
python train_net.py \
  --num-gpus 8 \
  --config-file configs/coco/COCO_OVD_Base_PIS.yaml \
  SEED 9 \
  OUTPUT_DIR output/kspr_det_seed9
```

Repeat the matched run with `SEED 99` and `SEED 999` to reproduce the reported three-seed aggregation.

## Evaluation

Evaluate a checkpoint on the generalized base/novel split with:

```bash
python train_net.py \
  --num-gpus 1 \
  --config-file configs/coco/COCO_OVD_Base_PIS.yaml \
  --eval-only \
  MODEL.WEIGHTS /path/to/model.pth
```

The evaluator reports base, novel, and all-category metrics. Keep the dataset split, semantic cache, appearance features, and checkpoint fixed when comparing methods.

## Qualitative Results

<p align="center">
  <img src="paper/comparison_4x5_grid.png" alt="KSPR-Det qualitative comparison on ZSFooD" width="100%">
</p>

## Citation

If you use this repository, please cite:

```bibtex
@misc{liu2026ksprdet,
  title  = {Knowledge-Constrained Semantic Prototype Reconstruction for Fine-Grained Open-Vocabulary Object Detection},
  author = {Liu, Shoulong and Min, Weiqing and Wang, Xinlong and Yao, Tao and Sheng, Guorui and Jiang, Shuqiang},
  year   = {2026}
}
```

## Acknowledgements

This codebase is developed from [Object-Centric OVD](https://github.com/hanoonaR/object-centric-ovd) and [Detectron2](https://github.com/facebookresearch/detectron2). It also uses [CLIP](https://github.com/openai/CLIP) and scikit-learn. We thank the authors for releasing their implementations.
