# CASR: Center-anchored Aligned Semantic Reconstruction for Learning High-level Anatomical Semantics in Chest X-ray Foundation Models

Official PyTorch implementation of CASR, a self-supervised framework that learns anatomically anchored,
cross-scale high-level semantics for chest X-ray foundation models.

## Requirements

- Python 3.8, PyTorch >= 1.10 with CUDA (install PyTorch for your CUDA version first)

```bash
pip install -r requirements.txt
```

## Usage

```
pretrain/     entry: main.py
downstream/   entries: main_classification.py, main_segmentation.py
```

**Pretrained weights.** The pretrained model used for evaluation is available at [this link](https://huggingface.co/Jiarui-Liang/CASR/resolve/main/casr_pretrained.pth).


## Dataset

| Task | Dataset | Description | Train / Val / Test |
|---|---|---|---|
| ChestX-ray14 | NIH ChestX-ray14 | pretraining; 14-class disease classification | 75,312 / 11,212 / 25,596 |
| VinDr-CXR | VinDr-CXR | 6-class disease classification | 12,000 / 3,000 / 3,000 |
| JSRT-Heart | JSRT | heart segmentation | 173 / 37 / 37 |
| JSRT-Clavicle | JSRT | clavicle segmentation | 173 / 37 / 37 |
| VinDr-Rib | VinDr-RibCXR | rib segmentation | 146 / 50 / 49 |
| Montgomery | NIH Montgomery | lung segmentation | 92 / 23 / 23 |

Download the datasets from their official sources and organize them as:

```
data/
├── ChestX-ray14/images/
├── VinDr-CXR/{train_jpeg,test_jpeg}/
├── JSRT/{heart,clavicle}/{train,val,test}/{images,masks}/
├── Montgomery/{train,val,test}/{images,masks}/
└── VinDr-Rib/data/{train,val}/*.json 
```

## Pretraining

```bash
cd pretrain
wget -P checkpoints https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain_full_checkpoint.pth

# Stage 1: warm-up
torchrun --nproc_per_node=4 main.py --mode C --arch vit_base --patch_size 8 \
    --datasets chestxray14 --datasets_config datasets_config.yaml \
    --epochs 200-200-100 --data_granularity 0-1-2 --batch_size_per_gpu 32 \
    --weights checkpoints/dino_vitbase8_pretrain_full_checkpoint.pth --output_dir checkpoints/stage1

# Stage 2: CASR objective
torchrun --nproc_per_node=4 main.py --mode CF --arch vit_base --patch_size 8 \
    --datasets chestxray14 --datasets_config datasets_config.yaml \
    --epochs 63-5-12 --data_granularity 0-1-2 --batch_size_per_gpu 18 --mask_ratio 0.75 \
    --weights checkpoints/stage1/checkpoint.pth --output_dir checkpoints/stage2
```

## Downstream evaluation

```bash
cd downstream

# Classification
python main_classification.py --data_set ChestXray14 --pretrained <path/to/casr_pretrained.pth> \
    --data_dir ../data/ChestX-ray14/images \
    --train_list dataset/Xray14_train_official.txt \
    --val_list dataset/Xray14_val_official.txt \
    --test_list dataset/Xray14_test_official.txt \
    --few_shot X

# Segmentation
python main_segmentation.py --data_set JSRT-Heart --pretrained <path/to/casr_pretrained.pth> \
    --train_data_dir ../data/JSRT/heart/train/images --train_mask_dir ../data/JSRT/heart/train/masks \
    --valid_data_dir ../data/JSRT/heart/val/images --valid_mask_dir ../data/JSRT/heart/val/masks \
    --test_data_dir ../data/JSRT/heart/test/images --test_mask_dir ../data/JSRT/heart/test/masks \
    --few_shot X
```
