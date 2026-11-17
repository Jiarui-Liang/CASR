# CASR

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
torchrun --nproc_per_node=<num_gpus> main.py --mode <C|CF> --datasets chestxray14 --datasets_config datasets_config.yaml --weights <init_checkpoint> --output_dir <output_dir>
```

- `--mode C`: stage 1, CS-SA warm-up, initialized from the [DINO ViT-B/8](https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain_full_checkpoint.pth) checkpoint.
- `--mode CF`: stage 2, full CASR objective, initialized from the stage-1 checkpoint.

## Downstream evaluation

```bash
python main_classification.py --data_set <ChestXray14|VinDrCXR> --pretrained <casr_pretrained.pth> --data_dir <image_dir> --train_list <train_list> --val_list <val_list> --test_list <test_list> --few_shot <n>
python main_segmentation.py --data_set <JSRT-Heart|JSRT-Clavicle|Montgomery|VinDr-Rib> --pretrained <casr_pretrained.pth> --train_data_dir <dir> --train_mask_dir <dir> --valid_data_dir <dir> --valid_mask_dir <dir> --test_data_dir <dir> --test_mask_dir <dir> --few_shot <n>
```
