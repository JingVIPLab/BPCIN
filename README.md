# BPCIN

Official Repository for "Bigraph Proto-calibration Cross-modal Inference Network for Few-shot Visual Question Answering".

## Installation

```bash
conda create -n bpcin python=3.9
conda activate bpcin

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

python -m pip install 'https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.5.0/en_core_web_lg-3.5.0-py3-none-any.whl'
python -m pip install 'git+https://github.com/openai/CLIP.git'
python -m pip install "visdom==0.2.4" --no-build-isolation -v

pip install -r requirements.txt
```

## Setup

### Environment

Please modify the source code of CLIP in the following ways (clip/model.py).

In class **ModifiedResNet** (function **forward**), please add and modify the following lines.

```python
def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        last_sec_layer = x		# added
        x = self.layer4(x)
        last_layer = x			# added
        x = self.attnpool(x)

        return x, last_sec_layer, last_layer		# modified
```

### Datasets

COCO-QA and VQA v2 use the same MS COCO 2014 images and VinVL features, so these large files can be shared instead of duplicated.

Only the split files have to live under `--dataset_root`. Images and VinVL features are looked up there by default too, but they can be pointed at another
disk with `--image_root` and `--vinvl_root` .

```text
data/                            # --dataset_root: splits only
|-- FSL COCO-QA/
|   |-- train.pth
|   |-- val.pth
|   `-- test.pth
|-- FSL VQA/
|   |-- train.pth
|   |-- val.pth
|   `-- test.pth
`-- COCO-QA/
    |-- train_fpait.pth
    `-- test_fpait.pth

<image_root>/                    # defaults to --dataset_root
|-- COCO/
    |-- COCO_train2014_000000000009.jpg
    |-- COCO_val2014_000000000042.jpg
    `-- ...

<vinvl_root>/                    # defaults to --dataset_root
|-- vinvl/
|   |-- COCO_train2014_000000000009.jpg.npz
|   |-- COCO_val2014_000000000042.jpg.npz
|   `-- ...
`-- predict_vinvl.npy
```

#### COCO images

Every supplied split uses images from both COCO `train2014` and `val2014`. Download both archives:

- [COCO download page](https://cocodataset.org/#download)

After extraction, place (or link) all `.jpg` files from both archives directly inside `<image_root>/COCO` (`data/COCO` by default). Do not leave them inside nested `train2014` and `val2014` directories, because each split stores filenames such as `COCO_train2014_000000421325.jpg`. The supplied splits do not reference COCO `test2015`, so that archive is unnecessary.

[Optional] The original question/annotation data are available separately if needed for preprocessing or comparison:

- [COCO-QA download page](https://www.cs.toronto.edu/~mren/research/imageqa/data/cocoqa/)
- [VQA v2 download page](https://visualqa.org/download.html)

#### VinVL features and Checkpoints

The official COCO and VQA downloads do **not** contain the VinVL features used by BPCIN. The preprocessed files and checkpoints can be downloaded here:

- [Preprocessed files](https://pan.baidu.com/s/1J9QaxPahpglaHDbnqs40QQ?pwd=avjr)

### Getting Started

#### Train

```bash
python main.py \
  --mode train \
  --device cuda:0 \
  --config ./config/5way_5shot_clip_coco_qa.py \
  --dataset_root ./data
```

#### Test

```bash
python main.py \
  --mode eval \
  --device cuda:0 \
  --config ./config/5way_5shot_clip_coco_qa.py \
  --dataset_root ./data \
  --checkpoint_dir ./checkpoints
```

