AerialFormer: Multi-resolution Transformer for Aerial Image Segmentation
=====

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/aerialformer-multi-resolution-transformer-for/semantic-segmentation-on-isaid)](https://paperswithcode.com/sota/semantic-segmentation-on-isaid?p=aerialformer-multi-resolution-transformer-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/aerialformer-multi-resolution-transformer-for/semantic-segmentation-on-isprs-potsdam)](https://paperswithcode.com/sota/semantic-segmentation-on-isprs-potsdam?p=aerialformer-multi-resolution-transformer-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/aerialformer-multi-resolution-transformer-for/semantic-segmentation-on-loveda)](https://paperswithcode.com/sota/semantic-segmentation-on-loveda?p=aerialformer-multi-resolution-transformer-for)

[[`arXiv`](https://arxiv.org/abs/2306.06842)]
[[`pdf`](https://arxiv.org/pdf/2306.06842.pdf)]

Aerial Image Segmentation is a top-down perspective semantic segmentation and has several challenging characteristics such as strong imbalance in the foreground-background distribution, complex background, intra-class heterogeneity, inter-class homogeneity, and small objects. To handle these problems, we inherit the advantages of Transformers and propose AerialFormer, which unifies Transformers at the contracting path with lightweight Multi-Dilated Convolutional Neural Networks (MD-CNNs) at the expanding path. AerialFormer is designed as a hierarchical structure, in which Transformer encoder outputs multi-scale features and MD-CNNs decoder aggregates information from the multi-scales. Thus, it takes both local and global context into consideration to render powerful representations and high-resolution segmentation. We have benchmarked AerialFormer on three common datasets including iSAID, LoveDA, and Postdam. Comprehensive experiments and extensive ablation studies show that our proposed AerialFormer outperforms previous state-of-the-art methods with remarkable performance. Our source code will be publicly available upon acceptance. 

![](assets/visualization.png)

# Introduction
Our code is implemented on [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) and its update is rapid. Please keep in mind you're using the same/compatibility version. Please refer to [get_started](https://github.com/open-mmlab/mmsegmentation/blob/0.x/docs/en/get_started.md) for installation and [dataset_prepare](https://github.com/open-mmlab/mmsegmentation/blob/0.x/docs/en/get_started.md) for dataset preparation on mmsegmentation. However, __NOT__ all of codes are the same (e.g. Potsdam dataset)


# Data preparation

Since some datasets don't allow to redistribute them, You need to get prepared the zip files. Please check [mmsegmentation/dataset_prepare](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#prepare-datasets) to get zip files.

After that, please run the following commands to prepare for datasets(iSAID, LoveDA, Potsdam)

<details open>
<summary><span style="font-size: 1.5em;">iSAID</span></summary>

Download the original images from [DOTA](https://captain-whu.github.io/DOTA/index.html) and annotations from [iSAID](https://captain-whu.github.io/iSAID/dataset.html).
Put your dataset source file in one directory. For more details, check [iSAID DevKit](https://github.com/CAPTAIN-WHU/iSAID_Devkit).

```bash
python tools/convert_datasets/isaid.py /path/to/potsdam
```
</details>

<details open>
<summary><span style="font-size: 1.5em;">Potsdam</span></summary>

For [Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx) dataset, please run the following command to re-organize the dataset. Put your dataset source file in one directory. We used '2_Ortho_RGB.zip' and '5_Labels_all_noBoundary.zip'.

- With Clutter. Number of class is __6__ classes.
```bash
python tools/convert_datasets/potsdam.py /path/to/potsdam
```

- Without Clutter. Number of class is __5__ classes.
```bash
python tools/convert_datasets/potsdam_no_clutter.py /path/to/potsdam
```

> Note that we changed some settings from the original [convert_dataset code](https://github.com/open-mmlab/mmsegmentation/blob/main/tools/dataset_converters/potsdam.py) from mmsegmentation.

</details>

<details open>
<summary><span style="font-size: 1.5em;">LoveDA</span></summary>

Download the dataset from Google Drive [here](https://drive.google.com/drive/folders/1ibYV0qwn4yuuh068Rnc-w4tPi0U0c-ti).
For LoveDA dataset, please run the following command to re-organize the dataset.
```bash
python tools/dataset_converters/loveda.py /path/to/loveDA
```
More details about LoveDA can be found [here](https://github.com/Junjue-Wang/LoveDA).

</details>
<br>

# Get Started (singularity/non-singularity)
We use `mmcv-full=="1.7.1` and `mmsegmentation==0.30.0`. Please follow the other dependencies to [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/blob/0.x/docs/en/get_started.md).

## Singularity Option
If you've not installed it, please refer to [AICV](https://github.com/UARK-AICV/UARK-AICV.github.io/wiki/Server-setup#singularity) to install singularity.

<details open>
<summary><span style="font-size: 1.5em;">Environment Setup</span></summary>

Build Image from `docker/Dockerfile`
```bash
export REGISTRY_NAME="user"
export IMAGE_NAME="aerialformer"
docker build -t $REGISTRY_NAME/$IMAGE_NAME docker/ # You can use 'thanyu/aerialformer'
```
</details>

<details open>
<summary><span style="font-size: 1.5em;">Training</span></summary>

- Single GPU

```bash
export DATAPATH="path/to/data" #If you do not specify, it'll be "$PWD/data"
bash tools/singularity_train.sh configs/path/to/config
```

For example, to run AerialFormer-T on iSAID dataset:

```bash
bash tools/singularity_train.sh configs/aerialformer/aerialformer_tiny_512x512_loveda.py
```
    
- Multi GPUs

```bash 
export DATAPATH="path/to/data" #If you do not specify, it'll be "$PWD/data"
bash tools/singularity_train.sh configs/path/to/config
```

For example, to train AerialFormer-S on LoveDA dataset:


```bash
bash tools/singularity_dist_train.sh configs/aerialformer/aerialformer_small_512x512_loveda.py 2
```

</details> 

<details>
<summary><span style="font-size: 1.5em;">Evaluation</span></summary>

- Single GPU

```bash
bash tools/singularity_test.sh configs/path/to/config work_dirs/path/to/trained_weight --eval metrics
```

For example, to test AerialFormer-T on Loveda dataset

```bash
bash tools/singularity_test.sh configs/aerialformer/aerialformer_tiny_512x512_loveda.py work_dirs/aerialformer_tiny_512x512_loveda/2023_0101_0000/latest.pth --eval mIoU
```

- Multi GPUs
```bash
bash tools/singularity_dist_test.sh configs/path/to/config work_dirs/work_dirs/path/to/trained_weight 2 --eval metrics
```

For example, to test AerialFormer-S on Loveda dataset

```bash
bash tools/singularity_dist_test.sh work_dirs/aerialformer_small_512x512_loveda/2023_0612_1009/aerialformer_small_512x512_loveda.py work_dirs/aerialformer_small_512x512_loveda/2023_0612_1009/latest.pth 2 --eval mIoU
```

</details>

---

## Non-singularity Option

<details open>
<summary><span style="font-size: 1.5em;">Environment Setup</span></summary>

__STEP 1.__ Run and install mmsegmentation by the following code. 
> For more information, refer to [mmsegmentaiton/get_started](https://github.com/open-mmlab/mmsegmentation/blob/0.x/docs/en/get_started.md). 

```
pip install -U openmim && mim install mmcv-full=="1.7.1"
pip install mmsegmentation==0.30.0
```

__STEP 2.__ Clone this repository and install.
```base
git clone https://github.com/UARK-AICV/AerialFormer.git
cd AerialFormer
pip install -v -e .
```
</details>

<details open>
<summary><span style="font-size: 1.5em;">Training</span></summary>

- Single GPU

```bash
python tools/train.py configs/path/to/config
```

For example, to train AerialFormer-T on LoveDA dataset:

```bash
python tools/train.py configs/aerialformer/aerialformer_tiny_512x512_loveda.py
```

- Multi GPUs

```bash
bash tools/dist_train.sh configs/path/to/config num_gpus
```

For example, to train AerialFormer-B on LoveDA dataset on two gpus:

```bash
bash tools/dist_train.sh configs/aerialformer/aerialformer_base_512x512_loveda.py 2
```

__Note batch size matters.__ We're using 8 batch sizes.

</details>

<details>
<summary><span style="font-size: 1.5em;">Evaluation</span></summary>

- Single GPU
```bash
python tools/test.py configs/path/to/config work_dirs/path/to/checkpoint --eval metrics
```

For example , to test AerialFormer-T on Loveda dataset
```bash
python tools/test.py work_dirs/aerialformer_tiny_512x512_loveda/2023_0101_0000/aerialformer_tiny_512x512_loveda.py work_dirs/aerialformer_tiny_512x512_loveda/2023_0101_0000/latest.pth --eval mIoU
```

- Multi GPUs
```bash
bash tools/dist_test.py configs/path/to/config work_dirs/path/to/checkpoint num_gpus --eval metrics
```

For example , to test AerialFormer-T on Loveda dataset
```bash
bash tools/dist_test.py work_dirs/aerialformer_tiny_512x512_loveda/2023_0101_0000/aerialformer_tiny_512x512_loveda.py work_dirs/aerialformer_tiny_512x512_loveda/2023_0101_0000/latest.pth 2 --eval mIoU
```

</details>

<br>

# Acknowledgement
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)

# Citation
