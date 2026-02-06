# [MIA 2025] Diffusion-based arbitrary-scale magnetic resonance image super-resolution via progressive k-space reconstruction and denoising

Official code for ``Diffusion-based arbitrary-scale magnetic resonance image super-resolution via progressive k-space reconstruction and denoising. Medical Image Analysis, 2025.'' By Jiazhen Wang, Zhihao Shi, Xiang Gu, Yan Yang, Jian Sun.

## Requirements


- CUDA (if using GPU)
- Other dependencies listed in `requirements.txt`

## Clone

```bash
# Clone the repository
https://github.com/Jiazhen-Wang/PRDDiff-main.git
cd PRDDiff-main
```

## Dataset

### FastMRI

The dataset and dataset documentation are available for download from  [Website](https://fastmri.med.nyu.edu/).

### Clinical pediatric cerebral palsy dataset

The dataset (Infant-PWMl-CP.zip, 2.86GB) and dataset documentation are available for download at [Google Drive](https://drive.google.com/drive/folders/1yBVICW9lcDANth-RlwJy1C9M6QNXJ0L2?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1XiwKp7Ayc81qefs3eu7pGg?pwd=fae8).

## Usage

### Training

```bash
python train.py 
```

### Testing

The checkpoint are available for download at  [Baidu Netdisk](https://pan.baidu.com/s/1KTHCE6eW37dXGRRGqVq2pw?pwd=qkra). Please download and place the files into the `savemodel/` directory.

For single-stage super-resolution strategy:

```bash
python test_single.py 
```

For multi-stage super-resolution strategy:

```bash
python test_multi.py 
```

## Citation

If you use this codebase in your research, please cite:

```
@article{wang2025diffusion,
title={Diffusion-based arbitrary-scale magnetic resonance image super-resolution via progressive k-space reconstruction and denoising},
author={Wang, Jiazhen and Shi, Zhihao and Gu, Xiang and Yang, Yan and Sun, Jian},
journal={Medical Image Analysis},
pages={103814},
year={2025},
publisher={Elsevier}
}
```

## Contact

For questions or collaboration, please contact: jzwang@stu.xjtu.edu.cn

