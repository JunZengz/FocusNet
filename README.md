# FocusNet: Transformer-enhanced Polyp Segmentation with Local and Pooling Attention

## Overview

## Create Environment
```
conda create -n FocusNet python==3.8.16
conda activate FocusNet
```

## Install Dependencies
```    
pip install -r requirements.txt
```

## Download Checkpoint 
Download pretrained checkpoints from [Google Drive](https://drive.google.com/file/d/18n1UdWEL31XN20hJDBqP0M5ccU4InnQT/view?usp=drive_link) and move it to the `pretrained_pth` directory.

## Download Dataset
Download PolypDB dataset from [this link](https://osf.io/pr7ms/files/osfstorage) and move it to the `data` directory.

## Train
```
python train_modality_wise.py 
```

## Test
```
python test_modality_wise.py 
```


## Citation
Please cite our paper if you find the work useful:
```
@article{zeng2025focusnet,
  title={FocusNet: Transformer-enhanced Polyp Segmentation with Local and Pooling Attention},
  author={Zeng, Jun and Santosh, KC and Nayak, Deepak Rajan and de Lange, Thomas and Varkey, Jonas and Berzin, Tyler and Jha, Debesh},
  journal={arXiv preprint arXiv:2504.13597},
  year={2025}
}
```

## Contact

Please contact zeng.cqupt@gamil.com for any further questions.
