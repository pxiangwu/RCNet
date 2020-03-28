## Code for ["Point cloud processing via recurrent set encoding"](https://arxiv.org/pdf/1911.10729.pdf).


## Requirements:
- PyTorch 0.3/0.4
- Python 3.6+
- CUDA 8.0 (Not sure if CUDA > 8.0 will work. This depends on PyTorch.)


## Usage and file structures:

- For classification, run `train_rcnet_cls.py`. For shape segmentation, run `run_seg.py`
- The dataset can be downloaded from https://github.com/charlesq34/pointnet:
```
# classification data
wget https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip

# segmentation data
wget https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip
```

- The dataloader for classification task is in file `utils/datasets.py`. The dataloader for segmentation task is in file `utils/part_dataset.py`.
- `utils/pointnet.py1` provides some modules, which are about spatial transformer (STN).  
- `utils/orderpoints.py` provides functions for partitioning the ambient space into structured beam. Currently the code is not the most efficient. For more efficient version, please refer to [here](https://github.com/pxiangwu/MotionNet/blob/master/data/data_utils.py#L105).
- `utils/provider` provides functions for some basic data augmentation, such at random jitter, random scale, etc.
- Directory `utils/gen_point_cloud` includes some codes (you may not need) for converting mesh into point clouds. It relies on pyntcloud library.


## Reference:
```
@inproceedings{wu2019point,
  title={Point cloud processing via recurrent set encoding},
  author={Wu, Pengxiang and Chen, Chao and Yi, Jingru and Metaxas, Dimitris},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  pages={5441--5449},
  year={2019}
}
```
