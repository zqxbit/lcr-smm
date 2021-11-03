# LCR-SMM
LCR-SMM: Large Convergence Region Semantic Map Matching through Expectation Maximization

<img src="https://github.com/zqxbit/videos/blob/main/fig1-1223.png" width="375">

LCR-SMM is a large convergence region semantic map matching algorithm, with a transformation sampling strategy to reduce the initial error.

## Dependencies
- PCL
- Eigen
- Sophus
- CERES

## Installation
```bash
$ git clone https://github.com/zqxbit/lcr-smm
$ cd lcr-smm
$ mkdir build && cd build
$ cmake ../
$ make
```
## Performing Semantic Map Matching
### Running the demo
#### Estimating the transformation
```bash
$ ./lcr-smm -s ../data/00_s_S.pcd -t ../data/00_s_T.pcd
```
```bash
Estimated Transformation:
    0.865958     0.500116 -0.000395343     -4.00093
   -0.500116     0.865958    0.0006292      -6.9226
 0.000657024 -0.000347144            1    0.0161345
           0            0            0            1
```
#### Displaying initial state
```bash
pcl_viewer -bc 255,255,255 init.pcd
```

<img src="https://github.com/zqxbit/videos/blob/main/00_init1102.png" width="500">

#### Displaying matched maps
```bash
pcl_viewer -bc 255,255,255 LCR.pcd
```

<img src="https://github.com/zqxbit/videos/blob/main/00_LCR1102.png" width="500">


### Performing Semantic Map Matching
```bash
$ ./lcr-smm -s Path/to/Source/Map -t Path/to/Target/Map
```

## Citation
If you find our work useful in your research, please consider citing:

```bash
@article{2021_LCR_SMM,
title={LCR-SMM: Large Convergence Region Semantic Map Matching through Expectation Maximization},
author={Zhang, Qingxiang and Wang, Meiling and Yue, Yufeng and Liu, Tong},
journal={IEEE/ASME Transactions on Mechatronics},
pages={1--11},
doi = {10.1109/TMECH.2021.3124994},
publisher={IEEE}
}
```
