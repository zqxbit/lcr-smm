# LCR-SMM
<img src="https://github.com/zqxbit/videos/blob/main/multi-robot0707.png" width="750">
LCR-SMM is a novel large convergence region semantic map matching algorithm.

## Dependencies
- PCL
- Eigen
- Sophus
- CERES

## Compiling
```bash
$ mkdir build && cd build
$ cmake ../
$ make
```
## Running 
```bash
$ ./lcr-smm -s ../data/00_s_S.pcd -t ../data/00_s_T.pcd
```

