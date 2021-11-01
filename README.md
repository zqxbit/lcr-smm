# LCR-SMM
LCR-SMM: Large Convergence Region Semantic Map Matching through Expectation Maximization

<img src="https://github.com/zqxbit/videos/blob/main/fig1-1223.png" width="500">

LCR-SMM is a large convergence region semantic map matching algorithm, with a transformation sampling strategy to reduce the initial error.

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

