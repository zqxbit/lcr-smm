# LCR-SMM
LCR-SMM: Large Convergence Region Semantic Map Matching through Expectation Maximization

<img src="https://github.com/zqxbit/videos/blob/main/fig1-1223.png" width="500">

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
## Running the Demo
### Estimating the transformation
```bash
$ ./lcr-smm -s ../data/00_s_S.pcd -t ../data/00_s_T.pcd
```
```bash

```
### Displaying initial state
```bash

```

### Displaying matched maps
```bash

```


## Performing Semantic Map Matching
```bash
$ ./lcr-smm -s Path/to/Source/Map -t Path/to/Target/Map
```
