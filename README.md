# APD-MVS
## About

APD-MVS is an MVS method that adopts adaptive patch deformation and an NCC-based matching metric. 

Our paper was accepted by CVPR 2023!

If you find this project useful for your research, please cite:  

```
@InProceedings{Wang_2023_CVPR,
    author    = {Wang, Yuesong and Zeng, Zhaojie and Guan, Tao and Yang, Wei and Chen, Zhuo and Liu, Wenkai and Xu, Luoyuan and Luo, Yawei},
    title     = {Adaptive Patch Deformation for Textureless-Resilient Multi-View Stereo},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {1621-1630}
}
```
## Dependencies

The code has been tested on Ubuntu 18.04 with Nvidia Titan RTX, and you can modify the CMakeList.txt to compile on Windows.
* [Cuda](https://developer.nvidia.cn/zh-cn/cuda-toolkit) >= 10.2
* [OpenCV](https://opencv.org/) >= 3.3.0
* [Boost](https://www.boost.org/) >= 1.62.0
* [cmake](https://cmake.org/) >= 2.8

**Besides make sure that your [GPU Compute Capability](https://en.wikipedia.org/wiki/CUDA) matches the CMakeList.txt!!!** Otherwise you won't get the depth results! For example, according to [GPU Compute Capability](https://en.wikipedia.org/wiki/CUDA), RTX3080's Compute Capability is 8.6. So you should set the 
cuda compilation parameter 'arch=compute_86,code=sm_86' or add a '-gencode arch=compute_86,code=sm_86'.

## Usage
### Compile APD-MVS

```  sh
git clone https://github.com/whoiszzj/APD-MVS.git
cd APD-MVS
mkdir build & cd build
cmake ..
make
```
### Prepare Datasets

#### ETH Dataset

You may download [train](https://www.eth3d.net/data/multi_view_training_dslr_undistorted.7z) and [test](https://www.eth3d.net/data/multi_view_test_dslr_undistorted.7z) dataset from ETH3D, and use the script [*colmap2mvsnet.py*](./colmap2mvsnet.py) to convert the dataset format(you may refer to [MVSNet](https://github.com/YoYo000/MVSNet#file-formats)). You can use the "scale" option in the script to generate any resolution you need.

```python
python colmap2mvsnet.py --dense_folder <ETH3D data path, such as ./ETH3D/office> --save_folder <The path to save> --scale_factor 2 # half resolution
```

#### Tanks & Temples Dataset

We use the version provided by MVSNet. The dataset can be downloaded from [here](https://drive.google.com/file/d/1YArOJaX9WVLJh4757uE8AEREYkgszrCo/view), and the format is exactly what we need.

#### Other Dataset

Such as DTU and BlenderMVS, you may explore them yourself. !!! But remember to modify the [ReadCamera](https://github.com/whoiszzj/APD-MVS/blob/d9f9731235f4db05712024213e32346b6a01f5d6/APD.cpp#L84) function when you test on DTU !!!

### Run

After you prepare the dataset, and you want to run the test for ETH3D/office, you can follow this command line.

```bash
./APD <ETH3D root path>/office
```

The result will be saved in the folder office/APD, and the point cloud is saved as  "APD.ply"

It is very easy to use, and you can modify our code as you need.

## Acknowledgements

This code largely benefits from the following repositories: [ACMM](https://github.com/GhiXu/ACMM.git), [Colmap](https://github.com/colmap/colmap.git). Thanks to their authors for opening the source of their excellent works.

If you have any question or find some bugs, please leave it in [Issues](https://github.com/whoiszzj/APD-MVS/issues)! Thank you!
