# TensorRT C++ api 部署 YOLOv8 + ByteTrack

## 一. 项目简介

- 基于 `TensorRT-v8` ，部署`YOLOv8` + `ByteTrack` 的目标跟踪；

- 支持 `Jetson` 系列嵌入式设备上部署，也可以在 `Linux x86_64`的服务器上部署；

本人所做的主要工作：

1. 参考 [tensorrtx](https://github.com/wang-xinyu/tensorrtx/tree/master/yolov8) 项目，模型 `.pth` -> `.engine`，提取出**推理部分代码，封装为C++的类**，便于其他项目调用；
2. 预处理更换成了自己写的 CUDA编程预处理；
3. 后处理去掉了CUDA编程，因为测试其相比CPU后处理提速并不明显；
4. 后处理的 `NMS` **大幅减小`conf_thres`超参数**，源于 `ByteTrack` 跟踪的原理，这一点**非常重要**；
5. `YOLOv8` 推理编译为一个动态链接库，以解耦项目；
6. 参考官方 [ByteTrack TensorRT部署](https://github.com/ifzhang/ByteTrack/tree/main/deploy/TensorRT/cpp)，修改其与YOLO检测器的接口；
7. `ByteTrack` 也编译为一个动态链接库，进一步解耦项目；
8. 增加类别过滤功能，可以在`main.cpp`第 8 行设置自己想要跟踪的类别。

## 二. 项目效果

![](./assets/effect.gif)

## 三. 环境配置

1. 基本要求：

- `TensorRT 8.0+`
- `OpenCV 3.4.0+`

2. 本人在 `Jetson Nano` 上的运行环境如下：

- 烧录的系统镜像为 `Jetpack 4.6.1`，该`jetpack` 原装环境如下：

| CUDA | cuDNN | TensorRT | OpenCV |
| ---- | ----- | -------- | ------ |
| 10.2 | 8.2   | 8.2.1    | 4.1.1  |

关于如何在 `Jetson nano` 上烧录镜像，网上资料还是很多的，这里就不赘述了，注意下载 `Jetpack`镜像时选择 4.6.1 版本，该版本对应的 TensorRT v8 版本

- 安装`Eigen`库

```bash
apt install libeigen3-dev
```

3. 如果是服务器上，保证基本环境版本满足，再安装`Eigen`库即可

提示：无论何种设备，记得确认 `CMakeLists.txt` 文件中相关库的路径。

## 四. 模型转换

目的：得到`TensorRT`的序列化文件，后缀 `.engine`

- 首先获取 `wts` 格式的模型文件，链接：[yolov8s.wts](https://pan.baidu.com/s/16d_MqVlUxnjOhLxVyjQy8w)，提取码：gsqm

- 然后按以下步骤执行：

```bash
cd {TensorRT-YOLOv8-ByteTrack}/tensorrtx-yolov8/
mkdir build
cd build
cp {path/to/yolov8s.wts} .
cmake ..
make
./yolov8 -s yolov8s.wts yolov8s.engine s

cd ../../
mkdir yolo/engine
cp tensorrtx-yolov8/build/yolov8s.engine yolo/engine
```

## 五. 运行项目

- 开始编译并运行目标跟踪的代码
- 按如下步骤运行

```bash
mkdir build
cd build
cmake ..
make
./main ../videos/demo.mp4  # 传入自己视频的路径
```

之后会在 `build` 目录下得到`result.mp4`，为跟踪效果的视频文件

如果想要跟踪的视频实时播放，可解开`main.cpp`第 94 行的注释。

## 六. 项目参考

主要参考了下面的项目：

- [tensorrtx](https://github.com/wang-xinyu/tensorrtx/tree/master/yolov8)

- [ByteTrack](https://github.com/ifzhang/ByteTrack)

