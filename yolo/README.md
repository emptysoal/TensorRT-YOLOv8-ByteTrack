# 封装 YOLOv8 TensorRT 推理

## 一. 项目简介

- 基于 `TensorRT-v8` ，运行`YOLOv8`推理；

- 支持嵌入式设备 `Jetson` 系列上部署，也可以在 `Linux x86_64`的服务器上部署；

本人所做的主要工作：

1. 参考 [tensorrtx](https://github.com/wang-xinyu/tensorrtx/tree/master/yolov8) 项目，模型 `.pth` -> `.engine`，提取出推理部分代码，并**封装为C++的类**，便于其他项目调用；
2. 预处理更换成了自己写的 CUDA编程预处理；
3. 后处理去掉了CUDA编程，因为测试其相比CPU后处理提速并不明显；
5. `YOLOv8` 推理编译为一个动态链接库，以解耦项目。

特点：

- 在其他项目中使用 `YOLOv8` 推理时，调用下面 3 行代码即可：

```C++
// 加载模型
std::string trtFile = "./engine/yolov8s.engine";
YoloDetecter detecter(trtFile);

// 使用TensorRT推理
std::vector<DetectResult> res = detecter.inference(img);
```

## 二. 环境配置

1. 基本要求：

- `TensorRT 8.0+`
- `OpenCV 3.4.0+`

2. 本人在 `Jetson Nano` 上的运行环境如下：

- 烧录的系统镜像为 `Jetpack 4.6.1`，该`jetpack` 原装环境如下：

| CUDA | cuDNN | TensorRT | OpenCV |
| ---- | ----- | -------- | ------ |
| 10.2 | 8.2   | 8.2.1    | 4.1.1  |

关于如何在 `Jetson nano` 上烧录镜像，网上资料还是很多的，这里就不赘述了，注意下载 `Jetpack`镜像时选择 4.6.1 版本，该版本对应的 TensorRT v8 版本

提示：无论何种设备，记得确认 `CMakeLists.txt` 文件中相关库的路径。

## 三. 模型转换

目的：把 `YOLOv8`的`pth`检测模型，转换成`TensorRT`的序列化文件，后缀 `.engine`

步骤：

1. 按照 [tensorrtx](https://github.com/wang-xinyu/tensorrtx/tree/master/yolov8) 项目操作，但作者亲测有以下注意点：
   - 拷贝`gen_wts.py`文件时，拷贝到 `YOLOv8`一级`ultralytics`目录下即可，且不需要安装`YOLOv8`，无需按其所写的二级目录；
   - 注意修改 `gen_wts.py` 文件中的输入输出目录。

2. 之后可成功得到 `yolov8s.engine` 文件（本人使用的是YOLOv8 s 模型，也可以使用其他的）

3. 在本项目中新建 `engine`目录，并放入转换后的模型文件

## 四. 运行项目

- 开始编译并运行
- 按如下步骤运行

```bash
mkdir build
cd build
cmake ..
make
cd ..
./main ./images  # 传入自己图像的目录
```

