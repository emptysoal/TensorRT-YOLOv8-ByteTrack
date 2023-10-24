# TensorRT C++ api deploy YOLOv8 + ByteTrack

## Introduction

- Based on `TensorRT-v8` , deploy `YOLOv8` + `ByteTrack` ;

-  Support `Jetson` series, also `Linux x86_64`;

Main work I have done:

1. Refer to [tensorrtx](https://github.com/wang-xinyu/tensorrtx/tree/master/yolov8) ，model:  `.pth` -> `.engine`,extract the inference part of the code, encapsulated into C++ classes, easy to call other projects ;
2. Preprocessing replaced with my own CUDA programming preprocessing;
3. Post-processing removed CUDA programming because it was not significantly faster in tests compared to CPU post-processing ;
4. The post-processed NMS greatly reduces conf_thres hyperparameters due to the principle of `ByteTrack` tracking, which is very important ;
5. `YOLOv8` inference compiles to a dynamic link library to decouple projects;
6. Reference official [ByteTrack TensorRT deploy](https://github.com/ifzhang/ByteTrack/tree/main/deploy/TensorRT/cpp) , modify its interface to the `YOLO` detector;
7. `ByteTrack` also compiles to a dynamic link library, further decoupling projects;
8. Add category filtering function, you can set the category you want to track in `main.cpp` line 8 .

## Effect

![](./assets/result_02.gif)

# Environment

1. Base requirements：

- `TensorRT 8.0+`
- `OpenCV 3.4.0+`

2. My running environment on `Jetson Nano` is as follows:

- The burned system image is `Jetpack 4.6.1`，original environment is as follows：

| CUDA | cuDNN | TensorRT | OpenCV |
| ---- | ----- | -------- | ------ |
| 10.2 | 8.2   | 8.2.1    | 4.1.1  |

- Install Eigen

```bash
apt install libeigen3-dev
```

## Model conversion

Convert the `pth` model of `YOLOv8` into a serialized file of `TensorRT` 

steps：

1. refer to [tensorrtx](https://github.com/wang-xinyu/tensorrtx/tree/master/yolov8) , however, I have the following points of attention：
   - When copy file `gen_wts.py`，copy to `YOLOv8` Level 1 `ultralytics` directory, there is no need to install `YOLOv8`
   - Note the modification of the input/output directory in the `gen wts.py` file.
2. The `yolov8s.engine` file can then be successfully obtained
3. Create a new `engine` directory in the  `yolo` directory and put the converted model file in.

## Run tracking

- Follow these steps

```bash
mkdir build
cd build
cmake ..
make
./main ../videos/demo.mp4  # The path to your own video
```

Then the `result.mp4` will be in the build directory, is to track the effect of the video file 

If you want the tracked video to play in real time, you can uncomment line 94 of main.cpp. 

# Reference

- [tensorrtx](https://github.com/wang-xinyu/tensorrtx/tree/master/yolov8)

- [ByteTrack](https://github.com/ifzhang/ByteTrack)

