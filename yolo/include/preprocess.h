#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

void preprocess(const cv::Mat& srcImg, float* dstData, const int dstHeight, const int dstWidth);
/*
srcImg:    source image for inference
dstData:   data after preprocess (resize / bgr to rgb / hwc to chw / normalize)
dstHeight: CNN input height
dstWidth:  CNN input width
*/

#endif  // PREPROCESS_H
