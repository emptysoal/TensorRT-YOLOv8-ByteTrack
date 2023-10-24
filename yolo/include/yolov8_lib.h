#ifndef YOLOV8_LIB
#define YOLOV8_LIB

#include <opencv2/opencv.hpp>
#include "public.h"
#include "yololayer.h"

using namespace nvinfer1;


struct DetectResult
{
    cv::Rect tlwh;  // top left width height
    float conf;
    int class_id;
};


class YoloDetecter
{
public:
    YoloDetecter(const std::string trtFile);
    ~YoloDetecter();
    std::vector<DetectResult> inference(cv::Mat& img);

private:
    void deserialize_engine();
    void inference();

private:
    Logger              gLogger;
    std::string         trtFile_;

    ICudaEngine *       engine;
    IRuntime *          runtime;
    IExecutionContext * context;

    cudaStream_t        stream;

    int                 kOutputSize;
    std::vector<int>    vTensorSize;  // bytes of input and output
    float *             inputData;
    float *             outputData;
    std::vector<void *> vBufferD;
};

#endif  // YOLOV8_LIB
