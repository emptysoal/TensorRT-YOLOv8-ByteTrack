#include "postprocess.h"


cv::Rect get_rect(cv::Mat &img, float bbox[4]) {
    float l, r, t, b;
    float r_w = kInputW / (img.cols * 1.0);
    float r_h = kInputH / (img.rows * 1.0);

    if (r_h > r_w) {
        l = bbox[0];
        r = bbox[2];
        t = bbox[1] - (kInputH - r_w * img.rows) / 2;
        b = bbox[3] - (kInputH - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - (kInputW - r_h * img.cols) / 2;
        r = bbox[2] - (kInputW - r_h * img.cols) / 2;
        t = bbox[1];
        b = bbox[3];
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(round(l), round(t), round(r - l), round(b - t));
}

static float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
            (std::max)(lbox[0], rbox[0]), //left
            (std::min)(lbox[2], rbox[2]), //right
            (std::max)(lbox[1], rbox[1]), //top
            (std::min)(lbox[3], rbox[3]), //bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
    float unionBoxS = (lbox[2] - lbox[0]) * (lbox[3] - lbox[1]) + (rbox[2] - rbox[0]) * (rbox[3] - rbox[1]) - interBoxS;
    return interBoxS / unionBoxS;
}

static bool cmp(const Detection &a, const Detection &b) {
    return a.conf > b.conf;
}

void nms(std::vector<Detection> &res, float *output, float conf_thresh, float nms_thresh) {
    int det_size = sizeof(Detection) / sizeof(float);
    std::map<float, std::vector<Detection>> m;

    for (int i = 0; i < output[0]; i++) {
        if (output[1 + det_size * i + 4] <= conf_thresh) continue;
        Detection det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Detection>());
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) {
        auto &dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto &item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    dets.erase(dets.begin() + n);
                    --n;
                }
            }
        }
    }
}
