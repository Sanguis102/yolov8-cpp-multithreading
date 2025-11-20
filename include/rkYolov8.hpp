#ifndef RKYOLOV8_H
#define RKYOLOV8_H

#include "rknn_api.h"
#include "opencv2/core/core.hpp"
#include "postprocess.h"

static void dump_tensor_attr(rknn_tensor_attr *attr);
static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz);
static unsigned char *load_model(const char *filename, int *model_size);

class rkYolov8
{
private:
    int ret;
    std::mutex mtx;
    std::string model_path;
    unsigned char *model_data;

    rknn_app_context_t app_ctx;
    letterbox_t letter_box;

    int img_width, img_height;
    float nms_threshold, box_conf_threshold;

public:
    rkYolov8(const std::string &model_path);
    int init(rknn_context *ctx_in, bool share_weight);
    rknn_context *get_pctx();
    cv::Mat infer(cv::Mat &orig_img);
    ~rkYolov8();
};

#endif
