#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mutex>
#include "rknn_api.h"
#include "postprocess.h"
#include "preprocess.h"
#include "rkYolov8.hpp"
#include "coreNum.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    std::string shape_str = attr->n_dims < 1 ? "" : std::to_string(attr->dims[0]);
    for (int i = 1; i < attr->n_dims; ++i)
    {
        shape_str += ", " + std::to_string(attr->dims[i]);
    }
}

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
    unsigned char *data;
    int ret;

    data = NULL;

    if (NULL == fp)
    {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0)
    {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char *)malloc(sz);
    if (data == NULL)
    {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");
    if (NULL == fp)
    {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}

rkYolov8::rkYolov8(const std::string &model_path)
{
    this->model_path = model_path;
    nms_threshold = NMS_THRESH;
    box_conf_threshold = BOX_THRESH;
    memset(&app_ctx, 0, sizeof(rknn_app_context_t));
    memset(&letter_box, 0, sizeof(letterbox_t));
}

int rkYolov8::init(rknn_context *ctx_in, bool share_weight)
{
    printf("Loading YOLOv8 model...\n");

    int model_data_size = 0;
    model_data = load_model(model_path.c_str(), &model_data_size);
    if (model_data == NULL)
    {
        printf("load_model error!\n");
        return -1;
    }

    int ret = 0;
    if (share_weight == true)
        ret = rknn_dup_context(ctx_in, &app_ctx.rknn_ctx);
    else
        ret = rknn_init(&app_ctx.rknn_ctx, model_data, model_data_size, 0, NULL);

    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }

    ret = rknn_query(app_ctx.rknn_ctx, RKNN_QUERY_IN_OUT_NUM, &app_ctx.io_num, sizeof(app_ctx.io_num));
    if (ret < 0)
    {
        printf("rknn_query io_num error ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", app_ctx.io_num.n_input, app_ctx.io_num.n_output);

    app_ctx.input_attrs = (rknn_tensor_attr *)calloc(app_ctx.io_num.n_input, sizeof(rknn_tensor_attr));
    for (int i = 0; i < app_ctx.io_num.n_input; i++)
    {
        app_ctx.input_attrs[i].index = i;
        ret = rknn_query(app_ctx.rknn_ctx, RKNN_QUERY_INPUT_ATTR, &(app_ctx.input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_query input_attrs error ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(app_ctx.input_attrs[i]));
    }

    app_ctx.output_attrs = (rknn_tensor_attr *)calloc(app_ctx.io_num.n_output, sizeof(rknn_tensor_attr));
    for (int i = 0; i < app_ctx.io_num.n_output; i++)
    {
        app_ctx.output_attrs[i].index = i;
        ret = rknn_query(app_ctx.rknn_ctx, RKNN_QUERY_OUTPUT_ATTR, &(app_ctx.output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_query output_attrs error ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(app_ctx.output_attrs[i]));
    }

    if (app_ctx.output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC &&
        app_ctx.output_attrs[0].type == RKNN_TENSOR_INT8)
    {
        app_ctx.is_quant = true;
    }
    else
    {
        app_ctx.is_quant = false;
    }

    if (app_ctx.input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("model is NCHW input fmt\n");
        app_ctx.model_channel = app_ctx.input_attrs[0].dims[1];
        app_ctx.model_height = app_ctx.input_attrs[0].dims[2];
        app_ctx.model_width = app_ctx.input_attrs[0].dims[3];
    }
    else
    {
        printf("model is NHWC input fmt\n");
        app_ctx.model_height = app_ctx.input_attrs[0].dims[1];
        app_ctx.model_width = app_ctx.input_attrs[0].dims[2];
        app_ctx.model_channel = app_ctx.input_attrs[0].dims[3];
    }
    printf("model input height=%d, width=%d, channel=%d\n",
           app_ctx.model_height, app_ctx.model_width, app_ctx.model_channel);

    // 设置模型绑定的核心/Set the core of the model that needs to be bound
    rknn_core_mask core_mask;
    switch (get_core_num())
    {
    case 0:
        core_mask = RKNN_NPU_CORE_0;
        break;
    case 1:
        core_mask = RKNN_NPU_CORE_1;
        break;
    case 2:
        core_mask = RKNN_NPU_CORE_2;
        break;
    default:
        core_mask = RKNN_NPU_CORE_0;
        break;
    }
    ret = rknn_set_core_mask(app_ctx.rknn_ctx, core_mask);
    if (ret < 0)
    {
        printf("rknn_set_core_mask error ret=%d (core_mask=%d)\n", ret, core_mask);
    }
    else
    {
        int core_id = (core_mask == RKNN_NPU_CORE_0) ? 0 : (core_mask == RKNN_NPU_CORE_1 ? 1 : 2);
        printf("Model bound to NPU Core %d successfully\n", core_id);
    }

    // 初始化后处理（加载标签文件）
    int ret_post = init_post_process();
    if (ret_post < 0)
    {
        printf("Warning: init_post_process failed, labels will not be loaded!\n");
    }
    else
    {
        printf("Post process init success!\n");
    }

    return 0;
}

rknn_context *rkYolov8::get_pctx()
{
    return &app_ctx.rknn_ctx;
}

cv::Mat rkYolov8::infer(cv::Mat &orig_img)
{
    std::lock_guard<std::mutex> lock(mtx);

    img_width = orig_img.cols;
    img_height = orig_img.rows;

    cv::Mat img;
    cv::cvtColor(orig_img, img, cv::COLOR_BGR2RGB);

    cv::Mat resized_img(app_ctx.model_height, app_ctx.model_width, CV_8UC3);

    // RGA硬件加速预处理
    rga_buffer_t src;
    rga_buffer_t dst;
    memset(&src, 0, sizeof(src));
    memset(&dst, 0, sizeof(dst));

    float scale_w = (float)app_ctx.model_width / img.cols;
    float scale_h = (float)app_ctx.model_height / img.rows;
    float min_scale = std::min(scale_w, scale_h);

    letter_box.scale = min_scale;
    letter_box.x_pad = 0;
    letter_box.y_pad = 0;

    int new_w = round(img.cols * min_scale);
    int new_h = round(img.rows * min_scale);

    // 使用RGA硬件加速进行resize
    int ret_rga = resize_rga(src, dst, img, resized_img, cv::Size(new_w, new_h));
    if (ret_rga != 0)
    {
        fprintf(stderr, "resize with rga error, fallback to opencv\n");
        // RGA失败时回退到OpenCV
        cv::resize(img, resized_img, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    }

    int pad_w = app_ctx.model_width - new_w;
    int pad_h = app_ctx.model_height - new_h;
    letter_box.x_pad = pad_w / 2;
    letter_box.y_pad = pad_h / 2;

    // 填充边框
    cv::copyMakeBorder(resized_img, resized_img, letter_box.y_pad, pad_h - letter_box.y_pad,
                       letter_box.x_pad, pad_w - letter_box.x_pad,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = app_ctx.model_width * app_ctx.model_height * app_ctx.model_channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = resized_img.data;

    int ret = rknn_inputs_set(app_ctx.rknn_ctx, app_ctx.io_num.n_input, inputs);
    if (ret < 0)
    {
        printf("rknn_inputs_set error ret=%d\n", ret);
        return orig_img;
    }

    ret = rknn_run(app_ctx.rknn_ctx, nullptr);
    if (ret < 0)
    {
        printf("rknn_run error ret=%d\n", ret);
        return orig_img;
    }

    rknn_output outputs[app_ctx.io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < app_ctx.io_num.n_output; i++)
    {
        outputs[i].index = i;
        outputs[i].want_float = (!app_ctx.is_quant);
    }

    ret = rknn_outputs_get(app_ctx.rknn_ctx, app_ctx.io_num.n_output, outputs, NULL);
    if (ret < 0)
    {
        printf("rknn_outputs_get error ret=%d\n", ret);
        return orig_img;
    }

    object_detect_result_list od_results;
    memset(&od_results, 0, sizeof(object_detect_result_list));

    post_process(&app_ctx, (void *)outputs, &letter_box, box_conf_threshold, nms_threshold, &od_results);

    char text[256];
    for (int i = 0; i < od_results.count; i++)
    {
        object_detect_result *det_result = &(od_results.results[i]);

        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;

        rectangle(orig_img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 3);

        sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
        // 注释掉检测目标打印，避免刷屏
        // printf("%s @ (%d %d %d %d) %.3f\n", coco_cls_to_name(det_result->cls_id),
        //        det_result->box.left, det_result->box.top,
        //        det_result->box.right, det_result->box.bottom, det_result->prop);

        int baseline;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        int label_y = y1 - label_size.height - baseline;
        if (label_y < 0) label_y = 0;

        cv::rectangle(orig_img, cv::Rect(cv::Point(x1, label_y), cv::Size(label_size.width, label_size.height + baseline)),
                      cv::Scalar(0, 255, 0), -1);

        cv::putText(orig_img, text, cv::Point(x1, label_y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }

    rknn_outputs_release(app_ctx.rknn_ctx, app_ctx.io_num.n_output, outputs);

    // 如果检测到目标，打印一些调试信息（可选）
    // 默认不打印，避免刷屏
    // 如需调试，可以取消下面的注释
    /*
    if (od_results.count > 0)
    {
        printf("Detection: %d objects found in frame\n", od_results.count);
    }
    */

    return orig_img;
}

rkYolov8::~rkYolov8()
{
    deinit_post_process();

    if (app_ctx.input_attrs != NULL)
    {
        free(app_ctx.input_attrs);
        app_ctx.input_attrs = NULL;
    }
    if (app_ctx.output_attrs != NULL)
    {
        free(app_ctx.output_attrs);
        app_ctx.output_attrs = NULL;
    }

    if (app_ctx.rknn_ctx != 0)
    {
        rknn_destroy(app_ctx.rknn_ctx);
        app_ctx.rknn_ctx = 0;
    }

    if (model_data)
    {
        free(model_data);
        model_data = NULL;
    }
}
