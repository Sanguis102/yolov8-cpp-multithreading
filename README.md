# RKNN YOLOv8 多线程推理

基于 RK3588 NPU 的高性能多线程 YOLOv8 推理框架，项目是[rknn-cpp-Multithreading](https://github.com/leafqycc/rknn-cpp-Multithreading)项目的YOLOv8实现，参考官方例程重新编写了后处理程序。


## 参考项目

本项目参考和使用了以下开源项目：

- **多线程架构**：[rknn-cpp-Multithreading](https://github.com/leafqycc/rknn-cpp-Multithreading-main) - YOLOv5 多线程推理框架
- **YOLOv8 实现**：[ultralytics_yolov8](https://github.com/airockchip/ultralytics_yolov8) - YOLOv8 RKNN 模型转换
- **官方例程**：[rknn_model_zoo/yolov8/cpp](https://github.com/airockchip/rknn_model_zoo/tree/main/examples/yolov8/cpp) - YOLOv8 单线程推理

##  项目特点

**运行特点**：
- 异步多线程 Pipeline 架构，提高 NPU 利用率
- 6 线程下 RK3588 可达 73 FPS，NPU 占用率 80%
- 支持视频文件和 USB 摄像头实时推理
- 简洁输出，只显示帧率，不刷屏

**使用技术**：
- C++14 模板编程（rknnPool 线程池）
- RGA 硬件加速预处理
- RKNN-API 1.5.0+ 异步推理
- YOLOv8 DFL 分布解码后处理
- 权重复用（`rknn_dup_context`）


