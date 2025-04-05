#pragma once
#ifndef QUICK_H
#define QUICK_H

#include <opencv2/opencv.hpp>

using namespace cv;

class QuickDemo {
public:
    // 图像的读取与保存
    // 创建灰度图像，创建hsv格式图像
    void colorSpace_Demo(cv::Mat &image);
    // 新增方法：矩阵创建
    void mat_create_Demo(cv::Mat &image);
    // 通过下标访问像素
    void pixel_visit_Demo(Mat &image);
    // 通过指针访问像素(z这是最快的)
    void pixel_visiter_Demo(Mat &image);
    //像素统计
    void pixel_statistic_Demo(Mat &image);
    //对图片的运算
    void operators_Demo(Mat &image);
    //逻辑运算
    void bitwise_Demo(Mat &image);
    //图像的加权混合
    void mix_image_Demo(Mat &image1, Mat &image2);
    //对比度增强
    void image_contrast_Demo(Mat &image1);
    //通道的拆分 与 合并
    void channel_Demo(Mat &image);
};

#endif