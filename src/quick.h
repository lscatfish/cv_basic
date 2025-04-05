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
    void operators_Demo(Mat& image);
};

#endif