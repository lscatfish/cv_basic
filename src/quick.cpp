#include "quick.h"

using namespace std;

//图像的读取与保存
//创建灰度图像，创建hsv格式图像
void QuickDemo::colorSpace_Demo(cv::Mat &image) {
    cv::Mat gray, hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    cv::namedWindow("hsv", cv::WINDOW_NORMAL);
    cv::namedWindow("gray", cv::WINDOW_NORMAL);
    cv::imshow("hsv", hsv);
    cv::imshow("gray", gray);
    cv::imwrite("../img/output/colorSpace_hsv.png", hsv);
    cv::imwrite("../img/output/colorSpace_gray.png", gray);
}

// 新增方法：矩阵创建
void QuickDemo::mat_create_Demo(cv::Mat &image) {
    cv::Mat m1, m2;
    // Mat::ones 是 OpenCV 中的一个静态方法，用于创建一个所有元素值都为 1 的矩阵
    //(如果是多个通道则只有channel1（蓝色）通道是1)
    /*
    Size(4, 4)：
    Size 是一个 OpenCV 中的结构体，用于表示图像的尺寸。
    Size(4, 4) 表示创建一个宽度为 4、高度为 4 的矩阵。*/
    /*
    CV_8UC3：
    CV_8UC3 是一个宏，表示矩阵的数据类型。
    CV_8UC3 表示矩阵的每个元素是一个 3 通道的 8 位无符号整数
    （即每个通道占用 8 位，总共有 3 个通道，通常用于彩色图像，分别对应红、绿、蓝三个颜色通道）。*/
    /* 这行代码的作用是创建一个 4×4 的矩阵，矩阵中的每个元素是一个 3 通道的 8 位无符号整数，
    且每个通道的值都为 1。换句话说，它创建了一个 4×4 的全白矩阵
    （在 OpenCV 中，白色通常表示为 (255, 255, 255)，但这里每个通道的值是 1，
    所以这个矩阵在视觉上不会显示为白色，而是会根据后续处理显示为其他颜色）。 */
    m1 = cv::Mat::ones(cv::Size(4, 4), CV_8UC3);
    /* [  1,   0,   0,   1,   0,   0,   1,   0,   0,   1,   0,   0;
          1,   0,   0,   1,   0,   0,   1,   0,   0,   1,   0,   0;
          1,   0,   0,   1,   0,   0,   1,   0,   0,   1,   0,   0;
          1,   0,   0,   1,   0,   0,   1,   0,   0,   1,   0,   0] */
    std::cout << "m1=" << endl
              << m1 << std::endl;

    m2 = cv::Mat::zeros(cv::Size(4, 4), CV_8UC3);    // 3通道全为0,4*4
    std::cout << "m2=" << endl
              << m2 << std::endl;
    m2 = Scalar(105, 3, 25);    //向每个通道赋值
    std::cout << "m2=" << endl
              << m2 << std::endl;
    Mat m10(4, 4, CV_32FC2, Scalar(1, 3));    // 32位双通道4*4,并且为通道分别赋值（1,3）
    std::cout << m10 << std::endl;
    Mat m3(cv::Size(245, 245), CV_32FC3, Scalar(255, 0, 0));
    imshow("m3", m3);
    Mat m4(cv::Size(245, 245), CV_32FC3, Scalar(0, 255, 0));
    imshow("m4", m4);
    Mat m5(cv::Size(245, 245), CV_32FC3, Scalar(0, 0, 255));
    imshow("m5", m5);
}

// 通过下标访问像素
void QuickDemo::pixel_visit_Demo(Mat &image) {
    int width   = image.cols;
    int height  = image.rows;
    int channel = image.channels( );
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            // gray image
            if (channel == 1) {
                int px                  = image.at< uchar >(h, w);
                image.at< uchar >(h, w) = 255 - px;
            }
            // 3 channel image
            if (channel == 3) {
                Vec3b bgr                  = image.at< Vec3b >(h, w);
                image.at< Vec3b >(h, w)[0] = 255 - bgr[0];
                image.at< Vec3b >(h, w)[1] = 255 - bgr[1];
                image.at< Vec3b >(h, w)[2] = 255 - bgr[2];
            };
        }
    }
    imshow("image反色", image);
}

// 通过指针访问像素
void QuickDemo::pixel_visiter_Demo(Mat &image) {
    int width   = image.cols;
    int height  = image.rows;
    int channel = image.channels( );
    if (channel == 1) {
        for (int h = 0; h < height; h++) {
            uchar *current_row = image.ptr< uchar >(h);
            for (int w = 0; w < width; w++) {
                int px         = *current_row;
                *current_row++ = 255 - px;
            }
        }
    }
    if (channel == 3) {
        for (int h = 0; h < height; h++) {
            uchar *current_row = image.ptr< uchar >(h);
            for (int w = 0; w < width; w++) {
                *current_row++ = 255 - *current_row;
                *current_row++ = 255 - *current_row;
                *current_row++ = 255 - *current_row;
            }
        }
    }
    imshow("image反色", image);
}

//像素统计
void QuickDemo::pixel_statistic_Demo(Mat &image) {
    double             min_v, max_v;
    Point              minLoc, maxLoc;
    std::vector< Mat > mv;
    split(image, mv);    // 将输入的多通道图像 image 分割成多个单通道图像，并将这些单通道图像存储在 mv 中。

    for (int i = 0; i < mv.size( ); i++) {
        /* 在图像中（矩阵/数组）中找到全局最小和最大值

            void minMaxLoc(InputArray src, CV_OUT double* minVal,
                            CV_OUT double* maxVal = 0, CV_OUT Point* minLoc = 0,
                            CV_OUT Point* maxLoc = 0, InputArray mask = noArray());
            src，输入图像，单通道图像。
            minVal，返回最小值的指针。若无需返回，此值设为 NULL。
            maxVal，返回最大值的指针。若无需返回，此值设为 NULL。
            minLoc，返回最小值位置的指针（二维情况下）。若无需返回，此值设为 NULL。
            maxLoc，返回最大值位置的指针（二维情况下）。若无需返回，此值设为 NULL。
            mask，最后一个参数是一个可选的掩码（mask），用于指定哪些像素参与计算，需要与输入图像集有相同尺寸。
                    传入一个空的 Mat 对象，表示没有掩码，即对整个图像通道进行计算。
        */
        minMaxLoc(mv[i], &min_v, &max_v, &minLoc, &maxLoc, Mat( ));
        std::cout << "No.channel:" << i
                  << "\tmin_value=" << min_v
                  << "\tmax_value=" << max_v << std::endl;
        std::cout << "No.channel:" << i
                  << "\tminLoc.x=" << minLoc.x
                  << "\tmaxLoc.y=" << minLoc.y << std::endl;
    }

    Mat mean, stddev;    //这里可以理解为矩阵
    /*
        void meanStdDev(InputArray src, OutputArray mean, OutputArray stddev,
                        InputArray mask=noArray())
        InputArray src 一般的彩色图，灰度图都可
        mean：输出参数，计算均值
        stddev：输出参数，计算标准差
        mask：可选参数
    */
    meanStdDev(image, mean, stddev);    //对mean与stddev创建了[1 x 3]的矩阵
    cout << mean.size( ) << endl
         << mean.elemSize1( ) << endl;    // elemSize1( )返回单个通道的字节数，这里是8,说明Mat里的数据是32位浮点channel3的类型
    cout << "mean=" << mean.at< double >(0)
         << "\t" << mean.at< double >(1)
         << "\t" << mean.at< double >(2) << std::endl;
    cout << "stddev=" << stddev.at< double >(0)
         << "\t" << stddev.at< double >(1)
         << "\t" << stddev.at< double >(2) << std::endl;
}

void QuickDemo::operators_Demo(Mat &image) {
    Mat dst1 = Mat::zeros(image.size( ), image.type( ));

    // 加减除法：被因数可以是相同数据类型的矩阵，也可以是一个标量。
    Mat m1 = Mat::zeros(image.size( ), image.type( ));
    m1     = Scalar(5, 5, 5);
    dst1   = image - m1;
    dst1   = dst1 + Scalar(30, 30, 30);
    imshow("加法操作", dst1);

    Mat dst2 = Mat::zeros(image.size( ), image.type( ));

    // 乘法：被因数必须是相同数据类型的矩阵（multiply函数对溢出uchar进行截断）。
    Mat m2(image.size( ), image.type( ), Scalar(2, 2, 2));
    multiply(image, m2, dst2);
    imshow("乘法操作", dst2);

    Mat dst3    = Mat::zeros(image.size( ), image.type( ));
    int width   = image.cols;
    int height  = image.rows;
    int channel = image.channels( );
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            Vec3b p1 = image.at< Vec3b >(h, w);
            Vec3b p2 = m2.at< Vec3b >(h, w);

            dst3.at< Vec3b >(h, w)[0] = saturate_cast< uchar >(p1[0] + p2[0]);
            dst3.at< Vec3b >(h, w)[1] = saturate_cast< uchar >(p1[1] + p2[1]);
            dst3.at< Vec3b >(h, w)[2] = saturate_cast< uchar >(p1[2] + p2[2]);
        }
    }
    imshow("image遍历加法", dst3);

    // 使用opencv自带函数，速度肯定更快
    Mat m_test(image.size( ), image.type( ), Scalar(20, 20, 20));
    Mat m_divide(image.size( ), image.type( ), Scalar(2, 2, 2));
    Mat dst4 = Mat::zeros(image.size( ), image.type( ));
    Mat dst5 = Mat::zeros(image.size( ), image.type( ));
    Mat dst6 = Mat::zeros(image.size( ), image.type( ));
    add(image, m_test, dst4);
    subtract(image, m_test, dst5);
    divide(image, m_divide, dst6);
    imshow("image加法", dst4);
    imshow("image减法", dst5);
    imshow("image除法", dst6);
}
