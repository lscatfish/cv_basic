#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <iostream>
#include "quick.h"


using namespace cv;

int main(int argc, char **argv) {
    Mat src = imread("../img/input/cube_in.png");
    if (src.empty( )) {
        printf("Could not load images...\n");
        return -1;
    }

    QuickDemo qk;               //实例化
    qk.image_contrast_Demo(src);    //调用

    waitKey(0);
    destroyAllWindows( );
    return 0;
}
