#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>

int main()
{
    int width=400;
    int height=300;
    int radius=30;
    int stride = 4;
    std::pair<int,int> center(height/2,0);
    
    cv::Mat base_img(width,height,CV_8UC3,cv::Scalar(128,128,128));
    int num_images = width/stride;

    cv::namedWindow("Base Image");
    cv::imshow("Base Image",base_img);

    while(1)
    {
        char c = cv::waitKey(0);
        if(c==27) break;
    }

    cv::imwrite("0.jpg",base_img); 
    for(int img=0;img<num_images;img++)
    {
        cv::Mat im = (base_img.clone());

        cv::circle(im,cv::Point(center.first,center.second),radius,cv::Scalar(0,0,255),-1);
        cv::imwrite(std::to_string(img+1)+".jpg",im); 
        center.second += stride;

        cv::namedWindow("Image");
        cv::imshow("Image",im);
        

        char c=cv::waitKey(30);
        if(c==27) break;
    }

    return 0;
}

