#ifndef MOG_H
#define MOG_H

#include <opencv2/opencv.hpp>
using namespace std;

struct PixelInfo
{
    vector<float> std_dev;
    vector<float> weight;
    vector<vector<int>> mean;
};

class MOG
{
    int rows;
    int cols;
    int num_gauss;
    int num_channel;
    vector<vector<PixelInfo> >frameInfo;
    float lr = 0.2;
    float T = 0.5;
    float match_scale = 2.5;
    float HIGH_DEV = 250.0;
    float LOW_WEIGHT = 0.1;
    float epsilon = 1e-20;
    float const_prob_factor;

    public:
    MOG(const cv::Mat& frame,int K=3,float lr=0.2,float T=0.5);
    void apply(const cv::Mat& frame,cv::Mat& result);
};
#endif

