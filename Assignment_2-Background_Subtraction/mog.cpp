#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include "mog.h"

using namespace std;

MOG::MOG(const cv::Mat& frame,int K,float lr,float T): frameInfo(frame.rows,vector<PixelInfo>(frame.cols)),T(T),lr(lr)
{
    this->rows = frame.rows;
    this->cols = frame.cols;
    this->num_gauss = K;
    this->num_channel = frame.channels();
    this->epsilon = 1e-20;
    for(int i=0;i<rows;i++)
    {
        for(int j=0;j<cols;j++)
        {
            PixelInfo& pixel = frameInfo[i][j];
            pixel.std_dev = vector<float>(K,250.0);
            pixel.weight = vector<float>(K,1.0/K);
            pixel.mean = vector<vector<int>>(K,vector<int>(frame.channels(),0));
        }
    }
    cout<<"Rows: "<<rows<<", Cols: "<<cols<<", channels:"<<num_channel<<", gauss:"<<num_gauss<<endl;
    const_prob_factor = (1.0/(pow((2*M_PI),(0.5*num_channel))));
}

void MOG::apply(const cv::Mat& frame, cv::Mat& result)
{
    result.create(rows,cols,CV_8UC1);
    for(int i=0;i<rows;i++)
    {
        for(int j=0;j<cols;j++)
        {
            cv::Vec3b img_px = frame.at<cv::Vec3b>(i,j);
            PixelInfo& info_px = frameInfo[i][j];

//            if(i==0 && j==0)
//            {
//                cout<<"Mean: \n";
//                for(int k=0;k<num_gauss;k++)
//                {
//                    cout<<k<<": ";
//                    for(int ch=0;ch<num_channel;ch++)
//                        cout<<info_px.mean[k][ch]<<", ";
//                    cout<<endl;
//                    cout<<"\tStd: "<<info_px.std_dev[k]<<endl;
//                    cout<<"\tWeight: "<<info_px.weight[k]<<endl;
//                }
//            }
            
            // Getting sort index of the pixels
            vector<pair<float,int> > sort_criteria;
            for(int k=0;k<num_gauss;k++)
                sort_criteria.push_back(make_pair(((-1*info_px.weight[k])/info_px.std_dev[k]),k));
            sort(sort_criteria.begin(),sort_criteria.end()); 
           
            // Determining background limit
            float cumsum = 0;
            //denotes whether the corresponding gaussian is background or not
            vector<bool> is_background(this->num_gauss,false); 

            for(int k=0;k<num_gauss;k++)
            {
                cumsum += info_px.weight[sort_criteria[k].second];
                is_background[sort_criteria[k].second] = true;
                if(cumsum > this->T)
                    break;
            }


            int match_gaussian = -1;
            int least_prob_gaussian = 0;
            float min_dist = INT_MAX;
            float max_dist = -1;
            float sum_prob = 0;

            for(int k=0;k<num_gauss;k++)
            {
                float answer = 0;
                for(int ch=0;ch<num_channel;ch++)
                {
                    int val = (info_px.mean[k][ch] - (int)img_px[ch]);
                    answer += (val*val);
                }
                answer /= info_px.std_dev[k];
                float dst = sqrt(answer);
                sum_prob += epsilon + (this->const_prob_factor)*(exp((-0.5)*answer)/pow(info_px.std_dev[k],(0.5*num_channel)));

                if(dst < (match_scale*info_px.std_dev[k]) && dst < min_dist)
                {
                    //if it is a match
                    min_dist = dst;
                    match_gaussian=k;
                }
                if(dst > max_dist)
                {
                    least_prob_gaussian = k;
                    max_dist = dst;
                }
            }
            if(match_gaussian == -1)
            {
                // No match possible
                result.at<uchar>(i,j) = 255; //classify as foreground
                info_px.std_dev[least_prob_gaussian] = this->HIGH_DEV;
                for(int ch=0;ch<num_channel;ch++)
                    info_px.mean[least_prob_gaussian][ch] = (int)img_px[ch];
                
                //float diff = info_px.weight[least_prob_gaussian] - this->LOW_WEIGHT;
                //diff /= (num_gauss-1);
                //info_px.weight[least_prob_gaussian] = this->LOW_WEIGHT;
                //for(int k=0;k<num_gauss;k++)
                //{
                //    if(k != least_prob_gaussian)
                //        info_px.weight[k] += diff;
                //}

            }
            else
            {
                result.at<uchar>(i,j) = (is_background[match_gaussian])?(0):(255);                    

                //if(i==0 && j==0)
                //{
                //    cout<<"Pixel: "<<i<<","<<j<<" is "<<is_background[match_gaussian]<<" , and value: "<<(int)result.at<uchar>(i,j)<<endl;
                //    cout<<"dist: "<<min_dist<<endl;
                //    cout<<"matched_gaussian: "<<match_gaussian<<endl;
                //}

                for(int k=0;k<num_gauss;k++)
                {
                    info_px.weight[k] -= (this->lr)*(info_px.weight[k]);
                    if(k==match_gaussian)
                    {
                        info_px.weight[k] += (this->lr);
                        float dst = min_dist * min_dist;
                        float prob_now = epsilon + (this->const_prob_factor)*(exp((-0.5)*dst)/pow(info_px.std_dev[k],(0.5*num_channel)));
                        float rho = (this->lr)*(prob_now);
                        rho /= sum_prob;
                        float answer = 0;
                        for(int ch=0;ch<num_channel;ch++)
                        {
                            info_px.mean[k][ch] = (1-rho)*(info_px.mean[k][ch]) + rho*(int)img_px[ch];
                            int val = (info_px.mean[k][ch] - (int)img_px[ch]);
                            answer += (val*val);
                        }

                        float prev_var = info_px.std_dev[k];
                        prev_var *= prev_var;

                        float new_var = prev_var*(1-rho) + rho*answer;
                        info_px.std_dev[k] = sqrt(new_var);

                        //if(i==0&&j==0)
                        //{
                        //    cout<<"rho: "<<rho<<endl;
                        //    cout<<"sum_prob: "<<sum_prob<<endl;
                        //    cout<<"(this->lr)"<<(this->lr)<<endl;
                        //    cout<<"const factor: "<<(this->const_prob_factor)<<endl;
                        //    cout<<"(exp((-0.5)*dst): "<<(exp((-0.5)*dst))<<endl;
                        //    cout<<"prob_now: "<<prob_now<<endl;
                        //    cout<<"answer: "<<answer<<endl;
                        //    cout<<"prev_var: "<<prev_var<<endl;
                        //    cout<<"new_var: "<<new_var<<endl<<endl;
                        //}
                    }
                }
            }
        }
    }
}
