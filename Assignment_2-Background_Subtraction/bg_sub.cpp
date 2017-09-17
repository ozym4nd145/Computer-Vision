#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include "mog.h"

using namespace std;
using namespace cv;

void processVideo(char* videoFilename,int arg);
int main(int argc,char**argv)
{
	namedWindow("Frame");
    namedWindow("FG Mask MOG 2");
	processVideo(argv[1],argc);
    return 0;
}

void processVideo(char* videoFilename,int arg)
{
	VideoCapture capture(videoFilename);
	Mat frame;
	Mat fgMaskMOG2;
	Mat myfgMask;
	Ptr<BackgroundSubtractor> pMOG2;
	pMOG2 = createBackgroundSubtractorMOG2();

	int keyboard=0;

	if(!capture.isOpened()){
        cerr << "Unable to open video file: " << videoFilename << endl;
        exit(EXIT_FAILURE);
    }
    if(arg==2)
        capture.read(frame);
    else
        frame = cv::imread("./debug_img/0.jpg");

    MOG my_mog(frame,4,0.1,0.79);
    int i=0;
	while( (char)keyboard != 'q' && (char)keyboard != 27 ){
        if(arg!=2) 
            frame = cv::imread("./debug_img/"+to_string(i++)+".jpg");
        else
        {
            if(!capture.read(frame)) {
                cerr << "Unable to read next frame." << endl;
                cerr << "Exiting..." << endl;
                exit(EXIT_FAILURE);
            }
        }
        i %= 100;
        pMOG2->apply(frame, fgMaskMOG2);
        my_mog.apply(frame, myfgMask);
        //cout<<myfgMask<<endl;
        imshow("Frame", frame);
        imshow("FG Mask MOG 2", fgMaskMOG2);
        imshow("My FG Mask ", myfgMask);
        keyboard = waitKey( 30 );
    }
    capture.release();
}
