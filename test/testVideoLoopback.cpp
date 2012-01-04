#include <iostream>
#include <fstream>
#include "../precomp.hpp"

#include <linux/videodev2.h>
#include "v4l2lb.h"

#define LOG(msg) { std::cout << msg <<endl; std::cout.flush(); }

using namespace std;
using namespace cv;


int main(int argc, char* argv[])
{
	vector<int> device_indices;
	if ( argc <= 1 ) {
		device_indices.push_back(0);
	} else {
		for (int i=1; i < argc; ++i)
			device_indices.push_back(atoi(argv[i]));
	}
	int device_cnt = device_indices.size();
	if ( device_cnt <1 ) {
		LOG("No device");
		return -1;
	}

	vector<VideoCapture> capture(device_cnt);
	for (int i = 0; i < device_cnt; ++i){
		if(!capture[i].open(device_indices[i])) {
		      LOG("Can't open device #"<<device_indices[i]);
		      device_indices.erase(device_indices.begin()+i);
		      capture.erase(capture.begin()+i);
		}
	}
	device_cnt = device_indices.size();
	if ( device_cnt <1 ) {
		LOG("No device");
		return -1;
	}

	namedWindow("video", CV_WINDOW_AUTOSIZE);
	Mat img;
	capture[0]>>img;
	LOG(img.type()<<" "<<img.cols<<" "<<img.rows);
    Mat dImg(img.rows, img.cols*device_cnt, img.type());

    //////////////////////////////////////////////////////////////////////
    string device_name="/dev/video3";
    int frame_size = 3*dImg.cols*dImg.rows/2;
    int dev_fd = v4l2lb_open_device(const_cast<char*>(device_name.c_str()), dImg.cols, dImg.rows, frame_size);
    
    cout<< "   "<<frame_size<<endl;
    //////////////////////////////////////////////////////////////////////
	vector<Mat> roi(device_cnt);
	for (int i = 0; i < device_cnt; ++i){
		roi[i]=dImg(Rect(img.cols*i,0,img.cols,img.rows));
	}
	
	
    /*
    CV_FOURCC('P','I','M','1')    = MPEG-1 codec
    CV_FOURCC('M','J','P','G')    = motion-jpeg codec (does not work well)
    CV_FOURCC('M', 'P', '4', '2') = MPEG-4.2 codec
    CV_FOURCC('D', 'I', 'V', '3') = MPEG-4.3 codec
    CV_FOURCC('D', 'I', 'V', 'X') = MPEG-4 codec
    CV_FOURCC('U', '2', '6', '3') = H263 codec
    CV_FOURCC('I', '2', '6', '3') = H263I codec
    CV_FOURCC('F', 'L', 'V', '1') = FLV1 codec
    */
    Mat vImg;
    VideoWriter writer("video.mpg", CV_FOURCC('M','P','G','1'), 24, dImg.size(), true);
    
	while(true){
		for (int i = 0; i < device_cnt; ++i){
			capture[i]>>img;
			img.copyTo(roi[i]);
            
            cvtColor(dImg,vImg,CV_RGB2YCrCb);
            vector<uchar> buf;
            //FIXME: 
            imencode(".jpg", vImg, buf);
            cout<< sizeof(vImg) <<endl;
            cout<< buf.size()<<endl;
            
            //write(dev_fd,&buf[0],buf.size());
            if (writer.isOpened()) writer<<vImg;
            
            imshow("video", dImg);
			char key = waitKey(5);
			switch (key)
			{
				case 's':
					imwrite("test0.jpg", roi[0]);
					imwrite("test1.jpg", roi[1]);
					break;
				case 27:
				case 'q':
					return 0;
					break;
			}
		}

	}
}

