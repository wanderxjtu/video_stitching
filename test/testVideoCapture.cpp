#include <iostream>
#include "../precomp.hpp"
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

	vector<Mat> roi(device_cnt);
	for (int i = 0; i < device_cnt; ++i){
		roi[i]=dImg(Rect(img.cols*i,0,img.cols,img.rows));
	}

	while(true){
		for (int i = 0; i < device_cnt; ++i){
			capture[i]>>img;
			img.copyTo(roi[i]);
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

