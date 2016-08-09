#include<opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <iostream>
#include<opencv2/highgui/highgui.hpp>
using namespace std; 
 using namespace cv;
	

int main(int argc, char **argv)
{

   printf("hello world\n");
   Mat img = cv::imread("/media/pawan/0B6F079E0B6F079E/All pictures/For analysis/pic2.png",CV_LOAD_IMAGE_COLOR);
   cv::imshow("opencvtest",img);
    waitKey(0);
    string dummy;
    cout << "Enter to continue..." << std::endl;
    getline(std::cin, dummy);

	return 0;
}

using namespace cv;
