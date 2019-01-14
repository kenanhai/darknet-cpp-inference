#include <iostream>
#include <opencv2/opencv.hpp>
#include <memory>
#include "dn_detector.h"
#include <string>
#include<fstream>
using namespace cv;

int main(int argc, char **argv)
{

  std::cout << argv[0] << " cfgfile weightfile labelfile imagefile" << std::endl;
  //float thresh = find_float_arg(argc, argv, "-thresh", .5);
  //char *filename = (argc > 4) ? argv[4]: 0;
  //char *outfile = find_char_arg(argc, argv, "-out", 0);
  //int fullscreen = find_arg(argc, argv, "-fullscreen");
  //test_detector("cfg/coco.data", argv[2], argv[3], filename, thresh, .5, outfile, fullscreen);
  std::string cfgfile = argv[1];
  std::string weightfile = argv[2];
  std::string labelfile = argv[3];
  std::string imagefile = argv[4];
  std::string txt_path = argv[5];
  std::string save_path = argv[6];

  std::shared_ptr<stereo_bm::DNDetector> detector;
  detector = std::make_shared<stereo_bm::DNDetector>(cfgfile, weightfile, labelfile);
  std::ifstream in(txt_path);
  std::string line,name;
  int n=0;
  while (getline(in, line))
  {
     std::cout<<line<<std::endl;
     cv::Mat image = cv::imread(line);
     int nPos1 = line.find_last_of("/");
     name = line.substr(nPos1+1,-1);
     std::cout << n <<"  "<< name <<std::endl;
     detector->Detect(image,name,save_path);
     n++;
  }
  in.close();
  return 0;
}
