#include "dn_detector.h"
#include <memory>
#include<string.h>

using namespace std;
//using namespace stereo_bm;
using namespace cv;

void stereo_bm::DNDetector::rgbgr_image_(image im)
{
  int i;
  for(i = 0; i < im.w*im.h; ++i){
    float swap = im.data[i];
    im.data[i] = im.data[i+im.w*im.h*2];
    im.data[i+im.w*im.h*2] = swap;
  }
}

image stereo_bm::DNDetector::ipl_to_image_(IplImage* src)
{
  int h = src->height;
  int w = src->width;
  int c = src->nChannels;
  image im = make_image(w, h, c);
  unsigned char *data = (unsigned char *)src->imageData;
  int step = src->widthStep;
  int i, j, k;

  for(i = 0; i < h; ++i){
    for(k= 0; k < c; ++k){
      for(j = 0; j < w; ++j){
        im.data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;
      }
    }
  }
  return im;
}
image stereo_bm::DNDetector::mat_to_image_(Mat m)
{
  IplImage ipl = m;
  image im = ipl_to_image_(&ipl);
  rgbgr_image_(im);
  return im;
}


char* stereo_bm::DNDetector::string_to_cstr(const std::string str){
  //std::string str = "string";
  char *cstr = new char[str.length() + 1];
  strcpy(cstr, str.c_str());
  return cstr;
  //delete [] cstr;
}

stereo_bm::DNDetector::DNDetector(const std::string& cfgfile, const std::string& weightfile, const std::string& name_list){
  names_ = get_labels(string_to_cstr(name_list));
  net = load_network(string_to_cstr(cfgfile),  string_to_cstr(weightfile), 0);
  set_batch_network(net, 1);
  srand(2222222);
}

void stereo_bm::DNDetector::Detect(const cv::Mat &img,std::string name,std::string save_path){
  double time;
  cv::Rect rect;
  image im = mat_to_image_(img);
  image sized = letterbox_image(im, net->w, net->h);
  //std::cout<<net->w<<" "<<net->h<<std::endl;
  layer l = net->layers[net->n-1];
  float *X = sized.data;
  time=what_time_is_it_now();
  network_predict(net, X);
  printf("Predicted in %f seconds.\n", what_time_is_it_now()-time);
  int nboxes = 0;
  detection *dets = get_network_boxes(net, im.w, im.h, thresh_, hier_thresh_, 0, 1, &nboxes);
  if (nms_) do_nms_sort(dets, nboxes, l.classes, nms_);
  image ** alphabet;
  //draw_detection
  int  i,j ;
  for(i = 0; i < nboxes; ++i){
  int class0 = -1;
  for(j = 0; j < l.classes; ++j){
  //printf("thresh is %f\n",thresh_);
  if (dets[i].prob[j] > thresh_){
        class0 = j;
        }}
  if(class0>=0)
  {
      box b = dets[i].bbox;
      //printf("%f %f %f %f\n", b.x, b.y, b.w, b.h); 
      /// top为矩形框上方纵坐标，bot为矩形框下方纵坐标（这里也是按照OpenCV的习惯，y轴往下，x轴往右，原点为左上角）
        int left  = (b.x-b.w/2.)*im.w;
        int right = (b.x+b.w/2.)*im.w;
        int top   = (b.y-b.h/2.)*im.h;
        int bot   = (b.y+b.h/2.)*im.h;

        /// 越界检测：
        if(left < 0) left = 0;
        if(right > im.w-1) right = im.w-1;
        if(top < 0) top = 0;
        if(bot > im.h-1) bot = im.h-1;
        rect.x = left;
        rect.y = top;
        rect.width = right - left;
        rect.height = bot - top;
        //printf("%d %d\n",img.cols,img.rows);
        //printf("%d %d\n",im.w,im.h);
        //printf("%d %d %d %d\n", rect.x, rect.y, rect.width, rect.height);
        cv::Mat crop_roi = img(rect);
        std::string new_name = save_path  + name;
        cv::imwrite(new_name,crop_roi);
  }
  }
  //draw_detections(im, dets, nboxes, thresh_, names_, alphabet, l.classes);
  free_detections(dets, nboxes);

  //save_image(im, "predictions");
//#ifdef OPENCV
//  make_window("predictions", 512, 512, 0);
//  show_image(im, "predictions", 0);
//#endif

  free_image(im);
  free_image(sized);
}
