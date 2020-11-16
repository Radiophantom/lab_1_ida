#include <QApplication>
#include <iostream>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <vector>
#include <math.h>

using namespace cv;
using namespace std;

int level_treshold = 0.05;
int absolute_max_distance = sqrt( 3*( 255^2 ) );

Mat centroid_matrix      = Mat::zeros(1, 6, CV_8UC3);
Mat prev_centroid_matrix = Mat::zeros(1, 6, CV_8UC3);

int centroid_centers_check( )
{
  int tmp_var;
  for( int i = 0; i < 6; i++ )
  {
    tmp_var = sqrt( ( centroid_matrix.at<Vec3b>(0,i)[0] - prev_centroid_matrix.at<Vec3b>(0,i)[0] )^2 +
                    ( centroid_matrix.at<Vec3b>(0,i)[1] - prev_centroid_matrix.at<Vec3b>(0,i)[1] )^2 +
                    ( centroid_matrix.at<Vec3b>(0,i)[2] - prev_centroid_matrix.at<Vec3b>(0,i)[2] )^2   );
    if( tmp_var/absolute_max_distance > level_treshold )
    {
      prev_centroid_matrix = centroid_matrix;
      return 1;
    }
  }
  return 0;
}

int find_pixel_claster( Vec3b pixel )
{
  vector <int> tmp_var_vector = {0,0,0,0,0,0};
  for( int i = 0; i < 6; i++ )
  {
    tmp_var_vector[i] = sqrt( ( prev_centroid_matrix.at<Vec3b>(0,i)[0] - pixel[0] )^2 +
                              ( prev_centroid_matrix.at<Vec3b>(0,i)[1] - pixel[1] )^2 +
                              ( prev_centroid_matrix.at<Vec3b>(0,i)[2] - pixel[2] )^2   );
  }
  return( min_element( tmp_var_vector.begin(), tmp_var_vector.end() ) - tmp_var_vector.begin() );
}

void centroid_calc( Mat img, Mat mask )
{
  int centroid_avg[6][3] = {};
  int centroid_pix_num[6] = {};
  for( int i = 0; i < img.rows; i++ )
  {
    for( int j = 0; j < img.cols; j++ )
    {
      centroid_avg[mask.at<uchar>(i,j)][0] += img.at<Vec3b>(i,j)[0];
      centroid_avg[mask.at<uchar>(i,j)][1] += img.at<Vec3b>(i,j)[1];
      centroid_avg[mask.at<uchar>(i,j)][2] += img.at<Vec3b>(i,j)[2];
      centroid_pix_num[mask.at<uchar>(i,j)] += 1;
    }
  }
  for( int i = 0; i < 6; i++ )
  {
    centroid_matrix.at<Vec3b>(0,i)[0] = centroid_avg[i][0]/centroid_pix_num[i];
    centroid_matrix.at<Vec3b>(0,i)[1] = centroid_avg[i][1]/centroid_pix_num[i];
    centroid_matrix.at<Vec3b>(0,i)[2] = centroid_avg[i][2]/centroid_pix_num[i];
  }
}

void claster_divide( Mat input_img, Mat mask )
{
  int tmp_pix_num = 0;
  for( int i = 0; i < input_img.rows; i++ )
  {
    for( int j = 0; j < input_img.cols; j++ )
    { 
      tmp_pix_num = mask.at<uchar>(i,j);
      input_img.at<Vec3b>(i,j)[0] = prev_centroid_matrix.at<Vec3b>(0,tmp_pix_num)[0];
      input_img.at<Vec3b>(i,j)[1] = prev_centroid_matrix.at<Vec3b>(0,tmp_pix_num)[1];
      input_img.at<Vec3b>(i,j)[2] = prev_centroid_matrix.at<Vec3b>(0,tmp_pix_num)[2];
    }
  }
}

int main(int argc, char *argv[])
{
  QApplication a(argc, argv);

  // centroid matrix initialization
  prev_centroid_matrix.at<Vec3b>(0,0) = {0  ,0  ,0  };
  prev_centroid_matrix.at<Vec3b>(0,1) = {50 ,50 ,50 };
  prev_centroid_matrix.at<Vec3b>(0,2) = {100,100,100};
  prev_centroid_matrix.at<Vec3b>(0,3) = {150,150,150};
  prev_centroid_matrix.at<Vec3b>(0,4) = {200,200,200};
  prev_centroid_matrix.at<Vec3b>(0,5) = {255,255,255};

  Mat img_orig = imread("/home/skr/qt_projects/lab_1/nature.jpg");
  Mat img_processing = img_orig.clone();

  Mat pix_matrix = Mat::zeros(img_orig.size(), CV_8U);

  int centroid_changed = 1; 
  while( centroid_changed == 1 )
  { 
    for( int i = 0; i < img_orig.rows; i++ )
    {
      for( int j = 0; j < img_orig.cols; j++ )
      {
        pix_matrix.at<uchar>(i,j) = find_pixel_claster( img_orig.at<Vec3b>(i,j) );
      }
    }
    centroid_calc( img_orig, pix_matrix );
    centroid_changed = centroid_centers_check();
  }
  
  claster_divide( img_processing, pix_matrix );
  imshow( "Original image", img_orig );
  imshow( "Clastered image", img_processing );
  waitKey();

  return a.exec();
}
