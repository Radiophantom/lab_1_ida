#include <QApplication>
#include <iostream>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <vector>
#include <math.h>

using namespace cv;
using namespace std;

// set level treshold for stopping iterations in the loop
const double stop_level_treshold = 0.01;

// maximum allowable Evklid distance
const double absolute_max_distance = 255*sqrt(3);

// Evklid distance calculate
double evklid_calc( Vec3b& x, Vec3b& y )
{
  return( sqrt( pow( (x[0] - y[0]), 2 ) + pow( ( x[1] - y[1] ), 2 ) + pow( ( x[2] - y[2] ), 2 ) ) );
}

// check if centroids coordinates changed or not
int centroid_centers_check( Mat& centroid_matrix, Mat& next_centroid_matrix )
{

  double tmp_value = 0;

  for( int i = 0; i < 6; i++ )
  {
    tmp_value = ( evklid_calc( centroid_matrix.at<Vec3b>(0,i), next_centroid_matrix.at<Vec3b>(0,i) ) / absolute_max_distance );
    if( tmp_value > stop_level_treshold )
    {
      centroid_matrix = next_centroid_matrix;
      return 1;
    }
  }

  return 0;

}

// find index of min element in massive
int min_element_index( double mas[6] )
{

  int index = 0;
  double min_element = mas[0];
  
  for( int i = 1; i < 6; i++ )
  {
    if( mas[i] < min_element )
    {
      index = i;
      min_element = mas[i];
    } 
  }

  return( index );

};

// find the claster for current pixel in the input image
int find_pixel_claster( Mat& centroid_matrix, Vec3b& pixel )
{

  double tmp_mas[6] = { };

  for( int i = 0; i < 6; i++ )
  {
    tmp_mas[i] = evklid_calc( centroid_matrix.at<Vec3b>(0,i), pixel );
  }

  return( min_element_index( tmp_mas ) );

}

// calculate new centroids for each claster
void claster_centroids_calc( Mat& input_img, Mat& claster_matrix, Mat& next_centroid_matrix )
{

  Mat centroid_avg          = Mat::zeros(1, 6, CV_64FC3);
  Mat centroid_pixel_number = Mat::zeros(1, 6, CV_64FC1);

  for( int i = 0; i < input_img.rows; i++ )
  {
    for( int j = 0; j < input_img.cols; j++ )
    {
      centroid_avg.at<Vec3d>(0, claster_matrix.at<uchar>(i,j))[0]         += input_img.at<Vec3b>(i,j)[0];
      centroid_avg.at<Vec3d>(0, claster_matrix.at<uchar>(i,j))[1]         += input_img.at<Vec3b>(i,j)[1];
      centroid_avg.at<Vec3d>(0, claster_matrix.at<uchar>(i,j))[2]         += input_img.at<Vec3b>(i,j)[2];
      centroid_pixel_number.at<double>(0, claster_matrix.at<uchar>(i,j))  += 1;
    }
  }

  for( int i = 0; i < 6; i++ )
  {
    centroid_avg.at<Vec3d>(0,i)[0] = centroid_avg.at<Vec3d>(0,i)[0] / centroid_pixel_number.at<double>(0,i);
    centroid_avg.at<Vec3d>(0,i)[1] = centroid_avg.at<Vec3d>(0,i)[1] / centroid_pixel_number.at<double>(0,i);
    centroid_avg.at<Vec3d>(0,i)[2] = centroid_avg.at<Vec3d>(0,i)[2] / centroid_pixel_number.at<double>(0,i);
  }
  
  centroid_avg.convertTo( next_centroid_matrix, CV_8UC3 );

}

// divide original image into calculated clasters
void divide_image_into_clasters( Mat& input_img, Mat& claster_mask, Mat& centroid_matrix )
{
  for( int i = 0; i < input_img.rows; i++ )
  {
    for( int j = 0; j < input_img.cols; j++ )
    {
      input_img.at<Vec3b>(i,j)[0] = centroid_matrix.at<Vec3b>(0,claster_mask.at<uchar>(i,j))[0];
      input_img.at<Vec3b>(i,j)[1] = centroid_matrix.at<Vec3b>(0,claster_mask.at<uchar>(i,j))[1];
      input_img.at<Vec3b>(i,j)[2] = centroid_matrix.at<Vec3b>(0,claster_mask.at<uchar>(i,j))[2];
    }
  }
}

int main(int argc, char *argv[])
{

    QApplication a(argc, argv);

    Mat centroid_matrix      = Mat::zeros(1, 6, CV_8UC3);
    Mat next_centroid_matrix = Mat::zeros(1, 6, CV_8UC3);

    centroid_matrix.at<Vec3b>(0,0)[0] = 0;   centroid_matrix.at<Vec3b>(0,0)[1] = 0;   centroid_matrix.at<Vec3b>(0,0)[2] = 0;  
    centroid_matrix.at<Vec3b>(0,1)[0] = 50;  centroid_matrix.at<Vec3b>(0,1)[1] = 50;  centroid_matrix.at<Vec3b>(0,1)[2] = 50; 
    centroid_matrix.at<Vec3b>(0,2)[0] = 100; centroid_matrix.at<Vec3b>(0,2)[1] = 100; centroid_matrix.at<Vec3b>(0,2)[2] = 100;
    centroid_matrix.at<Vec3b>(0,3)[0] = 150; centroid_matrix.at<Vec3b>(0,3)[1] = 150; centroid_matrix.at<Vec3b>(0,3)[2] = 150;
    centroid_matrix.at<Vec3b>(0,4)[0] = 200; centroid_matrix.at<Vec3b>(0,4)[1] = 200; centroid_matrix.at<Vec3b>(0,4)[2] = 200;
    centroid_matrix.at<Vec3b>(0,5)[0] = 255; centroid_matrix.at<Vec3b>(0,5)[1] = 255; centroid_matrix.at<Vec3b>(0,5)[2] = 255;

    Mat img_orig = imread("/home/skr/qt_projects/lab_1/nature.jpg");

    Mat img_processing = Mat::zeros( img_orig.size(), CV_8UC3 );
    Mat claster_matrix = Mat::zeros( img_orig.size(), CV_8UC1 );

    int centroid_changed = 1;

    while( centroid_changed != 0 )
    {
      for( int i = 0; i < img_orig.rows; i++ )
      {
        for( int j = 0; j < img_orig.cols; j++ )
        {
          claster_matrix.at<uchar>(i,j) = find_pixel_claster( centroid_matrix, img_orig.at<Vec3b>(i,j) );
        }
      }
      claster_centroids_calc( img_orig, claster_matrix, next_centroid_matrix );
      centroid_changed = centroid_centers_check( centroid_matrix, next_centroid_matrix );
    }

    divide_image_into_clasters( img_processing, claster_matrix, centroid_matrix );
    imshow( "Original image", img_orig );
    imshow( "Clastered image", img_processing );
    waitKey();

    return a.exec();
}
