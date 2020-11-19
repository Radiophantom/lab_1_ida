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
double stop_level_treshold = 0.1;

// maximum allowable Evklid distance
double absolute_max_distance = sqrt( 3*( 255^2 ) );

// check if centroids coordinates changed or not
int centroid_centers_check( Mat& centroid_matrix, Mat& next_centroid_matrix )
{

  double tmp_value = 0;

  for( int i = 0; i < 6; i++ )
  {
    tmp_value = sqrt( ( abs( centroid_matrix.at<Vec3b>(0,i)[0] - next_centroid_matrix.at<Vec3b>(0,i)[0] ) )^2 +
                      ( abs( centroid_matrix.at<Vec3b>(0,i)[1] - next_centroid_matrix.at<Vec3b>(0,i)[1] ) )^2 +
                      ( abs( centroid_matrix.at<Vec3b>(0,i)[2] - next_centroid_matrix.at<Vec3b>(0,i)[2] ) )^2   );
    if( ( tmp_value/absolute_max_distance ) > stop_level_treshold )
    {
      centroid_matrix = next_centroid_matrix;
      return 1;
    }
  }

  return 0;

}

// find the claster for current pixel in the input image
int find_pixel_claster( Mat& centroid_matrix, Vec3b& pixel )
{

  vector <double> tmp_value_vector = { 0, 0, 0, 0, 0, 0 };

  for( int i = 0; i < 6; i++ )
  {
    tmp_value_vector[i] = sqrt( ( abs( centroid_matrix.at<Vec3b>(0,i)[0] - pixel[0] ) )^2 +
                                ( abs( centroid_matrix.at<Vec3b>(0,i)[1] - pixel[1] ) )^2 +
                                ( abs( centroid_matrix.at<Vec3b>(0,i)[2] - pixel[2] ) )^2   );
  }

  return( min_element( tmp_value_vector.begin(), tmp_value_vector.end() ) - tmp_value_vector.begin() );

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
    Mat img_processing = img_orig.clone();

    Mat claster_matrix = Mat::zeros( img_processing.size(), CV_8UC1 );

    int centroid_changed = 1;

    while( centroid_changed != 0 )
    {
      for( int i = 0; i < img_processing.rows; i++ )
      {
        for( int j = 0; j < img_processing.cols; j++ )
        {
          claster_matrix.at<uchar>(i,j) = find_pixel_claster( centroid_matrix, img_processing.at<Vec3b>(i,j) );
        }
      }
      claster_centroids_calc( img_processing, claster_matrix, next_centroid_matrix );
      centroid_changed = centroid_centers_check( centroid_matrix, next_centroid_matrix );
    }

    divide_image_into_clasters( img_processing, claster_matrix, centroid_matrix );
    imshow( "Original image", img_orig );
    imshow( "Clastered image", img_processing );
    waitKey();

    return a.exec();
}
