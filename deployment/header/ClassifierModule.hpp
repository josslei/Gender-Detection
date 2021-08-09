/*
By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.

                  License Agreement For libfacedetection
                     (3-clause BSD License)

Copyright (c) 2021, Joss Lei, All rights reserved.
josslei@163.com
josslei.0@outlook.com

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <opencv2/opencv.hpp>

#ifdef SPEED_TEST
#define TIME_INTERVAL double
#define SPEED_TESTER(x) double t = (double)cv::getTickCount(); \
                        x \
                        t = ((double)cv::getTickCount() - t) / (double)cv::getTickFrequency();
#define TIME_COST t
#else
#define TIME_INTERVAL void
#define SPEED_TESTER(x) x
#define TIME_COST
#endif // SPEED_TEST

#ifdef OPENCV_CAM
#define BGR2RGB(src, dst); cv::cvtColor(src, dst, cv::COLOR_BGR2RGB);
#else
#define BGR2RGB(src, dst);
#endif // OPENCV_CAM

/* Constants */
constexpr double MEAN[3] = { 0.485 * 255.0, // B
                             0.456 * 255.0, // G
                             0.406 * 255.0  // R
                           }; // Mean
constexpr double  STD[3] = { 0.229 * 255.0, // B ~~ Model of Training No.4 is using the value from
                             0.224 * 255.0, // G ~~ ImageNet (forgot to change), but from Training
                             0.225 * 255.0  // R ~~ No.5.0, these values are based on the dataset.
                           }; // STD
constexpr int INPUT_SIZE[4] = { 1, 3, 200, 200 };

/* Global */
static cv::dnn::Net *Model;   // points to the model
static cv::Mat *Input_Mat;    // royal matrix, for input
static cv::Mat *Buffer;       // buffer used during the preprocessing

/* Initialization of the Classifier
 */
inline void init_classifier(const char *model_path)
{
   Model = new cv::dnn::Net(cv::dnn::readNetFromONNX(model_path));
   Input_Mat = new cv::Mat(4, INPUT_SIZE, CV_32FC1, cv::Scalar(0));
   Buffer = new cv::Mat();
}

/* Preprocessor
 *
 * Image will be resized to the fit size (no cropping)
 * @param image the input image
 * @param RGB2BGR swap channel blue and channel red
 */
inline void preprocess(const cv::Mat &image, bool RGB2BGR = false)
{
   // resize
   cv::resize(image, *Buffer, cv::Size(INPUT_SIZE[2], INPUT_SIZE[3]));
   // normalize & minus mean
   cv::dnn::blobFromImage(*Buffer, *Input_Mat,                    // in & out
                          (double)(1.0 / 255.0),                  // normalize (divides by 0xff)
                          cv::Size(INPUT_SIZE[2], INPUT_SIZE[3]), // useless param... I guess
                          cv::Scalar(MEAN[0], MEAN[1], MEAN[2]),  // minus the matrix by MEAN
                          RGB2BGR);                               // swap blue and red
   // devides by std
   for (int c = 0; c < INPUT_SIZE[1]; c++)   // walk the channels, c stands for channel
   {
      for (int j = 0; j < INPUT_SIZE[2]; j++)   // walk the rows, j stands for row
      {
         for (int k = 0; k < INPUT_SIZE[3]; k++)   // walk the columns, k stands for column
            Input_Mat->ptr<float>(0, c, j)[k] /= STD[c] / 255.0;
      }
   }
}

/* A softmax activation function
 * 
 * @param input the input matrix
 * @param output the output matrix
 */
inline void softmax(const cv::Mat &input, cv::Mat &output)
{
   double max = *std::max_element(input.begin<double>(), input.end<double>());
   cv::exp((input - max), output);
   output /= cv::sum(output)[0];
}

/* Predict based on the given image
 * Preprocesses the image and predict using the preprocessed image. 
 * The input image should be BGR, if not, set RGB2BGR true.
 * 
 * @param model the model object
 * @param image  input image, no need to fit the model
 * @param result result of the prediction will be stored in it
 * @param RGB2BGR if the input image is not in GBR but RGB, set this flag true
 * @return time cost by the prediction (if speed test is assembled)
 */
inline TIME_INTERVAL predict(const cv::Mat &image, cv::Mat &result, bool RGB2BGR = false)
{
SPEED_TESTER(
   // preprocess
   preprocess(image, RGB2BGR);
   // forward
   Model->setInput(*Input_Mat);
   softmax(Model->forward(), result);   // forward + softmax
) // SPEED_TESTER
   return TIME_COST;
}
