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
#define SPEED_TESTER(x)
#define TIME_COST
#endif // SPEED_TEST

/* Predict using input image. 
 * All parameters MUST have been allocated and initialized, this function will
 * NOT perform ANY check.
 * 
 * @param model point to the model object
 * @param blob preprocessed image
 * @param result result of the prediction will be stored in it
 * @return time cost by the prediction (if speed test is assembled)
 */
TIME_INTERVAL predict(cv::dnn::Net *model, cv::Mat *blob, cv::Mat *result)
{
SPEED_TESTER(
   double t = (double)cv::getTickCount();
   model->setInput(*blob);
   *result = net->forward();
) // SPEED_TESTER
   return TIME_COST;
}
