/*
By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


                  License Agreement For libfacedetection
                     (3-clause BSD License)

Copyright (c) 2021, Joss Lei, all rights reserved.
josslei@163.com
josslei.0@outlook.com

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

#include <iostream>
#include <string>
#include <exception>
#include <opencv2/opencv.hpp>
#include <facedetection/facedetectcnn.h>
#include "../header/ClassifierModule.hpp"

using namespace std;
using namespace cv;
#define RESIZE_SCALE 4
//define the buffer size. Do not change the size!
#define DETECT_BUFFER_SIZE 0x20000
// buffer of libfacedetection
static unsigned char lfd_buffer[DETECT_BUFFER_SIZE];
const char *labels[] = { "Female", "Male" };

int main(int argc, char **argv)
{
    if (2 != argc)
        cout << "Please specify the path of the model!\n";
    try
    {
        Mat frame;                  // read from cap
        Mat lfd_input;              // input  of libfacedetection
        int *lfd_result = nullptr;  // result of lib facedetection
        string lfd_str_confidence;  // confidence in string
        Mat cm_input;
        Mat cm_result;              // result of Classifier Module
        string cm_str_result;       // result in string
        // Init the Classifier Module
        init_classifier(argv[1]);
        // Camera
        VideoCapture cap(0);
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        while (waitKey(2) != 27)   // esc
        {
            TickMeter cvtm;
            cvtm.start();
            // start
            cap >> frame;
            cv::flip(frame, frame, 1);
            // send into libfacedetection
            resize(frame, lfd_input, Size(1280 / RESIZE_SCALE, 720 / RESIZE_SCALE));
            // face detect
            lfd_result = lfd_result = facedetect_cnn(lfd_buffer, (unsigned char *)(lfd_input.ptr(0)), lfd_input.cols, lfd_input.rows, (int)lfd_input.step);
            // crop, classify, and draw
            for (int i = 0; i < (lfd_result ? *lfd_result : 0); i++)
            {
              short *p = ((short*)(lfd_result + 1)) + 142 * i;
              int confidence = p[0];
              int x = p[1] * RESIZE_SCALE;
              int y = p[2] * RESIZE_SCALE;
              int w = p[3] * RESIZE_SCALE;
              int h = p[4] * RESIZE_SCALE;
              cm_input = frame(Rect(x, y, w, h));
              // predict
              predict(cm_input, cm_result);
              if (cm_result.ptr<float>(0)[0] >= 0.55)       cm_str_result = labels[0];
              else if (cm_result.ptr<float>(0)[1] >= 0.55)  cm_str_result = labels[1];
              else                                          cm_str_result = "Could not identify!";
              // draw
              lfd_str_confidence = std::to_string(confidence);
              putText(frame, (lfd_str_confidence + ' ' + cm_str_result).c_str(), Point(x, y - 3), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
              // draw face rectangle
              rectangle(frame, Rect(x, y, w, h), Scalar(0, 255, 0), 2);
              // print the result
              // by Shiqi Yu (shiqi.yu@gmail.com), modified by Joss Lei
              printf("face %d: confidence=%d, gender=%s, [%d, %d, %d, %d] [%f, %f] (%d,%d) (%d,%d) (%d,%d) (%d,%d) (%d,%d)\n", 
                      i, confidence, cm_str_result.c_str(), x, y, w, h, cm_result.ptr<float>(0)[0], cm_result.ptr<float>(0)[1],
                      p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13],p[14]);

            }
            imshow("Cap", frame);
            // end
            cvtm.stop();
            cout << "Total time = " << cvtm.getTimeMilli() << "ms\n";
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    return 0;
}
