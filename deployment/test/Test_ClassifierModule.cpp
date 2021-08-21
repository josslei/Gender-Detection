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

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <string>
#include "../header/ClassifierModule.hpp"
using namespace std;
const char *labels[] = { "Female", "Male" };

int main(int argc, char **argv)
{
    if (2 != argc)
        cout << "Please specify the path of the model!\n";
    try {
        cv::Mat frame, result;
		string str_result;
        init_classifier(argv[1]);
        // camera
        cv::VideoCapture cap(0);
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        while (cv::waitKey(30) != 27) // esc
        {
            // read frame
            cap >> frame;
            // crop
            cv::Mat input = frame(cv::Rect(320, 40, 640, 640));
            // predict
            double time_used = predict(input, result);

            // display result
            cout << "Output: " << result << "\nTime used: " << time_used << '\n';
            if (result.ptr<float>(0)[0] >= 0.55)        str_result = labels[0];
            else if (result.ptr<float>(0)[1] >= 0.55)   str_result = labels[1];
            else                                        str_result = "Could not identify!";
            cv::flip(frame, frame, 1);
            cv::rectangle(frame, cv::Rect(320, 40, 640, 640), cv::Scalar(0xFFFFFFFF), 10);
            cv::putText(frame, str_result, cv::Point(10, 30), 0, 1.0, cv::Scalar(0xFFFFFFFF), 3);
            cv::imshow("Cap", frame);
        }
    }
    catch (const cv::Exception& e) {
        cout << e.what() << endl;
        return e.code;
    }
    return 0;
}
