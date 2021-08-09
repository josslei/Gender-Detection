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

/*
int main()
{
    cv::Mat frame;
    // camera
    cv::VideoCapture cap(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    while (cv::waitKey(30) != 27) // esc
    {
        cap >> frame;
        cv::Mat cropped = cv::dnn::blobFromImage(frame, (double)(1.0 / 255.0), cv::Size(200, 200), cv::Scalar(0.485 * 255, 0.456 * 255, 0.406 * 255));
        cv::imshow("Null", frame);
        cout << cropped.step1(0) << ' ' <<
                cropped.step1(1) << ' ' <<
                cropped.step1(2) << ' ' <<
                cropped.step1(3) << endl;
        cout << cropped.ptr<float>(0, 2, 199)[199] << endl;
    }
}
*/
