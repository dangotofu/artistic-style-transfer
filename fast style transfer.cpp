#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    // Load the input images
    Mat source = imread("source.jpg");
    Mat target = imread("target.jpg");

    // Resize the images to the size expected by the CNN model
    Size size(512, 512);
    resize(source, source, size);
    resize(target, target, size);

    // Convert the images to RGB format
    cvtColor(source, source, COLOR_BGR2RGB);
    cvtColor(target, target, COLOR_BGR2RGB);

    // Load the pre-trained CNN model
    Net net = readNetFromTorch("model.pt");

    // Set the input and output layers of the CNN
    net.setInput(blobFromImage(source, 1.0, size));
    Mat output = net.forward("output_layer");

    // Apply the style transfer to the target image
    Mat result;
    addWeighted(output, 1.0, target, 1.0, 0.0, result);

    // Convert the result back to BGR format and save it to a file
    cvtColor(result, result, COLOR_RGB2BGR);
    imwrite("result.jpg", result);

    return 0;
}
