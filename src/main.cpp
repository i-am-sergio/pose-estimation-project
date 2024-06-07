#include <iostream>
#include <opencv2/opencv.hpp>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/optional_debug_tools.h>

using namespace cv;
using namespace std;

Mat resizeWithPadding(const Mat& src, Size targetSize) {
    int originalWidth = src.cols;
    int originalHeight = src.rows;
    float aspectRatio = static_cast<float>(originalWidth) / originalHeight;

    int targetWidth = targetSize.width;
    int targetHeight = targetSize.height;
    float targetAspectRatio = static_cast<float>(targetWidth) / targetHeight;

    int newWidth, newHeight;
    if (aspectRatio > targetAspectRatio) {
        newWidth = targetWidth;
        newHeight = static_cast<int>(targetWidth / aspectRatio);
    } else {
        newHeight = targetHeight;
        newWidth = static_cast<int>(targetHeight * aspectRatio);
    }

    Mat resizedImage;
    resize(src, resizedImage, Size(newWidth, newHeight));

    int top = (targetHeight - newHeight) / 2;
    int bottom = targetHeight - newHeight - top;
    int left = (targetWidth - newWidth) / 2;
    int right = targetWidth - newWidth - left;

    Mat paddedImage;
    copyMakeBorder(resizedImage, paddedImage, top, bottom, left, right, BORDER_CONSTANT, Scalar(0, 0, 0));

    return paddedImage;
}

int main() {
    // Ruta de la imagen y del modelo
    string image_path = "img1.png";
    string model_path = "movenet_lightning_fp16.tflite";

    // Leer la imagen
    Mat image = imread(image_path);
    if (image.empty()) {
        cerr << "Could not read the image: " << image_path << endl;
        return 1;
    }

    // Redimensionar la imagen a 192x192 con padding
    Mat input_image = resizeWithPadding(image, Size(192, 192));

    // Convertir la imagen a uint8
    input_image.convertTo(input_image, CV_8UC3);

    // Cargar el modelo TFLite
    unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    tflite::ops::builtin::BuiltinOpResolver resolver;
    unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);

    interpreter->AllocateTensors();

    // Obtener detalles del tensor de entrada
    int input = interpreter->inputs()[0];
    int output = interpreter->outputs()[0];

    // Preparar los datos de entrada
    uint8_t* input_tensor = interpreter->typed_tensor<uint8_t>(input);
    memcpy(input_tensor, input_image.data, input_image.total() * input_image.elemSize());

    // Ejecutar la inferencia
    interpreter->Invoke();

    // Obtener los resultados de salida
    float* output_data = interpreter->typed_output_tensor<float>(0);
    
    // print keypoints with format (y, x, score)
    for (int i = 0; i < 17; i++) {
        cout << output_data[i * 3] << ", " << output_data[i * 3 + 1] << ", " << output_data[i * 3 + 2] << endl;
    }


    // Redimensionar la imagen para visualización
    int width = 640;
    int height = 640;
    Mat display_image = resizeWithPadding(image, Size(width, height));

    // Dibujar los keypoints en la imagen
    vector<Point> keypoints(17);
    for (int i = 0; i < 17; ++i) { // 17 keypoints for MoveNet
        int y = static_cast<int>(output_data[i * 3] * height);
        int x = static_cast<int>(output_data[i * 3 + 1] * width);
        keypoints[i] = Point(x, y);
        circle(display_image, Point(x, y), 4, Scalar(0, 0, 255), -1);
    }

    // Definir las conexiones entre los keypoints
    vector<pair<int, int>> KEYPOINT_EDGES = {
        {0, 1}, {0, 2}, {1, 3}, {2, 4}, {0, 5}, {0, 6}, {5, 7},
        {7, 9}, {6, 8}, {8, 10}, {5, 6}, {5, 11}, {6, 12}, {11, 12}, {11, 13},
        {13, 15}, {12, 14}, {14, 16}
    };

    // Dibujar las líneas entre los keypoints
    for (const auto& edge : KEYPOINT_EDGES) {
        line(display_image, keypoints[edge.first], keypoints[edge.second], Scalar(0, 255, 0), 2);
    }

    // Mostrar la imagen con los keypoints
    imshow("pose estimation", display_image);
    waitKey(0);

    return 0;
}
