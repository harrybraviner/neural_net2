#include <iostream>
#include <cstring>
#include <fstream>
#include <chrono>

#include "NeuralNet.hpp"

using namespace std;

void read_image_from_file (unsigned char* pixels, ifstream file) {
    file.read((char*)pixels, 28*28);
}

void train_batch_on_range_of_images (int minIndex, int maxIndex, double learningRate, FullyConnectedNeuralNet *nn, double* mean_image) {
    // Open the MNIST files for reading
    ifstream mnist_images("data/train-images-idx3-ubyte", ios::in|ios::binary|ios::ate);
    ifstream mnist_labels("data/train-labels-idx1-ubyte", ios::in|ios::binary|ios::ate);

    char *intBytes = new char[4];
    char *intBytesRev = new char[4];

    mnist_images.seekg(4, ios::beg);
    mnist_images.read(intBytes, 4);
    for (int i=0; i<4; i++) {intBytesRev[i] = intBytes[3-i];}
    int numberOfTrainingImages = *((int*)intBytesRev);

    mnist_labels.seekg(4, ios::beg);
    mnist_labels.read(intBytes, 4);
    for (int i=0; i<4; i++) {intBytesRev[i] = intBytes[3-i];}
    int numberOfTrainingLabels = *((int*)intBytesRev);

    if (numberOfTrainingImages != numberOfTrainingLabels) {
        cerr << "WARNING: Found " << numberOfTrainingImages << " training images and "
             << numberOfTrainingLabels << " training labels. Clearly your datasets do not match!\n";
    }

    if (maxIndex > numberOfTrainingImages-1){
        cerr << "WARNING: maxIndex is greater than " << (numberOfTrainingImages-1) << ", the greatest training index available!\n";
    }

    int numTrainingCasesToDo = maxIndex - minIndex + 1;

    mnist_images.seekg(4*4 + minIndex*28*28, ios::beg);
    mnist_labels.seekg(2*4 + minIndex, ios::beg);

    auto t1 = chrono::high_resolution_clock::now();

    unsigned char *pixels = new unsigned char[28*28];
    unsigned char label;

    double *all_input = new double[numTrainingCasesToDo*28*28];
    int *all_target = new int[numTrainingCasesToDo];

    for (int i=0; i< numTrainingCasesToDo; i++) {
        mnist_images.read((char*)pixels, 28*28);
        mnist_labels.read((char*)&label, 1);
        
        double *input = &all_input[i*28*28];
        for (int j=0; j<28*28; j++) { input[j] = (double) pixels[j] / 255.0 - mean_image[j]; }

        all_target[i] = (int)label;
    }

    double CE = nn->BatchTrain(numTrainingCasesToDo, all_input, all_target);
    cout << "Before training, cross-entropy was: " << CE << "\n";

    auto t2 = chrono::high_resolution_clock::now();
    int elapsedMilliseconds = chrono::duration_cast<chrono::milliseconds>(t2-t1).count();

    cout << "Spent " << elapsedMilliseconds/1000.0 << " seconds training on "
         << numTrainingCasesToDo << " images. (" << (elapsedMilliseconds/numTrainingCasesToDo) << " ms per image.)\n";


    mnist_images.close();
    mnist_labels.close();
}

void get_mean_image(double *output) {
    ifstream mnist_images("data/train-images-idx3-ubyte", ios::in|ios::binary|ios::ate);

    char *intBytes = new char[4];
    char *intBytesRev = new char[4];

    mnist_images.seekg(4, ios::beg);
    mnist_images.read(intBytes, 4);
    for (int i=0; i<4; i++) {intBytesRev[i] = intBytes[3-i];}
    int numberOfTrainingImages = *((int*)intBytesRev);

    mnist_images.seekg(4*4, ios::beg);

    unsigned char *pixels = new unsigned char[28*28];
    for (int i=0; i<numberOfTrainingImages; i++) {
        mnist_images.read((char*)pixels, 28*28);
        for (int j=0; j<28*28; j++)
            output[j] += (double) pixels[j] / 255.0;
    }

    for (int j=0; j<28*28; j++)
        output[j] /= numberOfTrainingImages;

    mnist_images.close();
}

void get_error_rate_on_range_of_images(int minIndex, int maxIndex, FullyConnectedNeuralNet *nn, double* mean_image) {
    // Open the MNIST files for reading
    ifstream mnist_images("data/train-images-idx3-ubyte", ios::in|ios::binary|ios::ate);
    ifstream mnist_labels("data/train-labels-idx1-ubyte", ios::in|ios::binary|ios::ate);

    char *intBytes = new char[4];
    char *intBytesRev = new char[4];

    mnist_images.seekg(4, ios::beg);
    mnist_images.read(intBytes, 4);
    for (int i=0; i<4; i++) {intBytesRev[i] = intBytes[3-i];}
    int numberOfTrainingImages = *((int*)intBytesRev);

    mnist_labels.seekg(4, ios::beg);
    mnist_labels.read(intBytes, 4);
    for (int i=0; i<4; i++) {intBytesRev[i] = intBytes[3-i];}
    int numberOfTrainingLabels = *((int*)intBytesRev);

    if (numberOfTrainingImages != numberOfTrainingLabels) {
        cerr << "WARNING: Found " << numberOfTrainingImages << " images and "
             << numberOfTrainingLabels << " labels. Clearly your datasets do not match!\n";
    }

    if (maxIndex > numberOfTrainingImages-1){
        cerr << "WARNING: maxIndex is greater than " << (numberOfTrainingImages-1) << ", the greatest index available!\n";
    }

    int numClassificationsToDo = maxIndex - minIndex + 1;

    mnist_images.seekg(4*4 + minIndex*28*28, ios::beg);
    mnist_labels.seekg(2*4 + minIndex, ios::beg);

    auto t1 = chrono::high_resolution_clock::now();

    unsigned char *pixels = new unsigned char[28*28];
    unsigned char label;
    double *input = new double[28*28];
    double *output = new double[10];
    unsigned char outputLabel;
    double maxLabelAcc;
    unsigned int correct=0, incorrect=0;
    int *classificationTable = new int[10*10]();
    for (int i=0; i< numClassificationsToDo; i++) {
        mnist_images.read((char*)pixels, 28*28);
        mnist_labels.read((char*)&label, 1);
        
        for (int j=0; j<28*28; j++) { input[j] = (double) pixels[j] / 255.0 - mean_image[j]; }

        //cout << "Classifying image " << (i + minIndex) << "\n";
        nn->SetInput(input);
        nn->ForwardPropogate();
        nn->GetOutput(output);

        maxLabelAcc = output[0];
        outputLabel = 0;
        for (int j=1; j<10; j++) {
            if (output[j] > maxLabelAcc) {
                maxLabelAcc = output[j];
                outputLabel = j;
            }
        }

        classificationTable[label*10 + outputLabel] += 1;

        if (outputLabel == label) {
            //cout << "Correctly classified " << (int)label << "\n";
            correct++;
        } else {
            //cout << "Incorrectly classified as " << (int)outputLabel << ". "
            //     << "True label was " << (int)label << "\n";
            incorrect++;
        }
        
    }

    delete output;

    auto t2 = chrono::high_resolution_clock::now();
    int elapsedMilliseconds = chrono::duration_cast<chrono::milliseconds>(t2-t1).count();

    cout << "Spent " << elapsedMilliseconds/1000.0 << " seconds classifying "
         << numClassificationsToDo << " images. (" << (elapsedMilliseconds/numClassificationsToDo) << " ms per image.)\n";


    double error_rate = ((double)incorrect) / ((double) (correct + incorrect));
    cout << "Error rate was " << (error_rate*100.0) << "%\n";

    cout << "\t\t\tClassification\n";
    cout << "Label\t0\t1\t2\t3\t4\t5\t6\t7\t8\t9\n";
    for (int label=0; label<10; label++) {
        cout << label << "\t";
        for (int classification=0; classification<10; classification++) {
            cout << classificationTable[label*10 + classification] << "\t";
        }
        cout << "\n";
    }

    mnist_images.close();
    mnist_labels.close();
}

int main() {

    int numberOfInputs = 28*28;
    int numberOfOutputs = 10;
    int numberOfHiddenLayers = 1;
    int numberOfNodesInLayers[] = {numberOfInputs, 100, numberOfOutputs};
    
    FullyConnectedNeuralNet *nn = new FullyConnectedNeuralNet(numberOfHiddenLayers, numberOfNodesInLayers);
    nn->RandomiseWeights();

    double *mean_image = new double[28*28];
    get_mean_image(mean_image);

    //get_error_rate_on_range_of_images(0, 1000, nn, mean_image);

    double learningRate = 0.01;
    for (int i=0; i<30; i++) {
        for (int j=0; j<100; j++) {
            //train_batch_on_range_of_images(1001 + 100*j, 1100 + 100*j, learningRate, nn, mean_image);
            train_batch_on_range_of_images(1001 + j*100, 1100 + j*100, learningRate, nn, mean_image);
        }
        get_error_rate_on_range_of_images(0, 1000, nn, mean_image);
    }



    return 0;
}
