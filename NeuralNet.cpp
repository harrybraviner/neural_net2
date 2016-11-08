#include <cstring>
#include <random>
#include "math.h"

#include "NeuralNet.hpp"

FullyConnectedNeuralNet::FullyConnectedNeuralNet(int const numberOfHiddenLayers, int const numberOfNodesInLayers[])
{
    L = numberOfHiddenLayers+1;

    this->numberOfNodesInLayers = new int[numberOfHiddenLayers + 2];
    std::memcpy(this->numberOfNodesInLayers, numberOfNodesInLayers, sizeof(int)*(numberOfHiddenLayers + 2));

    // Compute the number of weights we need in total
    totalNumberOfWeights = 0;
    for (int l=1; l<=L; l++) {
        int cols = numberOfNodesInLayers[l-1]+1;
        int rows = numberOfNodesInLayers[l];
        totalNumberOfWeights += cols*rows;
    }
    // Assign contiguous memory to hold the weights
    allWeights = new double[totalNumberOfWeights]();
    // ...and there derivatives
    allWeightDerivs = new double[totalNumberOfWeights];

    W = new double*[L+1];
    W[0] = nullptr;
    dEdW = new double*[L+1];
    dEdW[0] = nullptr;

    int memCounter = 0;
    for (int l=1; l<=L; l++) {
        W[l] = &allWeights[memCounter];
        dEdW[l] = &allWeightDerivs[memCounter];
        int cols = numberOfNodesInLayers[l-1]+1;
        int rows = numberOfNodesInLayers[l];
        memCounter += cols*rows;
    }

    x = new double*[L+1];
    z = new double*[L+1];
    for (int i=0; i<=L; i++) x[i] = new double[numberOfNodesInLayers[i]];
    // Note the loop starting at 1, since there is no 'input to' layer 0
    z[0] = nullptr;
    for (int i=1; i<=L; i++) z[i] = new double[numberOfNodesInLayers[i]];

    activationFunction = new layerFunc_t[L+1];
    activationFunction[0] = nullptr;
    for (int l=1; l<L; l++) activationFunction[l] = &Sigmoid;
    activationFunction[L] = &Softmax;

    dEdz = new double*[L+1];
    dEdz[0] = nullptr;
    for (int l=1; l<=L; l++) {
        dEdz[l] = new double[numberOfNodesInLayers[l]];
    }
}

FullyConnectedNeuralNet::~FullyConnectedNeuralNet()
{
    delete[] allWeights;
    delete[] W;
    delete[] allWeightDerivs;
    delete[] dEdW;

    for (int i=0; i<L; i++) delete[] x[i];
    for (int i=1; i<L; i++) delete[] z[i];
    delete [] x; delete[] z;

    delete [] activationFunction;
    for (int l=1; l<=L; l++) delete [] dEdz[l];
    delete [] dEdz;
}

void FullyConnectedNeuralNet::RandomiseWeights()
{
    std::uniform_real_distribution<double> unif(-0.3, 0.3);
    std::default_random_engine re;
    for (int i=0; i<totalNumberOfWeights; i++){
        allWeights[i] = unif(re);
    }
}

void FullyConnectedNeuralNet::SetInput(const double *input)
{
    std::memcpy(x[0], input, sizeof(double)*numberOfNodesInLayers[0]);
}

void FullyConnectedNeuralNet::GetOutput(double *output)
{
    std::memcpy(output, x[L], sizeof(double)*numberOfNodesInLayers[L]);
}

void FullyConnectedNeuralNet::SetMatrix(int layer, const double *input)
{
    int rows = numberOfNodesInLayers[layer];
    int cols = numberOfNodesInLayers[layer-1]+1;
    std::memcpy(W[layer], input, sizeof(double)*rows*cols);
}

void FullyConnectedNeuralNet::GetMatrix(int layer, double *output)
{
    int rows = numberOfNodesInLayers[layer];
    int cols = numberOfNodesInLayers[layer-1]+1;
    std::memcpy(output, W[layer], sizeof(double)*rows*cols);
}

void FullyConnectedNeuralNet::GetMatrixDerivative(int layer, double *output)
{
    int rows = numberOfNodesInLayers[layer];
    int cols = numberOfNodesInLayers[layer-1]+1;
    std::memcpy(output, dEdW[layer], sizeof(double)*rows*cols);
}

void FullyConnectedNeuralNet::ForwardPropogate()
{
    for (int l=1; l<=L; l++) {
        // Compute the input to layer l
        int rows = numberOfNodesInLayers[l];
        int cols = numberOfNodesInLayers[l-1]+1;
        double *z = this->z[l];
        double *x = this->x[l-1];
        double *W = this->W[l];
        for (int row=0; row<rows; row++) {
            z[row] = W[row*cols];
            for (int col=1; col<cols; col++) {
                z[row] += W[row*cols + col]*x[col-1];
            }
        }

        // Compute the output from layer l
        activationFunction[l](rows, z, this->x[l]);
    }
}

void FullyConnectedNeuralNet::BackPropogate(int target)
{
    // Here we take the error function to be the cross entropy
    // We do assume sigmoid, sigmoid, ..., sigmoid, softmax mappings

    // Populate last layer - special case
    int rows = numberOfNodesInLayers[L];
    for (int i=0; i<rows; i++) {
        if (target == i)
            dEdz[L][i] = x[L][i] - 1.0;
        else
            dEdz[L][i] = x[L][i];
    }
    // Back propogate the derivative with respect to layer inputs
    for (int l=L-1; l>0; l--) {
        int cols = numberOfNodesInLayers[l] + 1;
        int rows = numberOfNodesInLayers[l+1];
        for (int i=0; i<cols-1; i++) {
            dEdz[l][i] = 0.0;
            for (int j=0; j<rows; j++) {
                dEdz[l][i] += dEdz[l+1][j]*W[l+1][j*cols + (i+1)];
            }
            dEdz[l][i] *= x[l][i]*(1.0 - x[l][i]);
        }
    }
    
    // Compute the derivatives with respect to the weights
    for (int l=L; l>0; l--) {
        int rows = numberOfNodesInLayers[l];
        int cols = numberOfNodesInLayers[l-1] + 1;
        for (int i=0; i<rows; i++) {
            dEdW[l][i*cols + 0] = dEdz[l][i];
            for (int j=1; j<cols; j++) {
                dEdW[l][i*cols + j] = dEdz[l][i]*x[l-1][j-1];
            }
        }
    }
}

void FullyConnectedNeuralNet::Sigmoid(int n, double *input, double *output)
{
    for (int i=0; i<n; i++) output[i] = 1.0 / (1.0 + exp(-input[i]));
}

void FullyConnectedNeuralNet::Softmax(int n, double *input, double *output)
{
    double sum = 0.0;
    for (int i=0; i<n; i++){
        output[i] = exp(input[i]);
        sum += output[i];
    }

    for (int i=0; i<n; i++) output[i] /= sum;
}
