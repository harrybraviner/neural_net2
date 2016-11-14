#ifndef _FULLYCONNECTEDNEURALNET_HPP_
#define _FULLYCONNECTEDNEURALNET_HPP_

#include "Integrator.hpp"
#include "MomentumIntegrator.hpp"

class FullyConnectedNeuralNet {
    public:
        FullyConnectedNeuralNet(int const numberOfHiddenLayers, int const numberOfNodesInLayers[]);
        ~FullyConnectedNeuralNet();
        int GetNumberOfHiddenLayers();

        void RandomiseWeights();
        void SetInput(const double *input);
        void GetOutput(double *output);
        double GetCrossEntropy(const int target);
        void ForwardPropogate();
        void BackPropogate(int target);

        void SetMatrix(int layer, const double *input);
        void GetMatrix(int layer, double *output);
        void GetMatrixDerivative(int layer, double *output);

        static void Sigmoid(int n, double *x, double *z);
        static void Softmax(int n, double *x, double *z);

        double BatchTrain(int N, const double *inputs, const int *targets);

    private:
        typedef void(*layerFunc_t)(int, double*, double*);

        int L;  // Index of the last layer
        // Note: layers are indexed 0, 1, ..., L
        // 0 is is the input layer
        // L is the output layer
        int totalNumberOfWeights;   // The number of weights in the entire net
        double *allWeights;    // Memory that will hold the weights
        double **W;    // Weights for each layer
        int *numberOfNodesInLayers;

        double *allWeightDerivs; // Derivative of loss function with respect to weights
        double **dEdW;  // As above, but per layer
        double *batchWeightsDeriv;    // Holds mean derivative in batch learning

        double **x; // Outputs from each layer
        double **z; // Inputs to each layer
        layerFunc_t *activationFunction; // z -> x

        double **dEdz; // Used in backpropogation

        Integrator *integrator; // Used to update the weights
};

// ToDo:
// * Batch gradient computation
// * Incorporate momentum integrator

#endif
