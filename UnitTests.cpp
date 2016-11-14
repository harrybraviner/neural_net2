#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE NeuralNet_UnitTests
#include <boost/test/unit_test.hpp>
#include "NeuralNet.hpp"
#include "math.h"


BOOST_AUTO_TEST_CASE ( Constructor_does_not_throw )
{
    int layerCount[] = {10, 5, 15};
    FullyConnectedNeuralNet *nn = new FullyConnectedNeuralNet(1, layerCount);
    (void)nn;   // To avoid a compiler warning

    delete nn;
}

BOOST_AUTO_TEST_CASE ( Feed_forward_on_new_net_gives_equal_probs )
{
    int layerCount[] = {10, 5, 15};
    FullyConnectedNeuralNet *nn = new FullyConnectedNeuralNet(1, layerCount);
    double input[] = {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0};
    double expected_output[] = {1.0/15.0, 1.0/15.0, 1.0/15.0, 1.0/15.0, 1.0/15.0, 1.0/15.0, 1.0/15.0, 1.0/15.0, 1.0/15.0, 1.0/15.0, 1.0/15.0, 1.0/15.0, 1.0/15.0, 1.0/15.0, 1.0/15.0};
    double *actual_output = new double[15];
    nn->SetInput(input);
    nn->ForwardPropogate();
    nn->GetOutput(actual_output);
    BOOST_CHECK_EQUAL_COLLECTIONS( expected_output, expected_output + 15, actual_output, actual_output + 15 );

    delete nn;
}

BOOST_AUTO_TEST_CASE ( Set_matrices_and_check_feed_forward )
{
    auto sig = [](double x) { return 1.0/(1.0 + exp(-x)); };

    double m1[] = { 0.0, 1.0, 0.0, 0.0,
                   -3.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    5.0, 1.0, 1.0, 1.0};
    double m2[] = {0.0, 0.0, 1.0, 1.0, 0.0,
                   6.0, 1.0, 1.0, 0.0, 0.0};
    int layerCount[] = {3, 4, 2};

    FullyConnectedNeuralNet *nn = new FullyConnectedNeuralNet(1, layerCount);
    nn->SetMatrix(1, m1);
    nn->SetMatrix(2, m2);

    double input[] = {1.0, -0.1, 2.0, 1.0};
    double z2_0 = sig(-3.0) + sig(-0.1);
    double z2_1 = 6.0 + sig(1.0) + sig(-3.0);
    double *expected_output = new double[2];
    expected_output[0] = exp(z2_0) / (exp(z2_0) + exp(z2_1));
    expected_output[1] = exp(z2_1) / (exp(z2_0) + exp(z2_1));
    double *actual_output = new double[2];
    nn->SetInput(input);
    nn->ForwardPropogate();
    nn->GetOutput(actual_output);

    BOOST_CHECK_EQUAL_COLLECTIONS( expected_output, expected_output + 2, actual_output, actual_output + 2);

    delete nn;
}

BOOST_AUTO_TEST_CASE ( Compare_numeric_derivs_to_backpropogation_zero_matrices )
{
    double delta = 1e-3; // Step size for numeric derivative
    double epsilon = 1e-7;  // Tolerance for difference

    // Some matrices with zero entries
    double m1[] = { 0.0,  0.0, 0.0,  0.0,
                    0.0,  0.0, 0.0,  0.0,
                    0.0,  0.0, 0.0,  0.0,
                    0.0,  0.0, 0.0,  0.0};
    double m2[] = {0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0};
    int layerCount[] = {3, 4, 2};

    double y[] = {1.0, 0.0};
    double input[] = {0.0, 0.0, 0.0};

    FullyConnectedNeuralNet *nn = new FullyConnectedNeuralNet(1, layerCount);

    auto compute_numeric_deriv = [m1, m2, delta, nn, input, y](int row, int col, int layer) {
        double *m1_copy = new double[4*4];
        std::memcpy(m1_copy, m1, sizeof(double)*4*4);
        double *m2_copy = new double[2*5];
        std::memcpy(m2_copy, m2, sizeof(double)*2*5);

        if (layer==1) {
            m1_copy[4*row + col] += delta;
        } else {
            m2_copy[5*row + col] += delta;
        }
        nn->SetMatrix(1, m1_copy);
        nn->SetMatrix(2, m2_copy);

        nn->SetInput(input);
        nn->ForwardPropogate();
        double *y_plus = new double[2];
        nn->GetOutput(y_plus);
        double C_plus = 0.0;
        for(int i = 0; i<2; i++) {
            C_plus += -1.0*y[i]*log(y_plus[i]);
        }

        if (layer==1) {
            m1_copy[4*row + col] -= 2.0*delta;
        } else {
            m2_copy[5*row + col] -= 2.0*delta;
        }
        nn->SetMatrix(1, m1_copy);
        nn->SetMatrix(2, m2_copy);

        nn->SetInput(input);
        nn->ForwardPropogate();
        double *y_minus = new double[2];
        nn->GetOutput(y_minus);
        double C_minus = 0.0;
        for(int i = 0; i<2; i++) {
            C_minus += -1.0*y[i]*log(y_minus[i]);
        }

        return (C_plus - C_minus)/(2.0*delta);
    };

    nn->SetMatrix(1, m1);
    nn->SetMatrix(2, m2);
    nn->SetInput(input);
    nn->ForwardPropogate();
    nn->BackPropogate(0);

    double *w1_derivs = new double[4*4];
    double *w2_derivs = new double[2*5];
    nn->GetMatrixDerivative(1, w1_derivs);
    nn->GetMatrixDerivative(2, w2_derivs);

    double maxDiff = 0.0;

    for (int row=0; row<4; row++) {
        for (int col=0; col<4; col++) {
            double bp_result = w1_derivs[row*4 + col];
            double numeric_result = compute_numeric_deriv(row, col, 1);
            double diff = abs(bp_result - numeric_result);
            maxDiff = std::max(maxDiff, diff);
            //printf("Backprop:\t%f\nNumeric:\t%f\n\n", bp_result, numeric_result);
        }
    }


    for (int row=0; row<2; row++) {
        for (int col=0; col<5; col++) {
            double bp_result = w2_derivs[row*5 + col];
            double numeric_result = compute_numeric_deriv(row, col, 2);
            double diff = abs(bp_result - numeric_result);
            maxDiff = std::max(maxDiff, diff);
            //printf("Backprop:\t%f\nNumeric:\t%f\n\n", bp_result, numeric_result);
        }
    }

    BOOST_CHECK( maxDiff <= epsilon );
}


BOOST_AUTO_TEST_CASE ( Compare_numeric_derivs_to_backpropogation )
{
    double delta = 1e-3; // Step size for numeric derivative
    double epsilon = 1e-7;  // Tolerance for difference

    // Some matrices with arbitrary non-zero entries
    double m1[] = { 0.2,  1.0, -0.1,  0.4,
                   -3.0,  0.1,  2.0,  0.2,
                   -0.5, 0.25, -1.0, -0.6,
                    5.0, -0.7,  1.2,  1.0};
    double m2[] = {0.1, 0.2, 1.0, 1.0, -0.4,
                   6.0, 1.0, 1.6, 0.3, -0.2};
    int layerCount[] = {3, 4, 2};

    double y[] = {0.0, 1.0};
    double input[] = {0.0, 0.0, 0.0};

    FullyConnectedNeuralNet *nn = new FullyConnectedNeuralNet(1, layerCount);

    auto compute_numeric_deriv = [m1, m2, delta, nn, input, y](int row, int col, int layer) {
        double *m1_copy = new double[4*4];
        std::memcpy(m1_copy, m1, sizeof(double)*4*4);
        double *m2_copy = new double[2*5];
        std::memcpy(m2_copy, m2, sizeof(double)*2*5);

        if (layer==1) {
            m1_copy[4*row + col] += delta;
        } else {
            m2_copy[5*row + col] += delta;
        }
        nn->SetMatrix(1, m1_copy);
        nn->SetMatrix(2, m2_copy);

        nn->SetInput(input);
        nn->ForwardPropogate();
        double *y_plus = new double[2];
        nn->GetOutput(y_plus);
        double C_plus = 0.0;
        for(int i = 0; i<2; i++) {
            C_plus += -1.0*y[i]*log(y_plus[i]);
        }

        if (layer==1) {
            m1_copy[4*row + col] -= 2.0*delta;
        } else {
            m2_copy[5*row + col] -= 2.0*delta;
        }
        nn->SetMatrix(1, m1_copy);
        nn->SetMatrix(2, m2_copy);

        nn->SetInput(input);
        nn->ForwardPropogate();
        double *y_minus = new double[2];
        nn->GetOutput(y_minus);
        double C_minus = 0.0;
        for(int i = 0; i<2; i++) {
            C_minus += -1.0*y[i]*log(y_minus[i]);
        }

        return (C_plus - C_minus)/(2.0*delta);
    };

    nn->SetMatrix(1, m1);
    nn->SetMatrix(2, m2);
    nn->SetInput(input);
    nn->ForwardPropogate();
    nn->BackPropogate(1);

    double *w1_derivs = new double[4*4];
    double *w2_derivs = new double[2*5];
    nn->GetMatrixDerivative(1, w1_derivs);
    nn->GetMatrixDerivative(2, w2_derivs);

    double maxDiff = 0.0;

    for (int row=0; row<4; row++) {
        for (int col=0; col<4; col++) {
            double bp_result = w1_derivs[row*4 + col];
            double numeric_result = compute_numeric_deriv(row, col, 1);
            double diff = abs(bp_result - numeric_result);
            maxDiff = std::max(maxDiff, diff);
            //printf("Backprop:\t%f\nNumeric:\t%f\n\n", bp_result, numeric_result);
        }
    }


    for (int row=0; row<2; row++) {
        for (int col=0; col<5; col++) {
            double bp_result = w2_derivs[row*5 + col];
            double numeric_result = compute_numeric_deriv(row, col, 2);
            double diff = abs(bp_result - numeric_result);
            maxDiff = std::max(maxDiff, diff);
            //printf("Backprop:\t%f\nNumeric:\t%f\n\n", bp_result, numeric_result);
        }
    }

    BOOST_CHECK( maxDiff <= epsilon );
}
