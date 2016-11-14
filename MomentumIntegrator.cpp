#include "MomentumIntegrator.hpp"

MomentumIntegrator::MomentumIntegrator(int N, double h, double momentum, double *x, double *a)
{
    this->x = x;
    this->a = a;
    v = new double[N]();
    this->N = N;
    this->h = h;
    this->momentum = momentum;
}

MomentumIntegrator::~MomentumIntegrator()
{
    delete v;
}

void MomentumIntegrator::Step()
{
    for (int i=0; i<N; i++) {
        v[i] = momentum*v[i] - a[i];
        x[i] += h*v[i];
    }
}
