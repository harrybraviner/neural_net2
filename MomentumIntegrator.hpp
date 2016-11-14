#ifndef _MOMENTUMINTEGRATOR_HPP_
#define _MOMENTUMINTEGRATOR_HPP_

#include "Integrator.hpp"

class MomentumIntegrator : public Integrator {
    private:
        double *x;
        double *v;
        double *a;
        int N;
        double h;
        double momentum;

    public:
        MomentumIntegrator(int N, double h, double momentum, double *x, double *a);
        ~MomentumIntegrator();
        void Step();

};

#endif
