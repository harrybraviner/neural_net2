#ifndef _INTEGRATOR_HPP_
#define _INTEGRATOR_HPP_

class Integrator {
    public:
        virtual void Step()=0;
        virtual ~Integrator() {};
};

#endif
