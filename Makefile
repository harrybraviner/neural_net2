CPP=g++
CPPFLAGS=-Wall -std=c++17 -g

MNISTTest : MNISTTest.cpp NeuralNet.o NeuralNet.hpp MomentumIntegrator.o MomentumIntegrator.hpp
	${CPP} ${CPPFLAGS} -o MNISTTest MNISTTest.cpp NeuralNet.o MomentumIntegrator.o

UnitTests : UnitTests.cpp NeuralNet.o NeuralNet.hpp MomentumIntegrator.o
	${CPP} ${CPPFLAGS} -lboost_unit_test_framework -o UnitTests UnitTests.cpp NeuralNet.o MomentumIntegrator.o

NeuralNet.o : NeuralNet.cpp NeuralNet.hpp
	${CPP} ${CPPFLAGS} -c $<

MomentumIntegrator.o : MomentumIntegrator.cpp MomentumIntegrator.hpp Integrator.hpp
	${CPP} ${CPPFLAGS} -c $<

clean :
	rm -f *.o UnitTests
