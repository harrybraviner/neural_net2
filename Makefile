CPP=g++
CPPFLAGS=-Wall -std=c++17

UnitTests : UnitTests.cpp NeuralNet.o NeuralNet.hpp
	${CPP} ${CPPFLAGS} -lboost_unit_test_framework -o UnitTests UnitTests.cpp NeuralNet.o

NeuralNet.o : NeuralNet.cpp NeuralNet.hpp
	${CPP} ${CPPFLAGS} -c $<
