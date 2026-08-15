# NanoJAX
NanoJAX is an autodiff (automatic differentiation) library for Python supporting calculations on the GPU using CUDA. It has a C++ core to support fast autodiff while maintaining the flexibility, simplicity, and development speed of Python. A C++ interface is also exposed (see main.cpp), which can be use as well. In fact, the Python interface leverages this C++ interface through bindings. In addition, there is also a small machine learning library featuring certain layers like MLPs, CNNs, and activation functions such as ReLU, Sigmoid, and more.



# Work in Progress
Currently, NanoJAX is a work in progress. The Python bindings for the C++ core have not yet been implemented and neither has CUDA support.
