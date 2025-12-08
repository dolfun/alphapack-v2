#include <pybind11/pybind11.h>

PYBIND11_MODULE(alphapack, m) {
  m.def("multiply", [](int a, int b) { return a * b; });
}