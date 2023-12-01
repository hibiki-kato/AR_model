#include <iostream>
#include <string>
#include <sstream>
#include "cnpy/cnpy.h"

int main() {
    // parameters
    double r = 3.9;
    int step = 1e+6;
    int dump = 5e+4;
    double x = 0.5;

    double vec[step + 1];

    for (int i = 0; i < dump; ++i) {
        x = r * x * (1 - x);
    }
    vec[0] = x;
    for (int i = 1; i < step + 1; ++i) {
        x = r * x * (1 - x);
        vec[i] = x;
    }

    auto oss = std::ostringstream();
    oss << "../../logistic/logistic_" << r << "_" << step <<"_" << dump <<"dumped.npy";
    std::string fname = oss.str();

    cnpy::npy_save(fname, vec, {(unsigned long)(step + 1)}, "w");
    std::cout << "saved to " << fname << std::endl;
    return 0;
    
}