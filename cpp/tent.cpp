#include <iostream>
#include <string>
#include <sstream>
#include <cmath>
#include <random>
#include "cnpy/cnpy.h"

int main() {
    // parameters
    double r = 2;
    int step = 1e+2;
    int dump = 0;
    double x = 0.01;

    double vec[step + 1];
    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    std::normal_distribution<> dist(0, 1e-10);

    for (int i = 0; i < dump; ++i) {
        x = r * x;
        x += dist(engine);
        if (x >= 1) {
            x -= 1.0;
        }
        std::cout << x << std::endl;
    }
    vec[0] = x;
    std::cout << x << std::endl;
    for (int i = 1; i < step + 1; ++i) {
        x = r * x;
        x += dist(engine);
        if (x >= 1) {
            x -= 1.0;
        }
        vec[i] = x;

    }

    auto oss = std::ostringstream();
    oss << "../../tent/tent_" << r << "_" << step <<"_" << dump <<"dumped.npy";
    std::string fname = oss.str();

    cnpy::npy_save(fname, vec, {(unsigned long)(step + 1)}, "w");
    std::cout << "saved to " << fname << std::endl;
    return 0;
    
}