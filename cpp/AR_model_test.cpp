/**
 * @file AR_model_test.cpp
 * @brief just a test for AR_model
 * @version 0.1
 * @date 2023-11-01
 * 
 * @copyright Copyright (c)
 * 
 */
#include <complex>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;
Eigen::VectorXd AR_predict(const Eigen::VectorXd& data, int p, int n_pred);

int main() {
    return 0;
}