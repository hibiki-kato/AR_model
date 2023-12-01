#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>
#include "mpfr-impl.h"
#include "AR.hpp"

int main(){
    Eigen::MatrixXd X(3, 3);
    X << 1, 2, 3,
         4, 5, 6,
         7, 8, 9;
    Eigen::VectorXd y(3);
    y << 1, 2, 3;
    double alpha = 0.1;
    Eigen::VectorXd weight(3);
    weight << 1, 1, 1;
    Eigen::VectorXd regularize_weight(3);
    regularize_weight << 1, 1, 1;

    Eigen::VectorXd coef = Ridge(X, y, alpha, weight, regularize_weight);
    std::cout << coef << std::endl;
    return 0;
}