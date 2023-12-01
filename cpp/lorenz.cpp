/**
 * @file lorenz.cpp
 * @author Hibiki Kato
 * @brief lorentz model
 * @version 0.1
 * @date 2023-10-06
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#include <iostream>
#include <string>
#include <sstream>
#include <eigen3/Eigen/Dense>
#include "cnpy/cnpy.h"
#include "Eigen_numpy_converter.hpp"

Eigen::Vector3d lorenz(const Eigen::Vector3d& x, double sigma, double rho, double beta);
void rungeKutta4(Eigen::Vector3d& x, double sigma, double rho, double beta, double dt);

int main() {
    // 初期条件とパラメータ
    double sigma = 10.0;
    double rho = 28.0;
    double beta = 8.0 / 3.0;
    double dt = 0.01;
    int time = 3e+3;
    int steps = static_cast<int>(time / dt + 0.5);
    int dump = 1e+2;
    int dumps = static_cast<int>(dump / dt + 0.5);
    // initial state
    Eigen::Vector3d x(1.0, 1.0, 1.0);
    Eigen::MatrixXd Mat = Eigen::MatrixXd::Zero(3, steps + 1);

    // dump
    for (int i = 0; i < dumps; ++i) {
        rungeKutta4(x, sigma, rho, beta, dt);
    }
    
    // loop
    Mat.col(0) = x;
    for (int i = 1; i < steps + 1; ++i) {
        rungeKutta4(x, sigma, rho, beta, dt);
        Mat.col(i) = x;
    }

    auto oss = std::ostringstream();
    oss << "../../lorenz/lorenz" << time << "sec"<< dump <<"dump_dt="<< dt <<"sigma" << sigma << "rho" << rho << "beta" << beta << ".npy";
    std::string fname = oss.str();

    EigenMat2npy(Mat, fname);
    std::cout << "saved as " << fname << std::endl;
    return 0;
}

// ローレンツ方程式の右辺関数
Eigen::Vector3d lorenz(const Eigen::Vector3d& x, double sigma, double rho, double beta) {
    Eigen::Vector3d dxdt;
    dxdt(0) = sigma * (x(1) - x(0));
    dxdt(1) = x(0) * (rho - x(2)) - x(1);
    dxdt(2) = x(0) * x(1) - beta * x(2);
    return dxdt;
}

// 4段4次ルンゲクッタ法
void rungeKutta4(Eigen::Vector3d& x, double sigma, double rho, double beta, double dt) {
    Eigen::Vector3d k1, k2, k3, k4;

    k1 = lorenz(x, sigma, rho, beta);
    k2 = lorenz(x + dt / 2.0 * k1, sigma, rho, beta);
    k3 = lorenz(x + dt / 2.0 * k2, sigma, rho, beta);
    k4 = lorenz(x + dt * k3, sigma, rho, beta);

    x += dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}
