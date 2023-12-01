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
#include <unordered_map>
#include "cnpy/cnpy.h"
#include "Eigen_numpy_converter.hpp"
#include "functions.hpp"

bool isPointInBall(const Eigen::VectorXd& point, const Eigen::VectorXd& center, double radius);

int main() {
    // 初期条件とパラメータ
    std::unordered_map<std::string, double> params;
    params["sigma"] = 10.0;
    params["rho"] = 28.0;
    params["beta"] = 8.0 / 3.0;
    
    double dt = 0.01;
    int minimum_time = 1e+2;
    int minimum_steps = static_cast<int>(minimum_time / dt + 0.5);
    int maximum_time = 1e+3;
    int maximum_steps = static_cast<int>(maximum_time / dt + 0.5);
    int dump = 1e+3;
    int dumps = static_cast<int>(dump / dt + 0.5);
    int trial = 1e+4;
    double epsilon = 1e-3;
    int threads = omp_get_max_threads();
    std::atomic<bool> success(false);
    std::atomic<int> count(0);

    #pragma omp parallel for schedule(dynamic) num_threads(threads) shared(success)
    for (int i = 0; i < trial; ++i) {
        count++;
        if (success) continue;
        if (omp_get_thread_num() == 0) {
            std::cout << "\r processing..." << count << "/" << trial << std::flush;
        }
        // initial state (random)
        Eigen::Vector3d x = Eigen::Vector3d::Random();
        Eigen::MatrixXd Mat = Eigen::MatrixXd::Zero(3, maximum_steps + 1);
        // dump
        for (int i = 0; i < dumps; ++i) {
            rungeKutta4(std::function<Eigen::Vector3d(Eigen::Vector3d, std::unordered_map<std::string, double>)>(lorenz), x, params, dt);
        }
        // loop
        Mat.col(0) = x;
        for (int j = 1; j < minimum_steps + 1; j++) {
            rungeKutta4(std::function<Eigen::Vector3d(Eigen::Vector3d, std::unordered_map<std::string, double>)>(lorenz), x, params, dt);
            Mat.col(j) = x;
        }

        // loop until the point is in the ball
        for (int j = minimum_steps + 1; j < maximum_steps + 1; j++) {
            rungeKutta4(std::function<Eigen::Vector3d(Eigen::Vector3d, std::unordered_map<std::string, double>)>(lorenz), x, params, dt);
            Mat.col(j) = x;
            if (isPointInBall(Mat.col(j), Mat.col(0), epsilon)) {
                //resize matrix
                Mat.conservativeResize(3, j + 1);
                success = true;
                auto oss = std::ostringstream();
                oss << "../../lorenz_orbit/" << Mat.cols() * dt << "sec"<< dump <<"dump_dt="<< dt <<"sigma" << params["sigma"] << "rho" << params["rho"] << "beta" << params["beta"] << "epsilon" << epsilon << ".npy";
                std::string fname = oss.str();
                EigenMat2npy(Mat, fname);
                std::cout << "saved as " << fname << std::endl;
                break;
            }
        }
    }
    return 0;
}

bool isPointInBall(const Eigen::VectorXd& point, const Eigen::VectorXd& center, double radius) {
    Eigen::VectorXd dist = point - center;
    if (dist.norm() < radius) {
        return true;
    } else {
        return false;
    }
}