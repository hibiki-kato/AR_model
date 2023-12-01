/**
 * @file functions.hpp
 * @author Hibiki Kato 
 * @brief header file for functions.cpp
 * @version 0.1
 * @date 2023-11-21
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#pragma once
#include <iostream>
#include <vector>
#include<eigen3/Eigen/Dense>
#include <functional>
#include <unordered_map>

Eigen::Vector3d lorenz(const Eigen::Vector3d& x, std::unordered_map<std::string, double> params);
Eigen::MatrixXd loc_max(const Eigen::MatrixXd& traj_abs, int loc_max_dim);
Eigen::VectorXd loc_max(const Eigen::VectorXd& traj_abs);

Eigen::MatrixXd poincare_section(const Eigen::MatrixXd& traj_abs, int cut_dim, double cut_value);
Eigen::VectorXd poincare_section(const Eigen::VectorXd& traj_abs, double cut_value);

std::vector<int> extractCommonColumns(const std::vector<Eigen::MatrixXd>& matrices);

// 4段4次ルンゲクッタ法
template <typename Vector>
void rungeKutta4(std::function<Vector(Vector, std::unordered_map<std::string, double>)> func, Vector& x, std::unordered_map<std::string, double> params, double dt) {
    Vector k1, k2, k3, k4;
    k1 = func(x, params);
    k2 = func(x + dt / 2.0 * k1, params);
    k3 = func(x + dt / 2.0 * k2, params);
    k4 = func(x + dt * k3, params);

    x += dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}