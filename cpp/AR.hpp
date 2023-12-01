/**
 * @file AR.hpp
 * @author hibiki kato
 * @brief header file for AR model. cpp file is AR.cpp
 * @version 0.1
 * @date 2023-11-06
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#pragma once
#include <iostream>
#include <vector>
#include<eigen3/Eigen/Dense>
#include <functional>

struct RidgeParams {
    double alpha;
    Eigen::VectorXd weight;
    Eigen::VectorXd regularize_weight;
};

Eigen::VectorXd Ridge(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, const RidgeParams& params);
Eigen::VectorXd predict(Eigen::VectorXd data, Eigen::VectorXd coef, int order, double rand = 0,int step = 0);

/**
 * @brief Cross validation function
 * 
 * @tparam T : struct for regression parameters
 * @param func : regression function (std::function)
 * @param X : data matrix (const 1 should be added to the first column)
 * @param y : target vector
 * @param params : struct for regression parameters
 * @param cv : number of folds
 * @return double : mean of residual sum of squares
 */
template<typename T>
double cross_validation(std::function<Eigen::VectorXd(const Eigen::MatrixXd&, const Eigen::VectorXd&, const T&)> func, const Eigen::MatrixXd& X, const Eigen::VectorXd& y, const T& params, int cv) {
    int n = X.rows();
    int m = X.cols();
    int fold_size = n / cv;
    double rss_sum = 0.0;
    for (int i = 0; i < cv; i++) {
        int start = i * fold_size;
        int end = (i == cv - 1) ? n : (i + 1) * fold_size;
        Eigen::MatrixXd X_train(n - (end - start), m);
        Eigen::VectorXd y_train(n - (end - start));
        Eigen::MatrixXd X_test(end - start, m);
        Eigen::VectorXd y_test(end - start);
        int train_idx = 0;
        int test_idx = 0;
        for (int j = 0; j < n; j++) {
            if (j >= start && j < end) {
                X_test.row(test_idx) = X.row(j);
                y_test(test_idx) = y(j);
                test_idx++;
            } else {
                X_train.row(train_idx) = X.row(j);
                y_train(train_idx) = y(j);
                train_idx++;
            }
        }
        Eigen::VectorXd coef = func(X_train, y_train, params);
        Eigen::VectorXd y_pred = X_test * coef;
        Eigen::VectorXd residuals = y_test - y_pred;
        rss_sum += residuals.squaredNorm();
    }
    return rss_sum / cv;
}