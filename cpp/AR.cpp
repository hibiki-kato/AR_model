/**
 * @file AR.cpp
 * @author hibiki kato
 * @brief functions for AR model. Header file is AR.h
 * @version 0.1
 * @date 2023-11-06
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <functional>
#include "AR.hpp"

/**
 * @brief Weighted Ridge regression
 *      This is used as ordinary Ridge regression when weight and regularize_weight are not designated.
 *      This is used as OLS when alpha is 0.
 *      Regression equation is
 *                      y = X * coef
 *     where coef is coefficient vector.
 *      
 *      Loss function is
 *                      (y - X * coef)^T * W * (y - X * coef) + alpha * coef^T * R * coef
 *    where W is weight vector and R is regularize weight vector.
 *      Analytical solution is
 *                      coef = (X^T * W * X + alpha * R)^{-1} * X^T * W * y
 * 
 * @param X : data matrix (const 1 should be added to the last column)
 * @param y : target vector
 * @param params : struct for regression parameters (alpha, weight, regularize_weight)
 * @return Eigen::VectorXd : coefficient vector
 */
Eigen::VectorXd Ridge(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, const RidgeParams& params) {   
    int dim = X.cols();

    Eigen::VectorXd coef;
    // if W and R are not designated, use Ridge regression
    if (params.weight.size() == 0 && params.regularize_weight.size() == 0){
        Eigen::MatrixXd R = Eigen::MatrixXd::Identity(dim, dim);
        R.bottomRightCorner(1, 1) *= 0; // weight of constant term is 0
        coef = (X.transpose() * X + params.alpha * R).colPivHouseholderQr().solve(X.transpose() * y);

    }
    else if (params.weight.size() == 0 && params.regularize_weight.size() != 0) {
        // Make weight vectors diagonal matrix
        Eigen::MatrixXd R = params.regularize_weight.asDiagonal();
        coef = (X.transpose() * X + params.alpha * R).colPivHouseholderQr().solve(X.transpose() * y);
    }
    else if (params.weight.size() != 0 && params.regularize_weight.size() == 0) {
        // Make weight vectors diagonal matrix
        Eigen::MatrixXd W = params.weight.asDiagonal();
        Eigen::MatrixXd R = Eigen::MatrixXd::Identity(dim, dim);
        R.bottomRightCorner(1, 1) *= 0; // weight of constant term is 0
        coef = (X.transpose() * W * X + params.alpha * R).colPivHouseholderQr().solve(X.transpose() * W * y);
    }
    else {
        // Make weight vectors diagonal matrix
        Eigen::MatrixXd W = params.weight.asDiagonal();
        Eigen::MatrixXd R = params.regularize_weight.asDiagonal();
        coef = (X.transpose() * W * X + params.alpha * R).colPivHouseholderQr().solve(X.transpose() * W * y);
    }
    return coef;
}

Eigen::VectorXd predict(Eigen::VectorXd data, Eigen::VectorXd coef, int order, double rand,int step){
    Eigen::VectorXd pred;
    if (rand != 0){
        //　generate random number
        std::random_device seed_gen;
        std::mt19937 engine(seed_gen());
        std::normal_distribution<> dist(0, rand);

        if (step != 0){
            pred = Eigen::VectorXd::Zero(step);
            pred.head(order) = data.head(order);
            for(int i = 0; i < pred.size() - order; i++){
                pred(order + i) = coef.tail(1)(0) + coef.head(order).dot(pred.segment(i, order)) + dist(engine);
            }
        }
        else{
            pred = Eigen::VectorXd::Zero(data.size());
            pred.head(order) = data.head(order);

            for(int i = 0; i < pred.size() - order; i++){
                pred(order + i) = coef.tail(1)(0) + coef.head(order).dot(pred.segment(i, order)) + dist(engine);
            }
        }
    } else {
        if (step != 0){
            pred = Eigen::VectorXd::Zero(step);
            pred.head(order) = data.head(order);
            for(int i = 0; i < pred.size() - order; i++){
                pred(order + i) = coef.tail(1)(0) + coef.head(order).dot(pred.segment(i, order));
            }
        }
        else{
            pred = Eigen::VectorXd::Zero(data.size());
            pred.head(order) = data.head(order);

            for(int i = 0; i < pred.size() - order; i++){
                pred(order + i) = coef.tail(1)(0) + coef.head(order).dot(pred.segment(i, order));
            }
        }
    }
    return pred;
}