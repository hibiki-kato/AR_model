#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <string>
#include <map>
#include <algorithm>
#include <functional>
#include <eigen3/Eigen/Dense>
#include "matplotlibcpp.h"
#include "Eigen_numpy_converter.hpp"
#include "AR.hpp"

namespace plt = matplotlibcpp;

int main(){
    auto clock_start = std::chrono::system_clock::now(); //timer starts
    /*
     ████          ██     ██████
    ██  ██         ██     █    █
    █    █   ███  ████    █    ██  ███   ███  ███   █████ ███   ███
    ██      ██  █  ██     █    ██ █  ██  ██  █  ██  ██  ██  █  █  ██
     ████   █   █  ██     ██████      █  █       █  █   █   ██ ██
        ██  █████  ██     █        ████  █    ████  █   █   ██  ███
    █    █  █      ██     █       █   █  █   █   █  █   █   ██     █
    ██  ██  ██  █  ██     █       █  ██  █   █  ██  █   █   ██ █   █
     ████    ████   ██    █       █████  █   █████  █   █   ██  ███
    */
    int target_var = 3; // target variable (from 1)
    bool vectorize = true;
    std::string trainDataPathStr = "../lorenz/lorenz1000sec10000dump_dt=0.01sigma10rho28beta2.66667.npy";
    int ite = 50;
    Eigen::VectorXd alphas = Eigen::VectorXd::LinSpaced(ite, -3, 1);
    for (auto& alpha : alphas) alpha = std::pow(10, alpha);
    int order = 1500;
    bool weighted = false;
    Eigen::VectorXd R = Eigen::VectorXd::Ones(order + 1); // weight vector for regularization
    for (int i = 0; i < order ; i++) {
        R(i) = std::pow(0.98, i);
    }
    R.tail(1)(0) = 0; // regularization weight for constant term is 0
    /*
    ██████                                                     █████
    ██   ██                                                    ██   ██           ██
    ██    ██                                                   ██    █           ██
    ██    ██ ████  ████     ████    ████    ████    ████       ██    ██   ████  ████   ████
    ██    ██ ███  ██  ██   ██  ██  ██  ██  ██  ██  ██  ██      ██    ██  ██  ██  ██   ██  ██
    ███████  █    █    ██  █    █  █    █  ██      ██          ██    ██       █  ██        █
    ██       █    █    ██  █       ██████  ████    ████        ██    ██   █████  ██    █████
    ██       █    █    ██  █       █          ███     ███      ██    ██  ██   █  ██   ██   █
    ██       █    █    ██  █    █  █       █   ██  █   ██      ██    █   █    █  ██   █    █
    ██       █    ██  ██   ██  ██  ██   █  ██  ██  ██  ██      ██   ██   ██  ██   █   ██  ██
    ██       █     ████     ████    ████    ████    ████       █████      ███ █   ██   ███ █
    */
    // load data
    trainDataPathStr = "../" + trainDataPathStr;
    const char* trainDataPath = trainDataPathStr.c_str();
    Eigen::VectorXd data;
    if (vectorize){
        data = npy2EigenMat<double>(trainDataPath).row(target_var - 1); 
    } else {
        data = npy2EigenVec<double>(trainDataPath);
    }
    // make training data
    Eigen::MatrixXd tra_X = Eigen::MatrixXd::Ones(data.rows() - order, 1 + order);
    Eigen::VectorXd tra_y = data.segment(order, data.rows() - order);
    for(int i = 0; i < order; i++){
        tra_X.col(i) = data.segment(i, data.rows() - order);
    }
    /*
      ████                  █       █████                                █
     ██  ██        ██       █      ██   ██                               █
    ██    ██                █      █     █                               █
    ██        ████ ██   █████      ██        ████    ████   ████  ████   █████
    █         ███  ██  ██  ██      ███      ██  ██  ██  ██  ███  ██  ██  ██  ██
    █         █    ██  █    █        ████   █    █       █  █    █    █  ██   █
    █   ████  █    ██  █    █          ███  ██████   █████  █    █       █    █
    ██     █  █    ██  █    █            █  █       ██   █  █    █       █    █
    ██     █  █    ██  █    █      █     █  █       █    █  █    █    █  █    █
     ██   ██  █    ██  ██  ██      ██   ██  ██   █  ██  ██  █    ██  ██  █    █
      █████   █    ██   █████       █████    ████    ███ █  █     ████   █    █
    */
    std::vector<double> scores(ite);
    std::vector<double> alphas_vec = std::vector<double>(alphas.data(), alphas.data() + alphas.size());
    RidgeParams params;
    if (weighted){
        params.regularize_weight = R;
    }
    for (int i=0; i < ite; i++){
        std::cout << "\r processing " << i+1 << " / " << ite << std::flush;
        params.alpha = alphas(i);
        scores[i] = cross_validation(std::function<Eigen::VectorXd(const Eigen::MatrixXd&, const Eigen::VectorXd&, const RidgeParams&)>(Ridge), tra_X, tra_y, params, 2);
    }
    int min_idx = std::distance(scores.begin(), std::min_element(scores.begin(), scores.end()));
    std::cout << "min score: " << scores[min_idx] << std::endl;
    std::cout << "alpha: " << alphas(min_idx) << std::endl;
    /*
             ██
    ██████   ██
    ██   ██  ██           ██   ██   ██
    ██    ██ ██           ██   ██
    ██    ██ ██   ████   ████ ████  ██  █████    █████
    ██    ██ ██  ██  ██   ██   ██   ██  ██  ██  ██  ██
    ███████  ██  █    ██  ██   ██   ██  ██   █  █    █
    ██       ██  █    ██  ██   ██   ██  █    █  █    █
    ██       ██  █    ██  ██   ██   ██  █    █  █    █
    ██       ██  █    ██  ██   ██   ██  █    █  █    █
    ██       ██  ██  ██    █    █   ██  █    █  ██  ██
    ██       ██   ████     ██   ██  ██  █    █   █████
                                                     █
                                                ██  ██
                                                 ████
    */
    plt::figure_size(1200, 400);
    std::map<std::string, std::string> keywords;
    keywords["color"] = "blue";
    keywords["marker"] = ".";
    plt::plot(alphas_vec, scores, keywords);
    plt::xscale("log");
    std::string fname = "sample.png";
    plt::save(fname);
    std::cout << "saved as " << fname << std::endl;

    //timer stops
    auto clock_end = std::chrono::system_clock::now();
    int hours = std::chrono::duration_cast<std::chrono::hours>(clock_end-clock_start).count(); //処理に要した時間を変換
    int minutes = std::chrono::duration_cast<std::chrono::minutes>(clock_end-clock_start).count(); //処理に要した時間を変換
    int seconds = std::chrono::duration_cast<std::chrono::seconds>(clock_end-clock_start).count(); //処理に要した時間を変換
    int milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(clock_end-clock_start).count(); //処理に要した時間を変換
    std::cout << hours << "h " << minutes % 60 << "m " << seconds % 60 << "s " << milliseconds % 1000 << "ms " << std::endl;
    return 0;
}