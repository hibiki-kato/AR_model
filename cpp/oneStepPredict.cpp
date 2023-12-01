#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
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
    int order = 1500;
    std::string predDataPathStr = "../lorenz/lorenz1000sec1000dump_dt=0.01sigma10rho28beta2.66667.npy";
    std::string coefDataPathStr = "../models/lorenz/lorenz1000sec10000dump_dt=0.01sigma10rho28beta2.66667_order1500alpha0.11.txt";
    int scope[] = {2000, 16000};
    /*
    ██████                     █
    ██   ██                    █  ██          ██
    ██    ██                   █              ██
    ██    ██ ████  ████    █████  ██   ████  ████
    ██    ██ ███  ██  ██  ██  ██  ██  ██  ██  ██
    ███████  █    █    █  █    █  ██  █    █  ██
    ██       █    ██████  █    █  ██  █       ██
    ██       █    █       █    █  ██  █       ██
    ██       █    █       █    █  ██  █    █  ██
    ██       █    ██   █  ██  ██  ██  ██  ██   █
    ██       █     ████    █████  ██   ████    ██
    */
    predDataPathStr = "../" + predDataPathStr;
    const char* predDataPath = predDataPathStr.c_str();
    Eigen::VectorXd data;
    if (vectorize){
        data = npy2EigenMat<double>(predDataPath).row(target_var - 1); // wide matrix
    } else {
        data = npy2EigenVec<double>(predDataPath);
    }
    // tile y
    Eigen::MatrixXd X_test = Eigen::MatrixXd::Ones(data.rows() - order, 1 + order);
    Eigen::VectorXd y_test = data.segment(order, data.rows() - order);
    for(int i = 0; i < order; i++){
        X_test.col(i) = data.segment(i, data.rows() - order);
    }

    // load coefficient
    coefDataPathStr = "../" + coefDataPathStr;
    const char* coefDataPath = coefDataPathStr.c_str();
    Eigen::VectorXd coef;
    std::ifstream file(coefDataPath);
    std::vector<double> coef_vec;
    double value;
    while (file >> value) {
        coef_vec.push_back(value);
    }
    file.close();
    coef = Eigen::VectorXd::Map(&coef_vec[0], coef_vec.size());
    std::cout << "coef: " << coef.size() << std::endl;

    Eigen::VectorXd y_pred = X_test * coef;
    // convert to std::vector
    std::vector<double> y_test_vec = std::vector<double>(y_test.data(), y_test.data() + y_test.size());
    std::vector<double> y_pred_vec = std::vector<double>(y_pred.data(), y_pred.data() + y_pred.size());
    std::cout << "score: " << (y_test - y_pred).squaredNorm() / y_pred.rows() << std::endl;
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
    std::map<std::string, std::string> rcParams;
    rcParams["font.size"] = "12";
    rcParams["font.family"] = "Times New Roman";
    plt::rcparams(rcParams);
    std::ostringstream oss;
    std::string fname;
    plt::figure_size(1000, 400);
    std::map<std::string, std::string> keywords;
    // keywords["marker"] = ".";
    keywords["lw"] = "0.5";
    keywords["color"] = "red";
    keywords["label"] = "true";
    if (scope[1] == 0){
        plt::xlim(scope[0], static_cast<int>(y_test.size()));
    } else {
        plt::xlim(scope[0], scope[1]);
    }
    // keywords["alpha"] = "0.5";
    plt::plot(y_test_vec, keywords);
    keywords["color"] = "blue";
    keywords["label"] = "prediction";
    plt::plot(y_pred_vec, keywords);
    plt::legend();
    oss << "../../oneStepPredict/" << predDataPathStr.substr(6, predDataPathStr.size() - 10) << "_order" << order <<".png";
    fname = oss.str();
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