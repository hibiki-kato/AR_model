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
    bool vectorize = false;
    std::string trainDataPathStr = "../lorenz/3137.05sec1000dump_dt=0.01sigma10rho28beta2.66667epsilon0.001.npy";
    double alpha = 0.1;
    int order = 1500;
    bool weighted = false;
    Eigen::VectorXd R = Eigen::VectorXd::Ones(order + 1); // weight vector for regularization
    for (int i = 0; i < order ; i++) {
        R(i) = std::pow(0.999, i);
    }
    R.tail(1)(0) = 0; // regularization weight for constant term is 0
    int scope = order*2;
    std::string predDataPathStr = "../lorenz/3137.05sec1000dump_dt=0.01sigma10rho28beta2.66667epsilon0.001.npy";

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
    std::cout << "data size: " << data.size() << std::endl;
    // Eigen::VectorXd W = Eigen::VectorXd::Ones(data.rows() - order); // data weight vector (fixed to 1)
    // make training data
    Eigen::MatrixXd tra_X = Eigen::MatrixXd::Ones(data.rows() - order, 1 + order);
    Eigen::VectorXd tra_y = data.segment(order, data.rows() - order);
    for(int i = 0; i < order; i++){
        tra_X.col(i) = data.segment(i, data.rows() - order);
    }

    /*
    ███████
    ██      ██  ██   ██   ██
    ██          ██   ██
    ██      ██ ████ ████  ██  █████    █████
    ██      ██  ██   ██   ██  ██  ██  ██  ██
    ██████  ██  ██   ██   ██  ██   █  █    █
    ██      ██  ██   ██   ██  █    █  █    █
    ██      ██  ██   ██   ██  █    █  █    █
    ██      ██  ██   ██   ██  █    █  █    █
    ██      ██   █    █   ██  █    █  ██  ██
    ██      ██   ██   ██  ██  █    █   █████
                                           █
                                      ██  ██
                                       ████
    */
    RidgeParams params;
    params.alpha = alpha;
    if (weighted){
        params.regularize_weight = R;
    }
    Eigen::VectorXd coef = Ridge(tra_X, tra_y, params);
    std::cout << "sum of coef: " << coef.head(order).sum() << std::endl;
    std::cout << "const; " << coef.tail(1)(0) << std::endl;
    // coefをtxt保存
    auto oss = std::ostringstream();
    oss << "../../models/" << trainDataPathStr.substr(6, trainDataPathStr.size() - 10) << "_order" << order << "alpha" << alpha << ".txt";
    std::string filename = oss.str();
    oss.str("");
    std::cout << "saved as " << filename << std::endl;
    std::ofstream ofs(filename);
    ofs << coef << std::endl;
    ofs.close();

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
    Eigen::VectorXd y;
    if (vectorize){
        y = npy2EigenMat<double>(predDataPath).row(target_var - 1); // wide matrix
    } else {
        y = npy2EigenVec<double>(predDataPath);
    }
    Eigen::VectorXd pred_y = predict(y, coef, order);
    // convert to std::vector
    std::vector<double> y_vec = std::vector<double>(y.data(), y.data() + y.size());
    std::vector<double> pred_y_vec = std::vector<double>(pred_y.data(), pred_y.data() + pred_y.size());
    std::cout << "score: " << (y - pred_y).squaredNorm() / y.rows() << std::endl;
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
    keywords["color"] = "red";
    keywords["label"] = "true";
    keywords["lw"] = "0.5";

    keywords["marker"] = ".";
    // keywords["alpha"] = "0.5";
    plt::plot(y_vec, keywords);

    keywords["color"] = "blue";
    keywords["label"] = "prediction";
    plt::plot(pred_y_vec, keywords);
    plt::legend();
    plt::xlim(order, order+scope);
    oss << "../../predict/" << predDataPathStr.substr(6, predDataPathStr.size() - 10) << "_order" << order <<"scope" << scope << "alpha" << alpha << "weighted.png";
    std::string fname = oss.str();
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