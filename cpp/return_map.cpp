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
#include "functions.hpp"

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
    int order = 1500;
    std::string predDataPathStr = "../lorenz/dat-upo-T12.692183z-periodic.npy";
    std::string coefDataPathStr = "../models/lorenz/dat-upo-T11.256767143036276z-periodic_order1500alpha0.1.txt";
    double dt = 0.01;
    double rand = 0;
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
   // load data
    predDataPathStr = "../" + predDataPathStr;
    const char* predDataPath = predDataPathStr.c_str();
    Eigen::VectorXd y;
    if (vectorize){
        y = npy2EigenMat<double>(predDataPath).row(target_var - 1); // wide matrix
    } else {
        y = npy2EigenVec<double>(predDataPath);
    }
    std::cout << "y: " << y.size() << std::endl;
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
    // predict
    Eigen::VectorXd pred_y = predict(y, coef, order, rand);
    std::vector<double> x = std::vector<double>(pred_y.data(), pred_y.data() + pred_y.size());
    plt::plot(x);
    plt::save("sample.png");
    // poincare_mapping
    Eigen::VectorXd pred_y_max = loc_max(pred_y.tail(pred_y.size()-order));
    Eigen::VectorXd y_max = loc_max(y.tail(y.size()-order));

    std::cout << "pred_y_max: " << pred_y_max.size() << std::endl;
    std::cout << "y_max: " << y_max.size() << std::endl;
    // convert to std::vector
    std::vector<double> y_max_vec = std::vector<double>(y_max.data(), y_max.data() + y_max.size());
    std::vector<double> pred_y_max_vec = std::vector<double>(pred_y_max.data(), pred_y_max.data() + pred_y_max.size());
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
    std::map<std::string, std::string> keywords;
    
    // //true value
    // plt::figure_size(2000, 1000);
    // plt::subplot(1, 2, 1);
    // keywords["s"] = "1";
    // plt::scatter(std::vector<double>(y_max_vec.begin() , y_max_vec.end() - 1), std::vector<double>(y_max_vec.begin() + 1, y_max_vec.end()));
    // plt::xlim(30.0, 47.5);
    // plt::ylim(30.0, 47.5);
    // plt::xlabel("z(t)");
    // plt::ylabel("z(t+1)");
    // plt::title("True");
    // //predicted value
    // plt::subplot(1, 2, 2);
    // plt::scatter(std::vector<double>(pred_y_max_vec.begin(), pred_y_max_vec.end() - 1), std::vector<double>(pred_y_max_vec.begin() + 1, pred_y_max_vec.end()));
    // plt::xlim(30.0, 47.5);
    // plt::ylim(30.0, 47.5);
    // plt::xlabel("z(t)");
    // plt::ylabel("z(t+1)");
    // plt::title("Predicted");

    // 1枚に重ねて書く
    plt::figure_size(800, 800);
    keywords["label"] = "true";
    plt::scatter(std::vector<double>(y_max_vec.begin() , y_max_vec.end() - 1), std::vector<double>(y_max_vec.begin() + 1, y_max_vec.end()), 5.0, keywords);
    keywords["label"] = "predicted";
    plt::scatter(std::vector<double>(pred_y_max_vec.begin(), pred_y_max_vec.end() - 1), std::vector<double>(pred_y_max_vec.begin() + 1, pred_y_max_vec.end()), 5.0, keywords);
    plt::xlim(30.0, 47.5);
    plt::ylim(30.0, 47.5);
    plt::xlabel("z(t)");
    plt::ylabel("z(t+1)");
    plt::title("True and Predicted");
    plt::legend();
    // plt::show();
    plt::save("sample.png");

    std::ostringstream oss;
    oss << "../../cobweb/" << predDataPathStr.substr(6, coefDataPathStr.size() - 10) << y.size() << "step.png";
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