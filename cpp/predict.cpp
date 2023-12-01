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
    int order = 1500;
    std::string predDataPathStr = "../lorenz/dat-upo-T12.692183z-periodic.npy";
    std::string coefDataPathStr = "../models/lorenz/dat-upo-T11.256767143036276z-periodic_order1500alpha0.1.txt";
    double scope[] = {order - 10, order*2};
    int step = 0;
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
    predDataPathStr = "../" + predDataPathStr;
    const char* predDataPath = predDataPathStr.c_str();
    Eigen::VectorXd y;
    if (vectorize){
        y = npy2EigenMat<double>(predDataPath).row(target_var - 1); // wide matrix
    } else {
        y = npy2EigenVec<double>(predDataPath);
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

    Eigen::VectorXd pred_y = predict(y, coef, order, rand, step);
    // convert to std::vector
    
    std::vector<double> y_vec = std::vector<double>(y.data(), y.data() + y.size());
    std::vector<double> pred_y_vec = std::vector<double>(pred_y.data(), pred_y.data() + pred_y.size());
    if (step == 0){
        std::cout << "score: " << (y - pred_y).squaredNorm() / y.rows() << std::endl;
    }
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
    if (step != 0) {
        plt::plot(pred_y_vec, keywords);
        if (scope[1] == 0){
            scope[1] = pred_y_vec.size();
        }
        plt::xlim(scope[0], scope[1]);
        oss << "../../predict/" << predDataPathStr.substr(6, predDataPathStr.size() - 10) << "_order" << order <<"scope" << scope <<"_"<< step << "sec.png";
        fname = oss.str();
        plt::save(fname);
        std::cout << "saved as " << fname << std::endl;

        plt::figure_size(1000, 1000);

    }
    else {
        keywords["color"] = "red";
        keywords["label"] = "true";
        if (scope[1] == 0){
            scope[1] = pred_y_vec.size();
        }
        plt::xlim(scope[0], scope[1]);
        keywords["marker"] = ".";
        // keywords["alpha"] = "0.5";
        plt::plot(y_vec, keywords);

        keywords["color"] = "blue";
        keywords["label"] = "prediction";
        plt::plot(pred_y_vec, keywords);
        plt::legend();
        oss << "../../predict/" << predDataPathStr.substr(6, predDataPathStr.size() - 10) << "_order" << order <<"scope" << scope <<".png";
        fname = oss.str();
        plt::save(fname);
        std::cout << "saved as " << fname << std::endl;
    }

    
    

    //timer stops
    auto clock_end = std::chrono::system_clock::now();
    int hours = std::chrono::duration_cast<std::chrono::hours>(clock_end-clock_start).count(); //処理に要した時間を変換
    int minutes = std::chrono::duration_cast<std::chrono::minutes>(clock_end-clock_start).count(); //処理に要した時間を変換
    int seconds = std::chrono::duration_cast<std::chrono::seconds>(clock_end-clock_start).count(); //処理に要した時間を変換
    int milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(clock_end-clock_start).count(); //処理に要した時間を変換
    std::cout << hours << "h " << minutes % 60 << "m " << seconds % 60 << "s " << milliseconds % 1000 << "ms " << std::endl;
    return 0;
}