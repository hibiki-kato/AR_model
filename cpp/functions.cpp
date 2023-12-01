/**
 * @file functions.cpp
 * @author Hibiki Kato
 * @brief my functions
 * @version 0.1
 * @date 2023-11-21
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <functional>
#include <unordered_map>
#include "functions.hpp"

// ローレンツ方程式の右辺関数
Eigen::Vector3d lorenz(const Eigen::Vector3d& x, std::unordered_map<std::string, double> params) {
    Eigen::Vector3d dxdt;
    dxdt(0) = params["sigma"] * (x(1) - x(0));
    dxdt(1) = x(0) * (params["rho"] - x(2)) - x(1);
    dxdt(2) = x(0) * x(1) - params["beta"] * x(2);
    return dxdt;
}

Eigen::MatrixXd loc_max(const Eigen::MatrixXd& traj_abs, int loc_max_dim){
    // 条件に合えば1, 合わなければ0のベクトルを作成
    std::vector<int> binLoc_max(traj_abs.cols());
    //　最初の3点と最後の3点は条件を満たせないので0
    for (int i = 0; i < 3; ++i){
        binLoc_max[i] = 0;
        binLoc_max[binLoc_max.size()-1-i] = 0;
    }
    for (int i = 0; i < traj_abs.cols()-6; ++i){
        //極大値か判定
        if (traj_abs(loc_max_dim, i+1) - traj_abs(loc_max_dim, i) > 0
        && traj_abs(loc_max_dim, i+2) - traj_abs(loc_max_dim, i+1) > 0
        && traj_abs(loc_max_dim, i+3) - traj_abs(loc_max_dim, i+2) > 0
        && traj_abs(loc_max_dim, i+4) - traj_abs(loc_max_dim, i+3) < 0
        && traj_abs(loc_max_dim, i+5) - traj_abs(loc_max_dim, i+4) < 0
        && traj_abs(loc_max_dim, i+6) - traj_abs(loc_max_dim, i+5) < 0){
            binLoc_max[i+3] = 1;
        } else{
            binLoc_max[i+3] = 0;
        }
    }
    //binLoc_maxの合計
    int count = std::accumulate(binLoc_max.begin(), binLoc_max.end(), 0);
    Eigen::MatrixXd loc_max_point(traj_abs.rows(),count);
    int col_now = 0;
    for (int i = 0; i < binLoc_max.size(); ++i){
        if (binLoc_max[i] == 1){
            loc_max_point.col(col_now) = traj_abs.col(i);
            col_now++;
        }
    }
    return loc_max_point;
}

Eigen::VectorXd loc_max(const Eigen::VectorXd& traj_abs){
    // 条件に合えば1, 合わなければ0のベクトルを作成
    std::vector<int> binLoc_max(traj_abs.size());
    //　最初の3点と最後の3点は条件を満たせないので0
    for (int i = 0; i < 3; ++i){
        binLoc_max[i] = 0;
        binLoc_max[binLoc_max.size()-1-i] = 0;
    }
    for (int i = 0; i < traj_abs.size()-6; ++i){
        //極大値か判定
        if (traj_abs(i+1) - traj_abs(i) > 0
        && traj_abs(i+2) - traj_abs(i+1) > 0
        && traj_abs(i+3) - traj_abs(i+2) > 0
        && traj_abs(i+4) - traj_abs(i+3) < 0
        && traj_abs(i+5) - traj_abs(i+4) < 0
        && traj_abs(i+6) - traj_abs(i+5) < 0){
            binLoc_max[i+3] = 1;
        }
        //
        else{
            binLoc_max[i+3] = 0;
        }
    }
    //binLoc_maxの合計
    int count = std::accumulate(binLoc_max.begin(), binLoc_max.end(), 0);
    Eigen::VectorXd loc_max_point(count);
    int col_now = 0;
    for (int i = 0; i < binLoc_max.size(); ++i){
        if (binLoc_max[i] == 1){
            loc_max_point(col_now) = traj_abs(i);
            col_now++;
        }
    }
    return loc_max_point;
}

Eigen::MatrixXd poincare_section(const Eigen::MatrixXd& traj_abs, int cut_dim, double cut_value){
    // 条件に合えば1, 合わなければ0のベクトルを作成
    std::vector<int> binSection(traj_abs.cols(), 0);

    for (int i = 0; i < traj_abs.cols() -1; ++i){
        if ((traj_abs(cut_dim, i) > cut_value && traj_abs(cut_dim - 1, i+1) < cut_value)
        || (traj_abs(cut_dim - 1, i) < cut_value && traj_abs(cut_dim - 1, i+1) > cut_value)){
            binSection[i] = 1;
            binSection[i+1] = 1;
        }
    }
    //binSectionの1の数を数える
    int count = std::accumulate(binSection.begin(), binSection.end(), 0);
    //binSectionの1の数だけの行列を作成
    Eigen::MatrixXd PoincareSection(traj_abs.rows(), count);
    int col_now = 0;
    for (int i = 0; i < binSection.size(); ++i){
        if (binSection[i] == 1){
            PoincareSection.col(col_now) = traj_abs.col(i);
            col_now++;
        }
    }
    return PoincareSection;
}
Eigen::VectorXd poincare_section(const Eigen::VectorXd& traj_abs, double cut_value){
    // 条件に合えば1, 合わなければ0のベクトルを作成
    std::vector<int> binSection(traj_abs.size(), 0);

    for (int i = 0; i < traj_abs.size() -1; ++i){
        if ((traj_abs(i) > cut_value && traj_abs(i+1) < cut_value)
        || (traj_abs(i) < cut_value && traj_abs(i+1) > cut_value)){
            binSection[i] = 1;
            binSection[i+1] = 1;
        }
    }
    //binSectionの1の数を数える
    int count = std::accumulate(binSection.begin(), binSection.end(), 0);
    //binSectionの1の数だけの行列を作成
    Eigen::VectorXd PoincareSection(count);
    int col_now = 0;
    for (int i = 0; i < binSection.size(); ++i){
        if (binSection[i] == 1){
            PoincareSection(col_now) = traj_abs(i);
            col_now++;
        }
    }
    return PoincareSection;
}

std::vector<int> extractCommonColumns(const std::vector<Eigen::MatrixXd>& matrices) {
    // ベースとなる行列を選ぶ（ここでは最初の行列）
    const Eigen::MatrixXd& baseMatrix = matrices.front();
    
    std::vector<int> commonColumns; // 共通の列のBase行列におけるインデックスを格納する配列
    // matricesの要素が1つの場合、Base行列の全てのインデックスを格納して返す
    if (matrices.size() == 1) {
        for (int i = 0; i < baseMatrix.cols(); ++i) {
            commonColumns.push_back(i);
        }
        return commonColumns;
    }
    // ベース行列の各列を走査
    for (int baseCol = 0; baseCol < baseMatrix.cols(); ++baseCol) {
        bool isCommon = true; // 共通の列があるかを示すフラグ

        // ベース行列の現在の列と同じ列が他の行列にあるか検証
        for (size_t i = 1; i < matrices.size(); ++i) {
            //フラグがfalse、つまり既に共通の列がない場合はループを抜ける
            if (!isCommon) {
                break;
            }
            const Eigen::MatrixXd& matrix = matrices[i];
            isCommon = false;
            // 共通の列があればtrue,　なければfalse
            for (int col=0; col < matrix.cols(); ++col) {
                // ベース行列の現在の列と他の行列の現在の列が等しいかを比較
                if (baseMatrix.col(baseCol) == matrix.col(col)) {
                    isCommon = true;
                    break;
                }
            }
        }

        // 共通の列であればインデックスを追加
        if (isCommon) {
            commonColumns.push_back(baseCol);
        }
    }

    return commonColumns;
}