/**
 * @file      ICPBenchmark.cpp
 * @brief     Pure ICP Algorithm Benchmark Implementation
 * @author    Seungwon Choi
 * @date      2025-12-22
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 */

#include "ICPBenchmark.h"
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <sstream>

namespace lidar_slam {
namespace benchmark {

// ============================================================================
// Utility Functions
// ============================================================================

std::pair<double, double> ICPAlgorithm::computePoseError(
    const Eigen::Matrix4f& estimated,
    const Eigen::Matrix4f& ground_truth)
{
    // Both estimated and ground_truth should now be in the same convention
    // Compute T_error = T_gt^{-1} * T_est
    // If estimated == ground_truth, T_error = I (identity), error = 0
    
    Eigen::Matrix3f R_gt = ground_truth.block<3, 3>(0, 0);
    Eigen::Vector3f t_gt = ground_truth.block<3, 1>(0, 3);
    
    Eigen::Matrix3f R_gt_inv = R_gt.transpose();
    Eigen::Vector3f t_gt_inv = -R_gt_inv * t_gt;
    
    Eigen::Matrix3f R_est = estimated.block<3, 3>(0, 0);
    Eigen::Vector3f t_est = estimated.block<3, 1>(0, 3);
    
    Eigen::Matrix3f R_error = R_gt_inv * R_est;
    Eigen::Vector3f t_error = R_gt_inv * t_est + t_gt_inv;
    
    // Translation error
    double translation_error = t_error.norm();
    
    // Rotation error from R_error: trace(R) = 1 + 2*cos(θ)
    // Use axis-angle representation for more stable computation
    Eigen::AngleAxisf aa(R_error);
    double rotation_error = std::abs(aa.angle()) * 180.0 / M_PI;
    
    return {translation_error, rotation_error};
}


// ============================================================================
// Point-to-Point ICP - Correspondence Finding
// ============================================================================

std::vector<CorrespondenceP2P> PointToPointICP::findCorrespondencesP2P(
    const PointCloudWithCovariance& source,
    const PointCloudWithCovariance& target,
    const Eigen::Matrix4f& current_transform,
    double max_distance)
{
    std::vector<CorrespondenceP2P> correspondences;
    correspondences.reserve(source.size());
    
    if (target.size() == 0) {
        return correspondences;
    }
    
    // Build KD-Tree for target cloud
    KDTree kdtree(3, target, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    kdtree.buildIndex();
    
    Eigen::Matrix3f R = current_transform.block<3, 3>(0, 0);
    Eigen::Vector3f t = current_transform.block<3, 1>(0, 3);
    
    const float max_dist_sq = static_cast<float>(max_distance * max_distance);
    
    for (size_t i = 0; i < source.size(); ++i) {
        Eigen::Vector3f p_transformed = R * source.points[i].point + t;
        
        // 1-NN search (only need nearest neighbor for Point-to-Point)
        size_t nn_idx;
        float nn_dist_sq;
        float query[3] = {p_transformed.x(), p_transformed.y(), p_transformed.z()};
        
        size_t found = kdtree.knnSearch(query, 1, &nn_idx, &nn_dist_sq);
        
        if (found == 0 || nn_dist_sq > max_dist_sq) {
            continue;
        }
        
        // Accept correspondence
        CorrespondenceP2P corr;
        corr.source_idx = static_cast<int>(i);
        corr.target_idx = static_cast<int>(nn_idx);
        corr.distance_sq = nn_dist_sq;
        correspondences.push_back(corr);
    }
    
    return correspondences;
}

// ============================================================================
// Point-to-Plane ICP - Correspondence Finding
// ============================================================================

std::vector<CorrespondenceP2Plane> PointToPlaneICP::findCorrespondencesP2Plane(
    const PointCloudWithCovariance& source,
    const PointCloudWithCovariance& target,
    const Eigen::Matrix4f& current_transform,
    double max_distance)
{
    std::vector<CorrespondenceP2Plane> correspondences;
    correspondences.reserve(source.size());
    
    if (target.size() < 5) {
        return correspondences;
    }
    
    // Build KD-Tree for target cloud
    KDTree kdtree(3, target, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    kdtree.buildIndex();
    
    Eigen::Matrix3f R = current_transform.block<3, 3>(0, 0);
    Eigen::Vector3f t = current_transform.block<3, 1>(0, 3);
    
    const int K = 5;  // Number of neighbors for plane fitting
    const float max_dist_sq = static_cast<float>(max_distance * max_distance);
    const float plane_residual_threshold = 0.3f;
    
    for (size_t i = 0; i < source.size(); ++i) {
        Eigen::Vector3f p_transformed = R * source.points[i].point + t;
        
        // K-NN search for plane fitting
        std::vector<size_t> indices(K);
        std::vector<float> distances_sq(K);
        float query[3] = {p_transformed.x(), p_transformed.y(), p_transformed.z()};
        
        size_t found = kdtree.knnSearch(query, K, indices.data(), distances_sq.data());
        
        if (found < 5 || distances_sq[0] > max_dist_sq) {
            continue;
        }
        
        // Collect neighbor points for plane fitting
        std::vector<Eigen::Vector3f> neighbors;
        neighbors.reserve(found);
        for (size_t k = 0; k < found; ++k) {
            neighbors.push_back(target.points[indices[k]].point);
        }
        
        // Check collinearity (avoid degenerate planes)
        if (neighbors.size() >= 3) {
            Eigen::Vector3f v1 = neighbors[1] - neighbors[0];
            Eigen::Vector3f v2 = neighbors[2] - neighbors[0];
            float cross_norm = v1.cross(v2).norm();
            if (cross_norm < 0.01f) {
                continue;
            }
        }
        
        // Fit plane using SVD
        Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
        for (const auto& pt : neighbors) {
            centroid += pt;
        }
        centroid /= static_cast<float>(neighbors.size());
        
        Eigen::MatrixXf A(neighbors.size(), 3);
        for (size_t k = 0; k < neighbors.size(); ++k) {
            A.row(k) = (neighbors[k] - centroid).transpose();
        }
        
        Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullV);
        Eigen::Vector3f plane_normal = svd.matrixV().col(2);
        float plane_d = -plane_normal.dot(centroid);
        
        // Compute point-to-plane residual
        float residual = std::abs(plane_normal.dot(p_transformed) + plane_d);
        
        if (residual > plane_residual_threshold) {
            continue;
        }
        
        // Accept correspondence
        CorrespondenceP2Plane corr;
        corr.source_idx = static_cast<int>(i);
        corr.target_idx = static_cast<int>(indices[0]);
        corr.target_normal = plane_normal;
        corr.source_normal = source.points[i].normal;  // Precomputed
        corr.distance_sq = distances_sq[0];
        correspondences.push_back(corr);
    }
    
    return correspondences;
}

// ============================================================================
// Symmetric ICP - Correspondence Finding
// ============================================================================

std::vector<CorrespondenceP2Plane> SymmetricICP::findCorrespondencesSymmetric(
    const PointCloudWithCovariance& source,
    const PointCloudWithCovariance& target,
    const Eigen::Matrix4f& current_transform,
    double max_distance)
{
    std::vector<CorrespondenceP2Plane> correspondences;
    correspondences.reserve(source.size());
    
    if (target.size() < 5) {
        return correspondences;
    }
    
    // Build KD-Tree for target cloud
    KDTree kdtree(3, target, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    kdtree.buildIndex();
    
    Eigen::Matrix3f R = current_transform.block<3, 3>(0, 0);
    Eigen::Vector3f t = current_transform.block<3, 1>(0, 3);
    
    const int K = 5;
    const float max_dist_sq = static_cast<float>(max_distance * max_distance);
    const float plane_residual_threshold = 0.3f;
    
    for (size_t i = 0; i < source.size(); ++i) {
        Eigen::Vector3f p_transformed = R * source.points[i].point + t;
        
        std::vector<size_t> indices(K);
        std::vector<float> distances_sq(K);
        float query[3] = {p_transformed.x(), p_transformed.y(), p_transformed.z()};
        
        size_t found = kdtree.knnSearch(query, K, indices.data(), distances_sq.data());
        
        if (found < 5 || distances_sq[0] > max_dist_sq) {
            continue;
        }
        
        // Collect neighbor points for plane fitting
        std::vector<Eigen::Vector3f> neighbors;
        neighbors.reserve(found);
        for (size_t k = 0; k < found; ++k) {
            neighbors.push_back(target.points[indices[k]].point);
        }
        
        // Check collinearity
        if (neighbors.size() >= 3) {
            Eigen::Vector3f v1 = neighbors[1] - neighbors[0];
            Eigen::Vector3f v2 = neighbors[2] - neighbors[0];
            float cross_norm = v1.cross(v2).norm();
            if (cross_norm < 0.01f) {
                continue;
            }
        }
        
        // Fit plane using SVD
        Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
        for (const auto& pt : neighbors) {
            centroid += pt;
        }
        centroid /= static_cast<float>(neighbors.size());
        
        Eigen::MatrixXf A(neighbors.size(), 3);
        for (size_t k = 0; k < neighbors.size(); ++k) {
            A.row(k) = (neighbors[k] - centroid).transpose();
        }
        
        Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullV);
        Eigen::Vector3f plane_normal = svd.matrixV().col(2);
        float plane_d = -plane_normal.dot(centroid);
        
        float residual = std::abs(plane_normal.dot(p_transformed) + plane_d);
        
        if (residual > plane_residual_threshold) {
            continue;
        }
        
        // Accept correspondence (need both source and target normals for Symmetric)
        CorrespondenceP2Plane corr;
        corr.source_idx = static_cast<int>(i);
        corr.target_idx = static_cast<int>(indices[0]);
        corr.target_normal = plane_normal;
        corr.source_normal = source.points[i].normal;  // Precomputed
        corr.distance_sq = distances_sq[0];
        correspondences.push_back(corr);
    }
    
    return correspondences;
}

// ============================================================================
// GICP - Correspondence Finding
// ============================================================================

std::vector<CorrespondenceGICP> GeneralizedICP::findCorrespondencesGICP(
    const PointCloudWithCovariance& source,
    const PointCloudWithCovariance& target,
    const Eigen::Matrix4f& current_transform,
    double max_distance)
{
    std::vector<CorrespondenceGICP> correspondences;
    correspondences.reserve(source.size());
    
    if (target.size() < 5) {
        return correspondences;
    }
    
    // Build KD-Tree for target cloud
    KDTree kdtree(3, target, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    kdtree.buildIndex();
    
    Eigen::Matrix3f R = current_transform.block<3, 3>(0, 0);
    Eigen::Vector3f t = current_transform.block<3, 1>(0, 3);
    
    const int K = 5;
    const float max_dist_sq = static_cast<float>(max_distance * max_distance);
    const float plane_residual_threshold = 0.3f;
    
    for (size_t i = 0; i < source.size(); ++i) {
        Eigen::Vector3f p_transformed = R * source.points[i].point + t;
        
        std::vector<size_t> indices(K);
        std::vector<float> distances_sq(K);
        float query[3] = {p_transformed.x(), p_transformed.y(), p_transformed.z()};
        
        size_t found = kdtree.knnSearch(query, K, indices.data(), distances_sq.data());
        
        if (found < 5 || distances_sq[0] > max_dist_sq) {
            continue;
        }
        
        // Collect neighbor points for plane fitting
        std::vector<Eigen::Vector3f> neighbors;
        neighbors.reserve(found);
        for (size_t k = 0; k < found; ++k) {
            neighbors.push_back(target.points[indices[k]].point);
        }
        
        // Check collinearity
        if (neighbors.size() >= 3) {
            Eigen::Vector3f v1 = neighbors[1] - neighbors[0];
            Eigen::Vector3f v2 = neighbors[2] - neighbors[0];
            float cross_norm = v1.cross(v2).norm();
            if (cross_norm < 0.01f) {
                continue;
            }
        }
        
        // Fit plane using SVD
        Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
        for (const auto& pt : neighbors) {
            centroid += pt;
        }
        centroid /= static_cast<float>(neighbors.size());
        
        Eigen::MatrixXf A(neighbors.size(), 3);
        for (size_t k = 0; k < neighbors.size(); ++k) {
            A.row(k) = (neighbors[k] - centroid).transpose();
        }
        
        Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullV);
        Eigen::Vector3f plane_normal = svd.matrixV().col(2);
        float plane_d = -plane_normal.dot(centroid);
        
        float residual = std::abs(plane_normal.dot(p_transformed) + plane_d);
        
        if (residual > plane_residual_threshold) {
            continue;
        }
        
        // Accept correspondence with covariance information
        CorrespondenceGICP corr;
        corr.source_idx = static_cast<int>(i);
        corr.target_idx = static_cast<int>(indices[0]);
        corr.source_covariance = source.points[i].covariance;
        corr.target_covariance = target.points[indices[0]].covariance;
        corr.source_eigenvalues = source.points[i].eigenvalues;
        corr.source_eigenvectors = source.points[i].eigenvectors;
        corr.target_eigenvalues = target.points[indices[0]].eigenvalues;
        corr.target_eigenvectors = target.points[indices[0]].eigenvectors;
        corr.distance_sq = distances_sq[0];
        correspondences.push_back(corr);
    }
    
    return correspondences;
}

// ============================================================================
// MC-ICP - Correspondence Finding
// ============================================================================

std::vector<CorrespondenceGICP> ManifoldConstrainedICP::findCorrespondencesMCICP(
    const PointCloudWithCovariance& source,
    const PointCloudWithCovariance& target,
    const Eigen::Matrix4f& current_transform,
    double max_distance)
{
    std::vector<CorrespondenceGICP> correspondences;
    correspondences.reserve(source.size());
    
    if (target.size() < 5) {
        return correspondences;
    }
    
    // Build KD-Tree for target cloud
    KDTree kdtree(3, target, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    kdtree.buildIndex();
    
    Eigen::Matrix3f R = current_transform.block<3, 3>(0, 0);
    Eigen::Vector3f t = current_transform.block<3, 1>(0, 3);
    
    const int K = 5;
    const float max_dist_sq = static_cast<float>(max_distance * max_distance);
    const float plane_residual_threshold = 0.3f;
    
    for (size_t i = 0; i < source.size(); ++i) {
        Eigen::Vector3f p_transformed = R * source.points[i].point + t;
        
        std::vector<size_t> indices(K);
        std::vector<float> distances_sq(K);
        float query[3] = {p_transformed.x(), p_transformed.y(), p_transformed.z()};
        
        size_t found = kdtree.knnSearch(query, K, indices.data(), distances_sq.data());
        
        if (found < 5 || distances_sq[0] > max_dist_sq) {
            continue;
        }
        
        // Collect neighbor points for plane fitting
        std::vector<Eigen::Vector3f> neighbors;
        neighbors.reserve(found);
        for (size_t k = 0; k < found; ++k) {
            neighbors.push_back(target.points[indices[k]].point);
        }
        
        // Check collinearity
        if (neighbors.size() >= 3) {
            Eigen::Vector3f v1 = neighbors[1] - neighbors[0];
            Eigen::Vector3f v2 = neighbors[2] - neighbors[0];
            float cross_norm = v1.cross(v2).norm();
            if (cross_norm < 0.01f) {
                continue;
            }
        }
        
        // Fit plane using SVD
        Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
        for (const auto& pt : neighbors) {
            centroid += pt;
        }
        centroid /= static_cast<float>(neighbors.size());
        
        Eigen::MatrixXf A(neighbors.size(), 3);
        for (size_t k = 0; k < neighbors.size(); ++k) {
            A.row(k) = (neighbors[k] - centroid).transpose();
        }
        
        Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullV);
        Eigen::Vector3f plane_normal = svd.matrixV().col(2);
        float plane_d = -plane_normal.dot(centroid);
        
        float residual = std::abs(plane_normal.dot(p_transformed) + plane_d);
        
        if (residual > plane_residual_threshold) {
            continue;
        }
        
        // Accept correspondence with eigenvalue/eigenvector information
        CorrespondenceGICP corr;
        corr.source_idx = static_cast<int>(i);
        corr.target_idx = static_cast<int>(indices[0]);
        corr.source_covariance = source.points[i].covariance;
        corr.target_covariance = target.points[indices[0]].covariance;
        corr.source_eigenvalues = source.points[i].eigenvalues;
        corr.source_eigenvectors = source.points[i].eigenvectors;
        corr.target_eigenvalues = target.points[indices[0]].eigenvalues;
        corr.target_eigenvectors = target.points[indices[0]].eigenvectors;
        corr.distance_sq = distances_sq[0];
        correspondences.push_back(corr);
    }
    
    return correspondences;
}

void ICPAlgorithm::transformPoints(
    const PointCloudWithCovariance& source,
    PointCloudWithCovariance& transformed,
    const Eigen::Matrix4f& transform)
{
    transformed.clear();
    transformed.reserve(source.size());
    
    Eigen::Matrix3f R = transform.block<3, 3>(0, 0);
    Eigen::Vector3f t = transform.block<3, 1>(0, 3);
    
    for (const auto& p : source.points) {
        PointWithCovariance tp;
        tp.point = R * p.point + t;
        tp.normal = R * p.normal;
        tp.covariance = R * p.covariance * R.transpose();
        tp.eigenvalues = p.eigenvalues;
        tp.eigenvectors = R * p.eigenvectors;
        transformed.push_back(tp);
    }
}

// ============================================================================
// Point-to-Point ICP
// ============================================================================

BenchmarkResult PointToPointICP::run(
    const PointCloudWithCovariance& source,
    const PointCloudWithCovariance& target,
    const Eigen::Matrix4f& initial_transform,
    const ICPConfig& config,
    const Eigen::Matrix4f& ground_truth)
{
    BenchmarkResult result;
    result.method = ICPMethod::POINT_TO_POINT;
    result.method_name = "Point-to-Point";
    result.source_points = source.size();
    result.target_points = target.size();
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    Eigen::Matrix4f current_transform = initial_transform;
    
    // Compute initial error
    auto [init_trans_err, init_rot_err] = computePoseError(initial_transform, ground_truth);
    result.initial_translation_error = init_trans_err;
    result.initial_rotation_error = init_rot_err;
    
    for (int iter = 0; iter < config.max_iterations; ++iter) {
        auto iter_start = std::chrono::high_resolution_clock::now();
        
        // Find correspondences using P2P-specific function
        auto correspondences = findCorrespondencesP2P(source, target, current_transform, 
                                                       config.max_correspondence_distance);
        
        if (correspondences.size() < 3) {
            break;
        }
        
        // Extract corresponding points
        Eigen::Matrix3f R = current_transform.block<3, 3>(0, 0);
        Eigen::Vector3f t = current_transform.block<3, 1>(0, 3);
        
        // Compute centroids
        Eigen::Vector3f centroid_source = Eigen::Vector3f::Zero();
        Eigen::Vector3f centroid_target = Eigen::Vector3f::Zero();
        
        for (const auto& corr : correspondences) {
            centroid_source += R * source.points[corr.source_idx].point + t;
            centroid_target += target.points[corr.target_idx].point;
        }
        centroid_source /= static_cast<float>(correspondences.size());
        centroid_target /= static_cast<float>(correspondences.size());
        
        // Build cross-covariance matrix
        Eigen::Matrix3f H = Eigen::Matrix3f::Zero();
        double total_cost = 0.0;
        
        for (const auto& corr : correspondences) {
            Eigen::Vector3f p_source = R * source.points[corr.source_idx].point + t - centroid_source;
            Eigen::Vector3f p_target = target.points[corr.target_idx].point - centroid_target;
            H += p_source * p_target.transpose();
            
            Eigen::Vector3f diff = (R * source.points[corr.source_idx].point + t) - target.points[corr.target_idx].point;
            total_cost += diff.squaredNorm();
        }
        
        // SVD for optimal rotation
        Eigen::JacobiSVD<Eigen::Matrix3f> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3f R_update = svd.matrixV() * svd.matrixU().transpose();
        
        // Handle reflection case
        if (R_update.determinant() < 0) {
            Eigen::Matrix3f V = svd.matrixV();
            V.col(2) *= -1;
            R_update = V * svd.matrixU().transpose();
        }
        
        Eigen::Vector3f t_update = centroid_target - R_update * centroid_source;
        
        // Apply update
        Eigen::Matrix4f delta = Eigen::Matrix4f::Identity();
        delta.block<3, 3>(0, 0) = R_update;
        delta.block<3, 1>(0, 3) = t_update;
        
        Eigen::Matrix4f new_transform = delta * current_transform;
        
        // Check convergence
        Eigen::Vector3f dt = new_transform.block<3, 1>(0, 3) - current_transform.block<3, 1>(0, 3);
        double step_size = dt.norm();
        
        current_transform = new_transform;
        
        // Record iteration stats
        auto iter_end = std::chrono::high_resolution_clock::now();
        double iter_time = std::chrono::duration<double, std::milli>(iter_end - start_time).count();
        
        auto [trans_err, rot_err] = computePoseError(current_transform, ground_truth);
        
        IterationStats stats;
        stats.iteration = iter;
        stats.cost = total_cost;
        stats.translation_error = trans_err;
        stats.rotation_error = rot_err;
        stats.step_size = step_size;
        stats.time_ms = iter_time;
        stats.num_correspondences = correspondences.size();
        result.history.push_back(stats);
        
        // Verbose output
        if (config.verbose) {
            std::cout << "    " << std::setw(4) << iter << " | "
                      << std::setw(13) << std::fixed << std::setprecision(6) << trans_err << " | "
                      << std::setw(13) << std::setprecision(6) << rot_err << " | "
                      << std::setprecision(2) << total_cost << std::endl;
        }
        
        // Convergence check
        if (step_size < config.convergence_threshold) {
            result.converged = true;
            if (config.verbose) {
                Eigen::Vector3f t_est = current_transform.block<3,1>(0,3);
                std::cout << "    Converged! Est trans: [" << t_est.x() << ", " << t_est.y() << ", " << t_est.z() << "]" << std::endl;
            }
            break;
        }
    }
    
    // Print final transform
    if (config.verbose) {
        Eigen::Vector3f t_final = current_transform.block<3,1>(0,3);
        std::cout << "    Final est trans: [" << t_final.x() << ", " << t_final.y() << ", " << t_final.z() << "]" << std::endl;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.total_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    result.total_iterations = static_cast<int>(result.history.size());
    
    if (!result.history.empty()) {
        result.final_cost = result.history.back().cost;
        result.final_translation_error = result.history.back().translation_error;
        result.final_rotation_error = result.history.back().rotation_error;
    }
    
    return result;
}

// ============================================================================
// Point-to-Plane ICP
// ============================================================================

BenchmarkResult PointToPlaneICP::run(
    const PointCloudWithCovariance& source,
    const PointCloudWithCovariance& target,
    const Eigen::Matrix4f& initial_transform,
    const ICPConfig& config,
    const Eigen::Matrix4f& ground_truth)
{
    BenchmarkResult result;
    result.method = ICPMethod::POINT_TO_PLANE;
    result.method_name = "Point-to-Plane";
    result.source_points = source.size();
    result.target_points = target.size();
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    Eigen::Matrix4f current_transform = initial_transform;
    
    // Compute initial error
    auto [init_trans_err, init_rot_err] = computePoseError(initial_transform, ground_truth);
    result.initial_translation_error = init_trans_err;
    result.initial_rotation_error = init_rot_err;
    
    for (int iter = 0; iter < config.max_iterations; ++iter) {
        // Find correspondences using P2Plane-specific function
        auto correspondences = findCorrespondencesP2Plane(source, target, current_transform,
                                                           config.max_correspondence_distance);
        
        if (correspondences.size() < 6) {
            break;
        }
        
        Eigen::Matrix3f R = current_transform.block<3, 3>(0, 0);
        Eigen::Vector3f t = current_transform.block<3, 1>(0, 3);
        
        // Build normal equation: H * delta = -g
        // Using Gauss-Newton with right perturbation on SE(3)
        Eigen::Matrix<float, 6, 6> H = Eigen::Matrix<float, 6, 6>::Zero();
        Eigen::Matrix<float, 6, 1> g = Eigen::Matrix<float, 6, 1>::Zero();
        double total_cost = 0.0;
        
        for (const auto& corr : correspondences) {
            Eigen::Vector3f p_source = source.points[corr.source_idx].point;
            Eigen::Vector3f p_transformed = R * p_source + t;
            Eigen::Vector3f p_target = target.points[corr.target_idx].point;
            Eigen::Vector3f n = corr.target_normal;  // Use precomputed normal from correspondence
            
            // Point-to-plane residual: r = n^T * (p_transformed - p_target)
            float residual = n.dot(p_transformed - p_target);
            total_cost += residual * residual;
            
            // Jacobian w.r.t. pose (right perturbation)
            // J = [n^T * R, -n^T * R * [p_source]_x]
            // where [p]_x is skew-symmetric matrix
            Eigen::Matrix<float, 1, 6> J;
            J.block<1, 3>(0, 0) = n.transpose() * R;  // Fixed: was n.transpose()
            
            // Skew-symmetric matrix for cross product
            Eigen::Matrix3f p_skew;
            p_skew <<     0, -p_source.z(),  p_source.y(),
                      p_source.z(),     0, -p_source.x(),
                     -p_source.y(),  p_source.x(),     0;
            
            J.block<1, 3>(0, 3) = -n.transpose() * R * p_skew;
            
            // Accumulate normal equation
            H += J.transpose() * J;
            g += residual * J.transpose();
        }
        
        // Solve H * delta = -g
        Eigen::Matrix<float, 6, 1> delta = H.ldlt().solve(-g);
        
        Eigen::Vector3f dt = delta.head<3>();
        Eigen::Vector3f dw = delta.tail<3>();
        
        // Convert axis-angle to rotation matrix (Rodrigues' formula)
        float angle = dw.norm();
        Eigen::Matrix3f dR = Eigen::Matrix3f::Identity();
        if (angle > 1e-10f) {
            Eigen::Vector3f axis = dw / angle;
            Eigen::Matrix3f K;
            K <<     0, -axis.z(),  axis.y(),
                 axis.z(),     0, -axis.x(),
                -axis.y(),  axis.x(),     0;
            dR = Eigen::Matrix3f::Identity() + std::sin(angle) * K + (1 - std::cos(angle)) * K * K;
        }
        
        // Update transform (right multiplication)
        Eigen::Matrix4f delta_transform = Eigen::Matrix4f::Identity();
        delta_transform.block<3, 3>(0, 0) = dR;
        delta_transform.block<3, 1>(0, 3) = dt;
        
        current_transform = current_transform * delta_transform;
        
        // Record iteration stats
        auto iter_end = std::chrono::high_resolution_clock::now();
        double iter_time = std::chrono::duration<double, std::milli>(iter_end - start_time).count();
        
        auto [trans_err, rot_err] = computePoseError(current_transform, ground_truth);
        
        IterationStats stats;
        stats.iteration = iter;
        stats.cost = total_cost;
        stats.translation_error = trans_err;
        stats.rotation_error = rot_err;
        stats.step_size = delta.norm();
        stats.time_ms = iter_time;
        stats.num_correspondences = correspondences.size();
        result.history.push_back(stats);
        
        // Verbose output
        if (config.verbose) {
            std::cout << "    " << std::setw(4) << iter << " | "
                      << std::setw(13) << std::fixed << std::setprecision(6) << trans_err << " | "
                      << std::setw(13) << std::setprecision(6) << rot_err << " | "
                      << std::setprecision(2) << total_cost << std::endl;
        }
        
        // Convergence check
        if (delta.head<3>().norm() < config.translation_tolerance &&
            delta.tail<3>().norm() < config.rotation_tolerance) {
            result.converged = true;
            if (config.verbose) {
                Eigen::Vector3f t_est = current_transform.block<3,1>(0,3);
                std::cout << "    Converged! Est trans: [" << t_est.x() << ", " << t_est.y() << ", " << t_est.z() << "]" << std::endl;
            }
            break;
        }
    }
    
    // Print final transform
    if (config.verbose) {
        Eigen::Vector3f t_final = current_transform.block<3,1>(0,3);
        std::cout << "    Final est trans: [" << t_final.x() << ", " << t_final.y() << ", " << t_final.z() << "]" << std::endl;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.total_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    result.total_iterations = static_cast<int>(result.history.size());
    
    if (!result.history.empty()) {
        result.final_cost = result.history.back().cost;
        result.final_translation_error = result.history.back().translation_error;
        result.final_rotation_error = result.history.back().rotation_error;
    }
    
    return result;
}

// ============================================================================
// Symmetric ICP
// ============================================================================

BenchmarkResult SymmetricICP::run(
    const PointCloudWithCovariance& source,
    const PointCloudWithCovariance& target,
    const Eigen::Matrix4f& initial_transform,
    const ICPConfig& config,
    const Eigen::Matrix4f& ground_truth)
{
    BenchmarkResult result;
    result.method = ICPMethod::SYMMETRIC;
    result.method_name = "Symmetric";
    result.source_points = source.size();
    result.target_points = target.size();
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    Eigen::Matrix4f current_transform = initial_transform;
    
    auto [init_trans_err, init_rot_err] = computePoseError(initial_transform, ground_truth);
    result.initial_translation_error = init_trans_err;
    result.initial_rotation_error = init_rot_err;
    
    for (int iter = 0; iter < config.max_iterations; ++iter) {
        auto correspondences = findCorrespondencesSymmetric(source, target, current_transform,
                                                             config.max_correspondence_distance);
        
        if (correspondences.size() < 6) {
            break;
        }
        
        Eigen::Matrix3f R = current_transform.block<3, 3>(0, 0);
        Eigen::Vector3f t = current_transform.block<3, 1>(0, 3);
        
        Eigen::Matrix<float, 6, 6> H = Eigen::Matrix<float, 6, 6>::Zero();
        Eigen::Matrix<float, 6, 1> g = Eigen::Matrix<float, 6, 1>::Zero();
        double total_cost = 0.0;
        
        for (const auto& corr : correspondences) {
            Eigen::Vector3f p_source = source.points[corr.source_idx].point;
            Eigen::Vector3f p_transformed = R * p_source + t;
            Eigen::Vector3f p_target = target.points[corr.target_idx].point;
            
            // Transform source normal to current frame
            Eigen::Vector3f n_source = R * corr.source_normal;
            Eigen::Vector3f n_target = corr.target_normal;
            
            // Symmetric normal: (n_p + n_q) WITHOUT normalization
            // Normalization makes Jacobian computation complex
            Eigen::Vector3f n_sym = n_source + n_target;
            
            // Skip if normals are opposite (degenerate case)
            if (n_sym.squaredNorm() < 1e-6f) continue;
            
            // Residual: n_sym^T * (p_transformed - p_target)
            float residual = n_sym.dot(p_transformed - p_target);
            total_cost += residual * residual;
            
            // Jacobian (same form as Point-to-Plane with n_sym instead of n)
            // J = [n_sym^T * R, -n_sym^T * R * [p_source]_x]
            Eigen::Matrix<float, 1, 6> J;
            J.block<1, 3>(0, 0) = n_sym.transpose() * R;  // Fixed: was n_sym.transpose()
            
            Eigen::Matrix3f p_skew;
            p_skew <<     0, -p_source.z(),  p_source.y(),
                      p_source.z(),     0, -p_source.x(),
                     -p_source.y(),  p_source.x(),     0;
            
            J.block<1, 3>(0, 3) = -n_sym.transpose() * R * p_skew;
            
            H += J.transpose() * J;
            g += residual * J.transpose();
        }
        
        Eigen::Matrix<float, 6, 1> delta = H.ldlt().solve(-g);
        
        Eigen::Vector3f dt = delta.head<3>();
        Eigen::Vector3f dw = delta.tail<3>();
        
        float angle = dw.norm();
        Eigen::Matrix3f dR = Eigen::Matrix3f::Identity();
        if (angle > 1e-10f) {
            Eigen::Vector3f axis = dw / angle;
            Eigen::Matrix3f K;
            K <<     0, -axis.z(),  axis.y(),
                 axis.z(),     0, -axis.x(),
                -axis.y(),  axis.x(),     0;
            dR = Eigen::Matrix3f::Identity() + std::sin(angle) * K + (1 - std::cos(angle)) * K * K;
        }
        
        Eigen::Matrix4f delta_transform = Eigen::Matrix4f::Identity();
        delta_transform.block<3, 3>(0, 0) = dR;
        delta_transform.block<3, 1>(0, 3) = dt;
        
        current_transform = current_transform * delta_transform;
        
        auto iter_end = std::chrono::high_resolution_clock::now();
        double iter_time = std::chrono::duration<double, std::milli>(iter_end - start_time).count();
        
        auto [trans_err, rot_err] = computePoseError(current_transform, ground_truth);
        
        IterationStats stats;
        stats.iteration = iter;
        stats.cost = total_cost;
        stats.translation_error = trans_err;
        stats.rotation_error = rot_err;
        stats.step_size = delta.norm();
        stats.time_ms = iter_time;
        stats.num_correspondences = correspondences.size();
        result.history.push_back(stats);
        
        // Verbose output
        if (config.verbose) {
            std::cout << "    " << std::setw(4) << iter << " | "
                      << std::setw(13) << std::fixed << std::setprecision(6) << trans_err << " | "
                      << std::setw(13) << std::setprecision(6) << rot_err << " | "
                      << std::setprecision(2) << total_cost << std::endl;
        }
        
        if (delta.head<3>().norm() < config.translation_tolerance &&
            delta.tail<3>().norm() < config.rotation_tolerance) {
            result.converged = true;
            break;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.total_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    result.total_iterations = static_cast<int>(result.history.size());
    
    if (!result.history.empty()) {
        result.final_cost = result.history.back().cost;
        result.final_translation_error = result.history.back().translation_error;
        result.final_rotation_error = result.history.back().rotation_error;
    }
    
    return result;
}

// ============================================================================
// GICP
// ============================================================================

BenchmarkResult GeneralizedICP::run(
    const PointCloudWithCovariance& source,
    const PointCloudWithCovariance& target,
    const Eigen::Matrix4f& initial_transform,
    const ICPConfig& config,
    const Eigen::Matrix4f& ground_truth)
{
    BenchmarkResult result;
    result.method = ICPMethod::GICP;
    result.method_name = "GICP";
    result.source_points = source.size();
    result.target_points = target.size();
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    Eigen::Matrix4f current_transform = initial_transform;
    
    auto [init_trans_err, init_rot_err] = computePoseError(initial_transform, ground_truth);
    result.initial_translation_error = init_trans_err;
    result.initial_rotation_error = init_rot_err;
    
    for (int iter = 0; iter < config.max_iterations; ++iter) {
        auto correspondences = findCorrespondencesGICP(source, target, current_transform,
                                                        config.max_correspondence_distance);
        
        if (correspondences.size() < 6) {
            break;
        }
        
        Eigen::Matrix3f R = current_transform.block<3, 3>(0, 0);
        Eigen::Vector3f t = current_transform.block<3, 1>(0, 3);
        
        Eigen::Matrix<float, 6, 6> H = Eigen::Matrix<float, 6, 6>::Zero();
        Eigen::Matrix<float, 6, 1> g = Eigen::Matrix<float, 6, 1>::Zero();
        double total_cost = 0.0;
        
        // GICP epsilon for planar covariance (original paper uses ~0.001)
        const float gicp_epsilon = 0.001f;
        
        for (const auto& corr : correspondences) {
            Eigen::Vector3f p_source = source.points[corr.source_idx].point;
            Eigen::Vector3f p_transformed = R * p_source + t;
            Eigen::Vector3f p_target = target.points[corr.target_idx].point;
            
            // GICP original paper: construct planar covariance from surface normal
            // C = R_n * diag(eps, 1, 1) * R_n^T where R_n aligns z-axis with normal
            auto makePlanarCovariance = [gicp_epsilon](const Eigen::Vector3f& normal) -> Eigen::Matrix3f {
                // Build rotation matrix that aligns z-axis with normal
                Eigen::Vector3f n = normal.normalized();
                Eigen::Vector3f arbitrary = (std::abs(n.x()) < 0.9f) ? 
                    Eigen::Vector3f(1, 0, 0) : Eigen::Vector3f(0, 1, 0);
                Eigen::Vector3f u = n.cross(arbitrary).normalized();
                Eigen::Vector3f v = n.cross(u);
                
                Eigen::Matrix3f R_n;
                R_n.col(0) = u;      // tangent 1
                R_n.col(1) = v;      // tangent 2  
                R_n.col(2) = n;      // normal
                
                // Eigenvalues: large variance in tangent directions, small in normal direction
                Eigen::Vector3f eigenvalues(1.0f, 1.0f, gicp_epsilon);
                
                return R_n * eigenvalues.asDiagonal() * R_n.transpose();
            };
            
            // Get normals (smallest eigenvector direction)
            Eigen::Vector3f n_source = source.points[corr.source_idx].normal;
            Eigen::Vector3f n_target = target.points[corr.target_idx].normal;
            
            Eigen::Matrix3f C_source = makePlanarCovariance(n_source);
            Eigen::Matrix3f C_target = makePlanarCovariance(n_target);
            
            // Combined covariance: C = C_target + R * C_source * R^T
            Eigen::Matrix3f C_combined = C_target + R * C_source * R.transpose();
            
            // Information matrix (inverse of combined covariance)
            Eigen::Matrix3f Omega = C_combined.inverse();
            
            // Residual
            Eigen::Vector3f d = p_transformed - p_target;
            total_cost += d.transpose() * Omega * d;
            
            // Jacobian w.r.t. pose (right perturbation)
            // p' = p_transformed + R*δt - R*[p_source]_x*δω
            // ∂p'/∂δt = R, ∂p'/∂δω = -R*[p_source]_x
            Eigen::Matrix<float, 3, 6> J;
            J.block<3, 3>(0, 0) = R;  // Correct for right perturbation
            
            Eigen::Matrix3f p_skew;
            p_skew <<     0, -p_source.z(),  p_source.y(),
                      p_source.z(),     0, -p_source.x(),
                     -p_source.y(),  p_source.x(),     0;
            
            J.block<3, 3>(0, 3) = -R * p_skew;
            
            // Weighted normal equation
            H += J.transpose() * Omega * J;
            g += J.transpose() * Omega * d;
        }
        
        Eigen::Matrix<float, 6, 1> delta = H.ldlt().solve(-g);
        
        Eigen::Vector3f dt = delta.head<3>();
        Eigen::Vector3f dw = delta.tail<3>();
        
        float angle = dw.norm();
        Eigen::Matrix3f dR = Eigen::Matrix3f::Identity();
        if (angle > 1e-10f) {
            Eigen::Vector3f axis = dw / angle;
            Eigen::Matrix3f K;
            K <<     0, -axis.z(),  axis.y(),
                 axis.z(),     0, -axis.x(),
                -axis.y(),  axis.x(),     0;
            dR = Eigen::Matrix3f::Identity() + std::sin(angle) * K + (1 - std::cos(angle)) * K * K;
        }
        
        Eigen::Matrix4f delta_transform = Eigen::Matrix4f::Identity();
        delta_transform.block<3, 3>(0, 0) = dR;
        delta_transform.block<3, 1>(0, 3) = dt;
        
        current_transform = current_transform * delta_transform;
        
        auto iter_end = std::chrono::high_resolution_clock::now();
        double iter_time = std::chrono::duration<double, std::milli>(iter_end - start_time).count();
        
        auto [trans_err, rot_err] = computePoseError(current_transform, ground_truth);
        
        IterationStats stats;
        stats.iteration = iter;
        stats.cost = total_cost;
        stats.translation_error = trans_err;
        stats.rotation_error = rot_err;
        stats.step_size = delta.norm();
        stats.time_ms = iter_time;
        stats.num_correspondences = correspondences.size();
        result.history.push_back(stats);
        
        // Verbose output
        if (config.verbose) {
            std::cout << "    " << std::setw(4) << iter << " | "
                      << std::setw(13) << std::fixed << std::setprecision(6) << trans_err << " | "
                      << std::setw(13) << std::setprecision(6) << rot_err << " | "
                      << std::setprecision(2) << total_cost << std::endl;
        }
        
        if (delta.head<3>().norm() < config.translation_tolerance &&
            delta.tail<3>().norm() < config.rotation_tolerance) {
            result.converged = true;
            break;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.total_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    result.total_iterations = static_cast<int>(result.history.size());
    
    if (!result.history.empty()) {
        result.final_cost = result.history.back().cost;
        result.final_translation_error = result.history.back().translation_error;
        result.final_rotation_error = result.history.back().rotation_error;
    }
    
    return result;
}

// ============================================================================
// MC-ICP (Manifold-Constrained ICP) - Our Method
// ============================================================================

Eigen::Matrix3f ManifoldConstrainedICP::computeGeometryAwareOmega(
    const Eigen::Vector3f& eigenvalues,
    const Eigen::Matrix3f& eigenvectors)
{
    float trace = eigenvalues.sum();
    if (trace < 1e-10f) {
        return Eigen::Matrix3f::Identity();
    }
    
    // Constraint strength: c_i = 1 - λ_i / tr(Σ)
    // Small eigenvalue (low variance) → c_i ≈ 1 (strong constraint)
    // Large eigenvalue (high variance) → c_i ≈ 0 (weak constraint)
    Eigen::Vector3f c;
    c[0] = 1.0f - eigenvalues[0] / trace;
    c[1] = 1.0f - eigenvalues[1] / trace;
    c[2] = 1.0f - eigenvalues[2] / trace;
    
    // Information weight: c_i / λ_i (higher for small λ_i)
    // Add regularization to avoid division by zero
    const float eps = 1e-4f * trace;  // Scale-aware regularization
    Eigen::Vector3f info;
    info[0] = c[0] / (eigenvalues[0] + eps);
    info[1] = c[1] / (eigenvalues[1] + eps);
    info[2] = c[2] / (eigenvalues[2] + eps);
    
    // Ω = V * diag(info) * V^T
    return eigenvectors * info.asDiagonal() * eigenvectors.transpose();
}

BenchmarkResult ManifoldConstrainedICP::run(
    const PointCloudWithCovariance& source,
    const PointCloudWithCovariance& target,
    const Eigen::Matrix4f& initial_transform,
    const ICPConfig& config,
    const Eigen::Matrix4f& ground_truth)
{
    BenchmarkResult result;
    result.method = ICPMethod::MC_ICP;
    result.method_name = "MC-ICP";
    result.source_points = source.size();
    result.target_points = target.size();
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    Eigen::Matrix4f current_transform = initial_transform;
    
    auto [init_trans_err, init_rot_err] = computePoseError(initial_transform, ground_truth);
    result.initial_translation_error = init_trans_err;
    result.initial_rotation_error = init_rot_err;
    
    for (int iter = 0; iter < config.max_iterations; ++iter) {
        auto correspondences = findCorrespondencesMCICP(source, target, current_transform,
                                                         config.max_correspondence_distance);
        
        if (correspondences.size() < 6) {
            break;
        }
        
        Eigen::Matrix3f R = current_transform.block<3, 3>(0, 0);
        Eigen::Vector3f t = current_transform.block<3, 1>(0, 3);
        
        Eigen::Matrix<float, 6, 6> H = Eigen::Matrix<float, 6, 6>::Zero();
        Eigen::Matrix<float, 6, 1> g = Eigen::Matrix<float, 6, 1>::Zero();
        double total_cost = 0.0;
        
        for (const auto& corr : correspondences) {
            Eigen::Vector3f p_source = source.points[corr.source_idx].point;
            Eigen::Vector3f p_transformed = R * p_source + t;
            Eigen::Vector3f p_target = target.points[corr.target_idx].point;
            
            // Geometry-aware information matrices
            Eigen::Matrix3f Omega_source = computeGeometryAwareOmega(
                corr.source_eigenvalues,
                corr.source_eigenvectors);
            Eigen::Matrix3f Omega_target = computeGeometryAwareOmega(
                corr.target_eigenvalues,
                corr.target_eigenvectors);
            
            // Transform source information to world frame and combine via covariance
            // Ω_combined = (Ω_target^(-1) + R * Ω_source^(-1) * R^T)^(-1)
            Eigen::Matrix3f Cov_source = (Omega_source + 1e-6f * Eigen::Matrix3f::Identity()).inverse();
            Eigen::Matrix3f Cov_target = (Omega_target + 1e-6f * Eigen::Matrix3f::Identity()).inverse();
            Eigen::Matrix3f Cov_combined = Cov_target + R * Cov_source * R.transpose();
            
            // Safe inverse with regularization
            Eigen::Matrix3f Omega = (Cov_combined + 1e-6f * Eigen::Matrix3f::Identity()).inverse();
            
            // Check for NaN and fallback to identity
            if (!Omega.allFinite()) {
                Omega = Eigen::Matrix3f::Identity();
            }
            
            // Residual
            Eigen::Vector3f d = p_transformed - p_target;
            total_cost += d.transpose() * Omega * d;
            
            // Jacobian w.r.t. pose (right perturbation)
            // p' = p_transformed + R*δt - R*[p_source]_x*δω
            Eigen::Matrix<float, 3, 6> J;
            J.block<3, 3>(0, 0) = R;  // Correct for right perturbation
            
            Eigen::Matrix3f p_skew;
            p_skew <<     0, -p_source.z(),  p_source.y(),
                      p_source.z(),     0, -p_source.x(),
                     -p_source.y(),  p_source.x(),     0;
            
            J.block<3, 3>(0, 3) = -R * p_skew;
            
            // Weighted normal equation
            H += J.transpose() * Omega * J;
            g += J.transpose() * Omega * d;
        }
        
        // Regularize H to ensure invertibility for MC-ICP
        H += 1e-6f * Eigen::Matrix<float, 6, 6>::Identity();
        Eigen::Matrix<float, 6, 1> delta = H.ldlt().solve(-g);
        
        // Check for NaN in delta - skip if invalid
        if (!delta.allFinite()) {
            break;
        }
        
        Eigen::Vector3f dt = delta.head<3>();
        Eigen::Vector3f dw = delta.tail<3>();
        
        float angle = dw.norm();
        Eigen::Matrix3f dR = Eigen::Matrix3f::Identity();
        if (angle > 1e-10f) {
            Eigen::Vector3f axis = dw / angle;
            Eigen::Matrix3f K;
            K <<     0, -axis.z(),  axis.y(),
                 axis.z(),     0, -axis.x(),
                -axis.y(),  axis.x(),     0;
            dR = Eigen::Matrix3f::Identity() + std::sin(angle) * K + (1 - std::cos(angle)) * K * K;
        }
        
        Eigen::Matrix4f delta_transform = Eigen::Matrix4f::Identity();
        delta_transform.block<3, 3>(0, 0) = dR;
        delta_transform.block<3, 1>(0, 3) = dt;
        
        current_transform = current_transform * delta_transform;
        
        auto iter_end = std::chrono::high_resolution_clock::now();
        double iter_time = std::chrono::duration<double, std::milli>(iter_end - start_time).count();
        
        auto [trans_err, rot_err] = computePoseError(current_transform, ground_truth);
        
        IterationStats stats;
        stats.iteration = iter;
        stats.cost = total_cost;
        stats.translation_error = trans_err;
        stats.rotation_error = rot_err;
        stats.step_size = delta.norm();
        stats.time_ms = iter_time;
        stats.num_correspondences = correspondences.size();
        result.history.push_back(stats);
        
        // Verbose output
        if (config.verbose) {
            std::cout << "    " << std::setw(4) << iter << " | "
                      << std::setw(13) << std::fixed << std::setprecision(6) << trans_err << " | "
                      << std::setw(13) << std::setprecision(6) << rot_err << " | "
                      << std::setprecision(2) << total_cost << std::endl;
        }
        
        if (delta.head<3>().norm() < config.translation_tolerance &&
            delta.tail<3>().norm() < config.rotation_tolerance) {
            result.converged = true;
            break;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.total_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    result.total_iterations = static_cast<int>(result.history.size());
    
    if (!result.history.empty()) {
        result.final_cost = result.history.back().cost;
        result.final_translation_error = result.history.back().translation_error;
        result.final_rotation_error = result.history.back().rotation_error;
    }
    
    return result;
}

// ============================================================================
// NDT (Normal Distributions Transform) Implementation
// ============================================================================

NDTVoxelMap::NDTVoxelMap(float voxel_size, int min_points_per_voxel)
    : voxel_size_(voxel_size),
      inv_voxel_size_(1.0f / voxel_size),
      min_points_per_voxel_(min_points_per_voxel)
{
}

NDTVoxelKey NDTVoxelMap::getVoxelKey(const Eigen::Vector3f& point) const
{
    return NDTVoxelKey(
        static_cast<int>(std::floor(point.x() * inv_voxel_size_)),
        static_cast<int>(std::floor(point.y() * inv_voxel_size_)),
        static_cast<int>(std::floor(point.z() * inv_voxel_size_))
    );
}

void NDTVoxelMap::build(const PointCloudWithCovariance& cloud)
{
    voxels_.clear();
    
    // First pass: accumulate points into voxels
    for (size_t i = 0; i < cloud.size(); ++i) {
        NDTVoxelKey key = getVoxelKey(cloud.points[i].point);
        NDTVoxel& voxel = voxels_[key];
        voxel.sum += cloud.points[i].point;
        voxel.sum_sq += cloud.points[i].point * cloud.points[i].point.transpose();
        voxel.point_count++;
    }
    
    // Second pass: compute mean and covariance
    finalizeVoxels();
}

void NDTVoxelMap::finalizeVoxels()
{
    for (auto& [key, voxel] : voxels_) {
        if (voxel.point_count < min_points_per_voxel_) {
            voxel.valid = false;
            continue;
        }
        
        float n = static_cast<float>(voxel.point_count);
        
        // Mean
        voxel.mean = voxel.sum / n;
        
        // Covariance: E[xx^T] - E[x]E[x]^T
        voxel.covariance = voxel.sum_sq / n - voxel.mean * voxel.mean.transpose();
        
        // Regularize covariance to ensure positive definiteness
        // Add regularization proportional to voxel size squared
        // This ensures covariance is not too small relative to voxel size
        float voxel_size_sq = voxel_size_ * voxel_size_;  // 0.25 for 0.5m voxel
        voxel.covariance += 0.1f * voxel_size_sq * Eigen::Matrix3f::Identity();
        
        // Compute eigendecomposition for regularization
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(voxel.covariance);
        Eigen::Vector3f eigenvalues = solver.eigenvalues();
        Eigen::Matrix3f eigenvectors = solver.eigenvectors();
        
        // Clamp minimum eigenvalue to avoid singularity
        // Minimum eigenvalue should be at least 1% of voxel size squared
        float min_eigenvalue = 0.01f * voxel_size_sq;
        for (int i = 0; i < 3; ++i) {
            if (eigenvalues[i] < min_eigenvalue) {
                eigenvalues[i] = min_eigenvalue;
            }
        }
        
        // Reconstruct covariance with regularized eigenvalues
        voxel.covariance = eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
        
        // Compute information matrix (inverse of covariance)
        voxel.information = voxel.covariance.inverse();
        
        // Check for valid information matrix
        if (!voxel.information.allFinite()) {
            voxel.information = Eigen::Matrix3f::Identity();
        }
        
        voxel.valid = true;
    }
}

const NDTVoxel* NDTVoxelMap::lookup(const Eigen::Vector3f& point) const
{
    NDTVoxelKey key = getVoxelKey(point);
    auto it = voxels_.find(key);
    if (it != voxels_.end() && it->second.valid) {
        return &(it->second);
    }
    return nullptr;
}

size_t NDTVoxelMap::getValidVoxelCount() const
{
    size_t count = 0;
    for (const auto& [key, voxel] : voxels_) {
        if (voxel.valid) count++;
    }
    return count;
}

NDTICP::NDTICP(float ndt_voxel_size)
    : ndt_voxel_size_(ndt_voxel_size)
{
}

BenchmarkResult NDTICP::run(
    const PointCloudWithCovariance& source,
    const PointCloudWithCovariance& target,
    const Eigen::Matrix4f& initial_transform,
    const ICPConfig& config,
    const Eigen::Matrix4f& ground_truth)
{
    BenchmarkResult result;
    result.method = ICPMethod::NDT;
    result.method_name = "NDT";
    result.source_points = source.size();
    result.target_points = target.size();
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Build NDT voxel map from target cloud
    NDTVoxelMap ndt_map(ndt_voxel_size_, 5);
    ndt_map.build(target);
    
    Eigen::Matrix4f current_transform = initial_transform;
    
    // Compute initial error
    auto [init_trans_err, init_rot_err] = computePoseError(initial_transform, ground_truth);
    result.initial_translation_error = init_trans_err;
    result.initial_rotation_error = init_rot_err;
    
    for (int iter = 0; iter < config.max_iterations; ++iter) {
        Eigen::Matrix3f R = current_transform.block<3, 3>(0, 0);
        Eigen::Vector3f t = current_transform.block<3, 1>(0, 3);
        
        // Build normal equation: H * delta = -g
        Eigen::Matrix<float, 6, 6> H = Eigen::Matrix<float, 6, 6>::Zero();
        Eigen::Matrix<float, 6, 1> g = Eigen::Matrix<float, 6, 1>::Zero();
        double total_cost = 0.0;
        size_t num_correspondences = 0;
        
        for (size_t i = 0; i < source.size(); ++i) {
            Eigen::Vector3f p_source = source.points[i].point;
            Eigen::Vector3f p_transformed = R * p_source + t;
            
            // Look up voxel for transformed point
            const NDTVoxel* voxel = ndt_map.lookup(p_transformed);
            if (voxel == nullptr) {
                continue;
            }
            
            // NDT residual: d = p_transformed - mean
            Eigen::Vector3f d = p_transformed - voxel->mean;
            
            // Check distance threshold
            float dist_sq = d.squaredNorm();
            if (dist_sq > config.max_correspondence_distance * config.max_correspondence_distance) {
                continue;
            }
            
            // NDT cost: d^T * Σ^{-1} * d
            float cost = d.transpose() * voxel->information * d;
            total_cost += cost;
            num_correspondences++;
            
            // Jacobian w.r.t. pose (right perturbation)
            // p' = R * p_source + t
            // ∂p'/∂δt = R, ∂p'/∂δω = -R * [p_source]_x
            Eigen::Matrix<float, 3, 6> J;
            J.block<3, 3>(0, 0) = R;
            
            Eigen::Matrix3f p_skew;
            p_skew <<     0, -p_source.z(),  p_source.y(),
                      p_source.z(),     0, -p_source.x(),
                     -p_source.y(),  p_source.x(),     0;
            
            J.block<3, 3>(0, 3) = -R * p_skew;
            
            // Weighted normal equation
            H += J.transpose() * voxel->information * J;
            g += J.transpose() * voxel->information * d;
        }
        
        if (num_correspondences < 6) {
            break;
        }
        
        // Regularize and solve
        H += 1e-6f * Eigen::Matrix<float, 6, 6>::Identity();
        Eigen::Matrix<float, 6, 1> delta = H.ldlt().solve(-g);
        
        // Debug output for first iteration
        if (iter == 0 && config.verbose) {
            std::cout << "[NDT iter0] corr=" << num_correspondences 
                      << ", cost=" << total_cost 
                      << ", |delta|=" << delta.norm() << std::endl;
        }
        
        // Check for NaN
        if (!delta.allFinite()) {
            break;
        }
        
        Eigen::Vector3f dt = delta.head<3>();
        Eigen::Vector3f dw = delta.tail<3>();
        
        // Convert axis-angle to rotation matrix (Rodrigues' formula)
        float angle = dw.norm();
        Eigen::Matrix3f dR = Eigen::Matrix3f::Identity();
        if (angle > 1e-10f) {
            Eigen::Vector3f axis = dw / angle;
            Eigen::Matrix3f K;
            K <<     0, -axis.z(),  axis.y(),
                 axis.z(),     0, -axis.x(),
                -axis.y(),  axis.x(),     0;
            dR = Eigen::Matrix3f::Identity() + std::sin(angle) * K + (1 - std::cos(angle)) * K * K;
        }
        
        // Update transform (right multiplication)
        Eigen::Matrix4f delta_transform = Eigen::Matrix4f::Identity();
        delta_transform.block<3, 3>(0, 0) = dR;
        delta_transform.block<3, 1>(0, 3) = dt;
        
        current_transform = current_transform * delta_transform;
        
        // Record iteration stats
        auto iter_end = std::chrono::high_resolution_clock::now();
        double iter_time = std::chrono::duration<double, std::milli>(iter_end - start_time).count();
        
        auto [trans_err, rot_err] = computePoseError(current_transform, ground_truth);
        
        IterationStats stats;
        stats.iteration = iter;
        stats.cost = total_cost;
        stats.translation_error = trans_err;
        stats.rotation_error = rot_err;
        stats.step_size = delta.norm();
        stats.time_ms = iter_time;
        stats.num_correspondences = num_correspondences;
        result.history.push_back(stats);
        
        // Verbose output
        if (config.verbose) {
            std::cout << "    " << std::setw(4) << iter << " | "
                      << std::setw(13) << std::fixed << std::setprecision(6) << trans_err << " | "
                      << std::setw(13) << std::setprecision(6) << rot_err << " | "
                      << std::setprecision(2) << total_cost << std::endl;
        }
        
        // Convergence check
        if (delta.head<3>().norm() < config.translation_tolerance &&
            delta.tail<3>().norm() < config.rotation_tolerance) {
            result.converged = true;
            break;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.total_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    result.total_iterations = static_cast<int>(result.history.size());
    
    if (!result.history.empty()) {
        result.final_cost = result.history.back().cost;
        result.final_translation_error = result.history.back().translation_error;
        result.final_rotation_error = result.history.back().rotation_error;
    }
    
    return result;
}

BenchmarkResult NDTICP::runWithRawTarget(
    const PointCloudWithCovariance& source,
    const PointCloudWithCovariance& target_raw,
    const Eigen::Matrix4f& initial_transform,
    const ICPConfig& config,
    const Eigen::Matrix4f& ground_truth)
{
    // NDT uses raw target to build voxel map (not downsampled)
    // This gives NDT the advantage of using all points for distribution estimation
    return run(source, target_raw, initial_transform, config, ground_truth);
}

// ============================================================================
// Utility Functions Implementation
// ============================================================================

void computeLocalCovariances(PointCloudWithCovariance& cloud, int k_neighbors)
{
    const size_t n = cloud.size();
    if (n < static_cast<size_t>(k_neighbors)) {
        return;
    }
    
    for (size_t i = 0; i < n; ++i) {
        // Find k nearest neighbors (brute force)
        std::vector<std::pair<float, size_t>> distances;
        distances.reserve(n);
        
        for (size_t j = 0; j < n; ++j) {
            if (i != j) {
                float dist = (cloud.points[i].point - cloud.points[j].point).squaredNorm();
                distances.emplace_back(dist, j);
            }
        }
        
        std::partial_sort(distances.begin(), 
                          distances.begin() + std::min(static_cast<size_t>(k_neighbors), distances.size()),
                          distances.end());
        
        // Compute mean
        Eigen::Vector3f mean = cloud.points[i].point;
        int count = 1;
        for (int k = 0; k < k_neighbors && k < static_cast<int>(distances.size()); ++k) {
            mean += cloud.points[distances[k].second].point;
            count++;
        }
        mean /= static_cast<float>(count);
        
        // Compute covariance
        Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
        Eigen::Vector3f diff = cloud.points[i].point - mean;
        cov += diff * diff.transpose();
        
        for (int k = 0; k < k_neighbors && k < static_cast<int>(distances.size()); ++k) {
            diff = cloud.points[distances[k].second].point - mean;
            cov += diff * diff.transpose();
        }
        cov /= static_cast<float>(count);
        
        // Eigen decomposition
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(cov);
        Eigen::Vector3f eigenvalues = solver.eigenvalues();
        Eigen::Matrix3f eigenvectors = solver.eigenvectors();
        
        // Sort in descending order (largest first)
        // Eigen returns in ascending order, so reverse
        cloud.points[i].eigenvalues = Eigen::Vector3f(eigenvalues[2], eigenvalues[1], eigenvalues[0]);
        cloud.points[i].eigenvectors.col(0) = eigenvectors.col(2);
        cloud.points[i].eigenvectors.col(1) = eigenvectors.col(1);
        cloud.points[i].eigenvectors.col(2) = eigenvectors.col(0);
        cloud.points[i].covariance = cov;
        cloud.points[i].normal = cloud.points[i].eigenvectors.col(2);  // Smallest eigenvalue direction
    }
}

PointCloudWithCovariance voxelDownsample(const PointCloudWithCovariance& cloud, float voxel_size)
{
    if (voxel_size <= 0) {
        return cloud;
    }
    
    std::unordered_map<int64_t, std::vector<size_t>> voxel_map;
    
    auto posToKey = [voxel_size](const Eigen::Vector3f& pos) -> int64_t {
        int64_t x = static_cast<int64_t>(std::floor(pos.x() / voxel_size));
        int64_t y = static_cast<int64_t>(std::floor(pos.y() / voxel_size));
        int64_t z = static_cast<int64_t>(std::floor(pos.z() / voxel_size));
        // Hash combine
        return x + y * 73856093LL + z * 19349669LL;
    };
    
    for (size_t i = 0; i < cloud.size(); ++i) {
        int64_t key = posToKey(cloud.points[i].point);
        voxel_map[key].push_back(i);
    }
    
    PointCloudWithCovariance result;
    result.reserve(voxel_map.size());
    
    for (const auto& [key, indices] : voxel_map) {
        // Use centroid
        Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
        for (size_t idx : indices) {
            centroid += cloud.points[idx].point;
        }
        centroid /= static_cast<float>(indices.size());
        
        // Find closest point to centroid
        float min_dist = std::numeric_limits<float>::max();
        size_t best_idx = indices[0];
        for (size_t idx : indices) {
            float dist = (cloud.points[idx].point - centroid).squaredNorm();
            if (dist < min_dist) {
                min_dist = dist;
                best_idx = idx;
            }
        }
        
        PointWithCovariance p = cloud.points[best_idx];
        p.point = centroid;  // Use centroid position
        result.push_back(p);
    }
    
    return result;
}

// ============================================================================
// KITTI Data Loading
// ============================================================================

PointCloudWithCovariance loadKITTIPointCloud(const std::string& bin_file_path)
{
    PointCloudWithCovariance cloud;
    
    std::ifstream file(bin_file_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open KITTI bin file: " << bin_file_path << std::endl;
        return cloud;
    }
    
    // Get file size
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // KITTI format: x, y, z, intensity (4 floats per point)
    size_t num_points = file_size / (4 * sizeof(float));
    cloud.reserve(num_points);
    
    std::vector<float> buffer(4);
    for (size_t i = 0; i < num_points; ++i) {
        file.read(reinterpret_cast<char*>(buffer.data()), 4 * sizeof(float));
        
        PointWithCovariance p;
        p.point = Eigen::Vector3f(buffer[0], buffer[1], buffer[2]);
        cloud.push_back(p);
    }
    
    return cloud;
}

std::vector<Eigen::Matrix4f> loadKITTIGroundTruth(const std::string& pose_file_path)
{
    std::vector<Eigen::Matrix4f> poses;
    
    std::ifstream file(pose_file_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open ground truth file: " << pose_file_path << std::endl;
        return poses;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
        
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 4; ++j) {
                iss >> pose(i, j);
            }
        }
        
        poses.push_back(pose);
    }
    
    return poses;
}

std::string getKITTIBinPath(const std::string& dataset_path, int sequence, int frame)
{
    std::ostringstream oss;
    oss << dataset_path << "/" 
        << std::setfill('0') << std::setw(2) << sequence 
        << "/velodyne/"
        << std::setfill('0') << std::setw(6) << frame << ".bin";
    return oss.str();
}

// ============================================================================
// Benchmark Runner
// ============================================================================

KITTIBenchmarkRunner::KITTIBenchmarkRunner(const KITTIBenchmarkConfig& config)
    : config_(config)
{
    initializeAlgorithms();
}

void KITTIBenchmarkRunner::initializeAlgorithms()
{
    algorithms_[ICPMethod::POINT_TO_POINT] = std::make_unique<PointToPointICP>();
    algorithms_[ICPMethod::POINT_TO_PLANE] = std::make_unique<PointToPlaneICP>();
    algorithms_[ICPMethod::SYMMETRIC] = std::make_unique<SymmetricICP>();
    algorithms_[ICPMethod::GICP] = std::make_unique<GeneralizedICP>();
    algorithms_[ICPMethod::MC_ICP] = std::make_unique<ManifoldConstrainedICP>();
    algorithms_[ICPMethod::NDT] = std::make_unique<NDTICP>(0.5f);  // NDT with 0.5m voxel size
}

std::map<float, std::vector<BenchmarkResult>> KITTIBenchmarkRunner::runAll()
{
    // Skip Point-to-Point - only run faster methods
    return run({ICPMethod::POINT_TO_PLANE, 
                ICPMethod::SYMMETRIC, ICPMethod::GICP, ICPMethod::MC_ICP, ICPMethod::NDT});
}

std::map<float, std::vector<BenchmarkResult>> KITTIBenchmarkRunner::run(
    const std::vector<ICPMethod>& methods)
{
    std::map<float, std::vector<BenchmarkResult>> all_results;
    
    // Load point clouds
    // ICP aligns source to target: finds T such that T * p_source ≈ p_target
    // We want to estimate the motion from frame1 to frame2
    // So source = frame2, target = frame1
    // ICP finds T = T_1^{-1} * T_2 (which matches GT)
    std::string bin_path1 = getKITTIBinPath(config_.dataset_path, config_.sequence, config_.frame1);
    std::string bin_path2 = getKITTIBinPath(config_.dataset_path, config_.sequence, config_.frame2);
    
    if (config_.verbose) {
        std::cout << "Loading point clouds..." << std::endl;
        std::cout << "  Source (frame2): " << bin_path2 << std::endl;
        std::cout << "  Target (frame1): " << bin_path1 << std::endl;
    }
    
    // SWAP: source = frame2, target = frame1
    PointCloudWithCovariance source_raw = loadKITTIPointCloud(bin_path2);
    PointCloudWithCovariance target_raw = loadKITTIPointCloud(bin_path1);
    
    if (source_raw.size() == 0 || target_raw.size() == 0) {
        std::cerr << "Failed to load point clouds!" << std::endl;
        return all_results;
    }
    
    if (config_.verbose) {
        std::cout << "  Source points: " << source_raw.size() << std::endl;
        std::cout << "  Target points: " << target_raw.size() << std::endl;
    }
    
    // Load ground truth
    std::vector<Eigen::Matrix4f> gt_poses = loadKITTIGroundTruth(config_.ground_truth_path);
    
    // KITTI poses are in CAMERA coordinate (Z-forward, X-right, Y-down)
    // KITTI velodyne is in VELODYNE coordinate (X-forward, Y-left, Z-up)
    // We need to convert GT from camera to velodyne frame
    // T_vel = T_cam_vel^{-1} * T_cam * T_cam_vel
    // where T_cam_vel transforms velodyne -> camera
    
    // Approximate camera-to-velodyne rotation (ignoring translation for simplicity)
    // Camera: Z-forward, X-right, Y-down
    // Velodyne: X-forward, Y-left, Z-up
    // R_cam_vel rotates velodyne frame to camera frame
    Eigen::Matrix3f R_cam_vel;
    R_cam_vel << 0, -1,  0,   // cam_x = -vel_y
                 0,  0, -1,   // cam_y = -vel_z
                 1,  0,  0;   // cam_z = vel_x
    
    Eigen::Matrix4f T_cam_vel = Eigen::Matrix4f::Identity();
    T_cam_vel.block<3,3>(0,0) = R_cam_vel;
    
    Eigen::Matrix4f T_vel_cam = T_cam_vel.inverse();
    
    // Relative pose from frame1 to frame2 in camera frame
    Eigen::Matrix4f gt_relative_cam = Eigen::Matrix4f::Identity();
    if (config_.frame1 < static_cast<int>(gt_poses.size()) && 
        config_.frame2 < static_cast<int>(gt_poses.size())) {
        gt_relative_cam = gt_poses[config_.frame1].inverse() * gt_poses[config_.frame2];
    }
    
    // Convert to velodyne frame
    Eigen::Matrix4f gt_relative = T_vel_cam * gt_relative_cam * T_cam_vel;
    
    if (config_.verbose) {
        Eigen::Vector3f gt_trans = gt_relative.block<3,1>(0,3);
        std::cout << "  GT translation: [" << gt_trans.x() << ", " << gt_trans.y() << ", " << gt_trans.z() << "]" << std::endl;
        std::cout << "  GT translation norm: " << gt_trans.norm() << " m" << std::endl;
    }
    
    // Initial transform (identity - no prior knowledge)
    Eigen::Matrix4f initial_transform = Eigen::Matrix4f::Identity();
    
    // Run for each voxel size
    for (float voxel_size : config_.voxel_sizes) {
        if (config_.verbose) {
            std::cout << "\n=== Voxel size: " << voxel_size << " ===" << std::endl;
        }
        
        // Downsample
        PointCloudWithCovariance source = voxelDownsample(source_raw, voxel_size);
        PointCloudWithCovariance target = voxelDownsample(target_raw, voxel_size);
        
        if (config_.verbose) {
            std::cout << "  Downsampled source: " << source.size() << std::endl;
            std::cout << "  Downsampled target: " << target.size() << std::endl;
        }
        
        // Compute covariances
        computeLocalCovariances(source, config_.k_neighbors);
        computeLocalCovariances(target, config_.k_neighbors);
        
        // Run each method (pass target_raw for NDT)
        std::vector<BenchmarkResult> results = runForVoxelSize(
            voxel_size, methods, source, target, target_raw, initial_transform, gt_relative);
        
        all_results[voxel_size] = results;
    }
    
    return all_results;
}

std::vector<BenchmarkResult> KITTIBenchmarkRunner::runForVoxelSize(
    float voxel_size,
    const std::vector<ICPMethod>& methods,
    const PointCloudWithCovariance& source,
    const PointCloudWithCovariance& target,
    const PointCloudWithCovariance& target_raw,
    const Eigen::Matrix4f& initial_transform,
    const Eigen::Matrix4f& ground_truth)
{
    std::vector<BenchmarkResult> results(methods.size());
    
    ICPConfig icp_config;
    icp_config.max_iterations = config_.max_iterations;
    icp_config.convergence_threshold = config_.convergence_threshold;
    icp_config.max_correspondence_distance = config_.max_correspondence_distance;
    icp_config.k_neighbors = config_.k_neighbors;
    icp_config.verbose = config_.verbose;
    
    // Run methods in parallel using OpenMP
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < methods.size(); ++i) {
        ICPMethod method = methods[i];
        
        BenchmarkResult result;
        
        if (method == ICPMethod::NDT) {
            // NDT uses raw target for voxel map building (not downsampled)
            // Use 1.0m voxel for larger basin of convergence (frame-to-frame ~0.5m motion)
            NDTICP ndt_algorithm(1.0f);  // NDT with 1.0m voxel size
            result = ndt_algorithm.runWithRawTarget(
                source, target_raw, initial_transform, icp_config, ground_truth);
        } else {
            // Other methods use downsampled target with k-NN covariances
            std::unique_ptr<ICPAlgorithm> algorithm;
            switch (method) {
                case ICPMethod::POINT_TO_POINT:
                    algorithm = std::make_unique<PointToPointICP>();
                    break;
                case ICPMethod::POINT_TO_PLANE:
                    algorithm = std::make_unique<PointToPlaneICP>();
                    break;
                case ICPMethod::SYMMETRIC:
                    algorithm = std::make_unique<SymmetricICP>();
                    break;
                case ICPMethod::GICP:
                    algorithm = std::make_unique<GeneralizedICP>();
                    break;
                case ICPMethod::MC_ICP:
                    algorithm = std::make_unique<ManifoldConstrainedICP>();
                    break;
                default:
                    break;
            }
            if (algorithm) {
                result = algorithm->run(
                    source, target, initial_transform, icp_config, ground_truth);
            }
        }
        result.voxel_size = voxel_size;
        
        results[i] = result;
    }
    
    return results;
}

void KITTIBenchmarkRunner::printResults(
    const std::map<float, std::vector<BenchmarkResult>>& results)
{
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "ICP BENCHMARK RESULTS" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    for (const auto& [voxel_size, voxel_results] : results) {
        std::cout << "\nVoxel Size: " << voxel_size << "m" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        std::cout << std::left << std::setw(15) << "Method"
                  << std::right << std::setw(10) << "Iters"
                  << std::setw(12) << "Trans(m)"
                  << std::setw(12) << "Rot(deg)"
                  << std::setw(12) << "Time(ms)"
                  << std::setw(10) << "Conv" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        
        for (const auto& r : voxel_results) {
            std::cout << std::left << std::setw(15) << r.method_name
                      << std::right << std::setw(10) << r.total_iterations
                      << std::setw(12) << std::fixed << std::setprecision(4) << r.final_translation_error
                      << std::setw(12) << r.final_rotation_error
                      << std::setw(12) << std::setprecision(2) << r.total_time_ms
                      << std::setw(10) << (r.converged ? "Yes" : "No") << std::endl;
        }
    }
    std::cout << std::string(80, '=') << std::endl;
}

void KITTIBenchmarkRunner::exportToCSV(
    const std::map<float, std::vector<BenchmarkResult>>& results,
    const std::string& filename)
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open CSV file: " << filename << std::endl;
        return;
    }
    
    // Header
    file << "voxel_size,method,iterations,trans_error,rot_error,time_ms,converged,"
         << "source_points,target_points,init_trans_error,init_rot_error\n";
    
    for (const auto& [voxel_size, voxel_results] : results) {
        for (const auto& r : voxel_results) {
            file << voxel_size << ","
                 << r.method_name << ","
                 << r.total_iterations << ","
                 << r.final_translation_error << ","
                 << r.final_rotation_error << ","
                 << r.total_time_ms << ","
                 << (r.converged ? 1 : 0) << ","
                 << r.source_points << ","
                 << r.target_points << ","
                 << r.initial_translation_error << ","
                 << r.initial_rotation_error << "\n";
        }
    }
    
    std::cout << "Results exported to: " << filename << std::endl;
}

void KITTIBenchmarkRunner::exportConvergenceCurves(
    const std::map<float, std::vector<BenchmarkResult>>& results,
    const std::string& filename)
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open CSV file: " << filename << std::endl;
        return;
    }
    
    // Header
    file << "voxel_size,method,iteration,cost,trans_error,rot_error,step_size,time_ms,num_correspondences\n";
    
    for (const auto& [voxel_size, voxel_results] : results) {
        for (const auto& r : voxel_results) {
            for (const auto& s : r.history) {
                file << voxel_size << ","
                     << r.method_name << ","
                     << s.iteration << ","
                     << s.cost << ","
                     << s.translation_error << ","
                     << s.rotation_error << ","
                     << s.step_size << ","
                     << s.time_ms << ","
                     << s.num_correspondences << "\n";
            }
        }
    }
    
    std::cout << "Convergence curves exported to: " << filename << std::endl;
}

} // namespace benchmark
} // namespace lidar_slam
