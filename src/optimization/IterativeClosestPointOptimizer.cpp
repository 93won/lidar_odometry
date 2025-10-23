/**
 * @file      IterativeClosestPointOptimizer.cpp
 * @brief     Two-frame ICP optimizer implementation
 * @author    Seungwon Choi
 * @date      2025-10-04
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "IterativeClosestPointOptimizer.h"
#include "../util/MathUtils.h"
#include "../util/PointCloudUtils.h"
#include <spdlog/spdlog.h>
#include <chrono>
#include <numeric>
#include <algorithm>

namespace lidar_odometry {
namespace optimization {

IterativeClosestPointOptimizer::IterativeClosestPointOptimizer(const ICPConfig& config)
    : m_config(config), m_adaptive_estimator(nullptr) {
    
    // Initialize voxel filter
    m_voxel_filter = std::make_unique<util::VoxelGrid>();
    m_voxel_filter->setLeafSize(0.4f);  // Default voxel size
    
    spdlog::info("[IterativeClosestPointOptimizer] Initialized with max_iterations={}, max_correspondence_distance={}", 
                 m_config.max_iterations, m_config.max_correspondence_distance);
}

IterativeClosestPointOptimizer::IterativeClosestPointOptimizer(const ICPConfig& config, 
                                            std::shared_ptr<optimization::AdaptiveMEstimator> adaptive_estimator)
    : m_config(config), m_adaptive_estimator(adaptive_estimator) {
    
    // Initialize voxel filter
    m_voxel_filter = std::make_unique<util::VoxelGrid>();
    m_voxel_filter->setLeafSize(0.4f);  // Default voxel size
    
    spdlog::info("[IterativeClosestPointOptimizer] Initialized with max_iterations={}, max_correspondence_distance={} and AdaptiveMEstimator", 
                 m_config.max_iterations, m_config.max_correspondence_distance);
}

bool IterativeClosestPointOptimizer::optimize_loop(std::shared_ptr<database::LidarFrame> curr_keyframe,
                                          std::shared_ptr<database::LidarFrame> matched_keyframe,
                                          Sophus::SE3f &optimized_relative_transform,
                                          float& inlier_ratio)
{


    // Deep copy keyframes to avoid modifying original poses
    auto curr_keyframe_copy = std::make_shared<database::LidarFrame>(*curr_keyframe);
    auto matched_keyframe_copy = std::make_shared<database::LidarFrame>(*matched_keyframe);

    Sophus::SE3f optimized_curr_pose = curr_keyframe_copy->get_pose();

    // Reset adaptive estimator
    m_adaptive_estimator->reset();

    bool success = false;

    // build kd tree of local map of matched keyframe

    // set temp local map
    auto local_feature_matched = get_frame_cloud(matched_keyframe_copy);     // matched frame feature cloud (local coordinates)
    PointCloudPtr transformed_matched_cloud = std::make_shared<PointCloud>();

    util::transform_point_cloud(local_feature_matched, transformed_matched_cloud, matched_keyframe_copy->get_pose().matrix());
    matched_keyframe_copy->set_local_map(transformed_matched_cloud);
    matched_keyframe_copy->build_local_map_kdtree();

    for(int icp_iter = 0; icp_iter < 100; ++icp_iter)
    {

        curr_keyframe_copy->set_pose(optimized_curr_pose);

        // find correspondances

        DualFrameCorrespondences correspondences;
        size_t num_correspondences = find_correspondences_loop(matched_keyframe_copy, curr_keyframe_copy, correspondences);

        // Let's do icp
        ceres::Problem problem;
        // SE3 parameters: [tx, ty, tz, rx, ry, rz]

        // curr
        Eigen::Vector6d se3_tangent = optimization::SE3GlobalParameterization::se3_to_tangent(
            optimized_curr_pose.cast<double>()
        );
        std::array<double, 6> pose_params;
        std::copy(se3_tangent.data(), se3_tangent.data() + 6, pose_params.data());
        problem.AddParameterBlock(pose_params.data(), 6);
        problem.SetParameterization(pose_params.data(), new optimization::SE3GlobalParameterization());

        // matched
        Eigen::Vector6d matched_se3_tangent = optimization::SE3GlobalParameterization::se3_to_tangent(
            matched_keyframe_copy->get_pose().cast<double>()
        );
        std::array<double, 6> matched_pose_params;
        std::copy(matched_se3_tangent.data(), matched_se3_tangent.data() + 6, matched_pose_params.data());
        problem.AddParameterBlock(matched_pose_params.data(), 6);
        problem.SetParameterization(matched_pose_params.data(), new optimization::SE3GlobalParameterization());

        // matched is fixed
        problem.SetParameterBlockConstant(matched_pose_params.data());

        // calculate residual normalization scale
        double residual_normalization_scale = 1.0;
        if(icp_iter == 0 && !correspondences.residuals.empty())
        {
            std::vector<double> residuals = correspondences.residuals;
            std::sort(residuals.begin(), residuals.end());
            double mean = std::accumulate(residuals.begin(), residuals.end(), 0.0) / residuals.size();
            double variance = 0.0;
            for (double val : residuals) {
                variance += (val - mean) * (val - mean);
            }
            variance /= residuals.size();
            double std_dev = std::sqrt(variance);
            // Calculate residual normalization scale
            residual_normalization_scale = std_dev / 6.0;
        }

        // calculate adaptive delta
        double adpative_delta = m_config.robust_loss_delta;  // Default delta
        if (m_adaptive_estimator && m_adaptive_estimator->get_config().use_adaptive_m_estimator) {
            // Extract residuals for PKO
            std::vector<double> residuals;
            residuals.reserve(correspondences.residuals.size());
            for (double residual : correspondences.residuals) {
                // Normalize residuals using the scale calculated in first iteration
                double normalized_residual = residual / std::max(residual_normalization_scale, 1e-6);
                residuals.push_back(normalized_residual);
            }

            if (!residuals.empty()) {
                // Calculate scale factor using AdaptiveMEstimator on normalized residuals
                double scale_factor = m_adaptive_estimator->calculate_scale_factor(residuals);

                // Use scale factor as Huber loss delta
                adpative_delta = scale_factor;
            }
        }

        // Add residual blocks (point-to-plane factors)
        for (size_t i = 0; i < correspondences.size(); ++i) {
            // Calculate normalization weight for residual
            double normalization_weight = 1.0 / std::max(residual_normalization_scale, 1e-6);   
            auto factor = new optimization::PointToPlaneFactorDualFrame(
                correspondences.points_last[i].cast<double>(),    // p: last frame point (local)
                correspondences.points_curr[i].cast<double>(),    // q: curr frame point (local)
                correspondences.normals_last[i].cast<double>(),   // nq: normal from last frame (world)
                normalization_weight  // Apply normalization through weight
            );

            if (m_config.use_robust_loss) {
                ceres::LossFunction* loss_function = nullptr;
                if (m_adaptive_estimator) {
                    const std::string& loss_type = m_adaptive_estimator->get_config().loss_type;
                    if (loss_type == "cauchy") {
                        loss_function = new ceres::CauchyLoss(adpative_delta);
                    } else if (loss_type == "huber") {
                        loss_function = new ceres::HuberLoss(adpative_delta);
                    } else {
                        loss_function = new ceres::CauchyLoss(adpative_delta);
                    }
                } else {
                    loss_function = new ceres::HuberLoss(adpative_delta);
                }
                problem.AddResidualBlock(factor, loss_function, matched_pose_params.data(), pose_params.data());
            } else {
                problem.AddResidualBlock(factor, nullptr, matched_pose_params.data(), pose_params.data());
            }
        }

        // Configure solver
        ceres::Solver::Options options;
        options.max_num_iterations = 10;
        options.linear_solver_type = ceres::DENSE_QR;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        
        // Update optimized pose
        Eigen::Vector6d optimized_tangent;
        std::copy(pose_params.data(), pose_params.data() + 6, optimized_tangent.data());
        
        
        Sophus::SE3f optimized_curr_pose_new = optimization::SE3GlobalParameterization::tangent_to_se3(optimized_tangent).cast<float>();


        Sophus::SE3f delta_pose = optimized_curr_pose_new.inverse() * optimized_curr_pose;

        float delta_norm_trans = delta_pose.translation().norm();
        float delta_norm_so3 = delta_pose.so3().log().norm();

        optimized_curr_pose = optimized_curr_pose_new;




        if(delta_norm_trans < m_config.translation_tolerance && delta_norm_so3 < m_config.rotation_tolerance)
        {
            // update optimized_relative_transform (from current_old to current_new)

            spdlog::info("[IterativeClosestPointOptimizer] Loop closure ICP converged at iteration {}", icp_iter+1);
            optimized_relative_transform = curr_keyframe->get_pose().inverse() * optimized_curr_pose;
            success = true;
            break;
        }
    }

    
    // Calculate inlier ratio for validation
    if (success) {
        // Update current keyframe pose to optimized pose
        curr_keyframe_copy->set_pose(optimized_curr_pose);
        
        // Inlier ratio using kd-tree of matched keyframe
        int inlier_count = 0;
        int total_count = 0;
        auto curr_feature_cloud = get_frame_cloud(curr_keyframe_copy);
        for (size_t i = 0; i < curr_feature_cloud->size(); ++i) {
            const auto& point_local = curr_feature_cloud->at(i);
            
            // Transform to world coordinates
            Eigen::Vector3f point_local_eigen(point_local.x, point_local.y, point_local.z);
            Eigen::Vector3f point_world_eigen = curr_keyframe_copy->get_pose().matrix().block<3,1>(0,3) + 
                                                curr_keyframe_copy->get_pose().matrix().block<3,3>(0,0) * point_local_eigen;
            
            // Convert to Point3D for KdTree query
            util::Point3D point_world;
            point_world.x = point_world_eigen.x();
            point_world.y = point_world_eigen.y();
            point_world.z = point_world_eigen.z();
            
            // Find nearest neighbor in matched keyframe's kd-tree (returns squared distance)
            std::vector<int> indices(1);
            std::vector<float> sqdist(1);
            matched_keyframe_copy->get_local_map_kdtree()->nearestKSearch(point_world, 1, indices, sqdist);

            // Check if distance is below threshold (0.1m)
            if (std::sqrt(sqdist[0]) < 1.0f) {
                inlier_count++;
            }
            total_count++;
        }

        inlier_ratio = static_cast<float>(inlier_count) / static_cast<float>(total_count);

        spdlog::info("[IterativeClosestPointOptimizer] Loop closure optimization inlier ratio: {:.3f}", inlier_ratio);

        if(inlier_ratio < 0.5f)
        {
            success = false;
        }
            
    } 

    return success;
}



bool IterativeClosestPointOptimizer::optimize(std::shared_ptr<database::LidarFrame> last_keyframe,
                                     std::shared_ptr<database::LidarFrame> curr_frame,
                                     const Sophus::SE3f& initial_transform,
                                     Sophus::SE3f& optimized_transform) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Reset stats
    /*
        struct OptimizationStats {
            size_t num_correspondences = 0;
            size_t num_iterations = 0;
            double initial_cost = 0.0;
            double final_cost = 0.0;
            double optimization_time_ms = 0.0;
            bool converged = false;
        };                                
    */
    m_last_stats = OptimizationStats();//记录信息
    
    // Initialize current transform estimate
    Sophus::SE3f current_transform = initial_transform;
    optimized_transform = current_transform;
    
    double total_initial_cost = 0.0;
    double total_final_cost = 0.0;
    int total_iterations = 0;
    double residual_normalization_scale = 1.0;  // Initialize normalization scale

    // Reset adaptive estimator
    m_adaptive_estimator->reset();

    // ICP iteration loop
    for (int icp_iter = 0; icp_iter < m_config.max_iterations; ++icp_iter) {
        
        // Set current poses for correspondence finding
        // Keep last keyframe at its actual pose, update curr frame with current estimate
        // last_keyframe pose should remain unchanged (it's the reference keyframe)
        curr_frame->set_pose(current_transform); // Curr frame at current estimate
        
        // Find correspondences with current transform estimate
        DualFrameCorrespondences correspondences;
        size_t num_correspondences = find_correspondences(last_keyframe, curr_frame, correspondences);
        
        if (num_correspondences < static_cast<size_t>(m_config.min_correspondence_points)) {
            spdlog::warn("[ICP] Insufficient correspondences: {} < {} at iteration {}", 
                        num_correspondences, m_config.min_correspondence_points, icp_iter + 1);
            return false;
        }
        
        // Analyze residual distribution on first iteration
        if (icp_iter == 0 && !correspondences.residuals.empty()) {
            std::vector<double> residuals = correspondences.residuals;
            std::sort(residuals.begin(), residuals.end());
            
            double mean = std::accumulate(residuals.begin(), residuals.end(), 0.0) / residuals.size();
            double variance = 0.0;
            for (double val : residuals) {
                variance += (val - mean) * (val - mean);
            }
            variance /= residuals.size();
            double std_dev = std::sqrt(variance);
            
            size_t n = residuals.size();
            double median = (n % 2 == 0) ? (residuals[n/2-1] + residuals[n/2]) / 2.0 : residuals[n/2];
            double q25 = residuals[n/4];
            double q75 = residuals[3*n/4];
            double min_val = residuals[0];
            double max_val = residuals[n-1];
            
            // Debug residual distribution (only in debug mode)
            spdlog::debug("[IterativeClosestPointOptimizer] First iteration residual distribution:");
            spdlog::debug("  Count: {}, Mean: {:.4f}, Std: {:.4f}, Median: {:.4f}", 
                        n, mean, std_dev, median);
            spdlog::debug("  Min: {:.4f}, Q25: {:.4f}, Q75: {:.4f}, Max: {:.4f}", 
                        min_val, q25, q75, max_val);
            
            // Calculate residual normalization scale (same as ICP)
            residual_normalization_scale = std_dev / 6.0;
            spdlog::debug("  Normalization scale (std/6): {:.6f}", residual_normalization_scale);
        }
        
        // Setup optimization problem for this iteration
        ceres::Problem problem;
        
        // SE3 parameters: [tx, ty, tz, rx, ry, rz]
        Eigen::Vector6d se3_tangent = optimization::SE3GlobalParameterization::se3_to_tangent(
            current_transform.cast<double>()
        );
        std::array<double, 6> pose_params;
        std::copy(se3_tangent.data(), se3_tangent.data() + 6, pose_params.data());
        
        // Add parameter blocks for two poses
        // Use actual last keyframe pose instead of identity
        Eigen::Vector6d last_se3_tangent = optimization::SE3GlobalParameterization::se3_to_tangent(
            last_keyframe->get_pose().cast<double>()
        );
        std::array<double, 6> source_pose_params;
        std::copy(last_se3_tangent.data(), last_se3_tangent.data() + 6, source_pose_params.data());
        problem.AddParameterBlock(source_pose_params.data(), 6);
        problem.AddParameterBlock(pose_params.data(), 6);
        
        // Set parameterizations
        problem.SetParameterization(source_pose_params.data(), new optimization::SE3GlobalParameterization());
        problem.SetParameterization(pose_params.data(), new optimization::SE3GlobalParameterization());
        
        // Fix source pose to identity
        problem.SetParameterBlockConstant(source_pose_params.data());
        
        // Calculate adaptive Huber loss delta using AdaptiveMEstimator if available
        double huber_delta = m_config.robust_loss_delta;  // Default delta
        
        if (m_adaptive_estimator && m_adaptive_estimator->get_config().use_adaptive_m_estimator) {
            
            // Extract residuals for PKO
            std::vector<double> residuals;
            residuals.reserve(correspondences.residuals.size());
            for (double residual : correspondences.residuals) {
                // Normalize residuals using the scale calculated in first iteration
                double normalized_residual = residual / std::max(residual_normalization_scale, 1e-6);
                residuals.push_back(normalized_residual);
            }
            
            if (!residuals.empty()) {
                // Calculate scale factor using AdaptiveMEstimator on normalized residuals
                double scale_factor = m_adaptive_estimator->calculate_scale_factor(residuals);
                
                // Use scale factor as Huber loss delta
                huber_delta = scale_factor;
            }
        }
        
        // Add residual blocks (point-to-plane factors)
        for (size_t i = 0; i < correspondences.size(); ++i) {
            // Calculate normalization weight for residual
            double normalization_weight = 1.0 / std::max(residual_normalization_scale, 1e-6);
            
            auto factor = new optimization::PointToPlaneFactorDualFrame(
                correspondences.points_last[i].cast<double>(),    // p: last frame point (local)
                correspondences.points_curr[i].cast<double>(),    // q: curr frame point (local)
                correspondences.normals_last[i].cast<double>(),   // nq: normal from last frame (world)
                normalization_weight  // Apply normalization through weight
            );
            
            if (m_config.use_robust_loss) {
                // Create appropriate loss function based on AdaptiveMEstimator loss type
                ceres::LossFunction* loss_function = nullptr;
                
                if (m_adaptive_estimator) {
                    const std::string& loss_type = m_adaptive_estimator->get_config().loss_type;
                    
                    if (loss_type == "cauchy") {
                        loss_function = new ceres::CauchyLoss(huber_delta);
                    } else if (loss_type == "huber") {
                        loss_function = new ceres::HuberLoss(huber_delta);
                    } else {
                        // Default to Cauchy loss
                        loss_function = new ceres::CauchyLoss(huber_delta);
                    }
                    
                    // Debug loss function info only for first residual block to avoid spam
                    if (i == 0) {
                        spdlog::debug("[IterativeClosestPointOptimizer] Iter {}: Using {} loss with delta={:.6f}, norm_weight={:.3f}", 
                                     icp_iter + 1, loss_type, huber_delta, normalization_weight);
                    }
                } else {
                    // Default to Huber loss if no adaptive estimator
                    loss_function = new ceres::HuberLoss(huber_delta);
                }
                
                problem.AddResidualBlock(factor, loss_function, source_pose_params.data(), pose_params.data());
            } else {
                problem.AddResidualBlock(factor, nullptr, source_pose_params.data(), pose_params.data());
            }
        }
        
        // Setup solver options for this iteration
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.minimizer_progress_to_stdout = false;  // Reduce output
        options.max_num_iterations = 10;  // Few iterations per ICP step
        options.function_tolerance = 1e-6;
        options.parameter_tolerance = 1e-8;
        
        // Solve for this iteration
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        
        // Extract updated transform
        Eigen::Vector6d updated_tangent = Eigen::Map<const Eigen::Vector6d>(pose_params.data());
        Sophus::SE3d updated_se3_d = optimization::SE3GlobalParameterization::tangent_to_se3(updated_tangent);
        Sophus::SE3f updated_transform = updated_se3_d.cast<float>();
        
        // Calculate transformation delta
        Sophus::SE3f delta_transform = current_transform.inverse() * updated_transform;
        
        // Extract translation and rotation deltas
        Eigen::Vector3f delta_translation = delta_transform.translation();
        float translation_delta = delta_translation.norm();
        float rotation_delta = delta_transform.so3().log().norm();
        
       
        // Update current transform
        current_transform = updated_transform;
        
        // Accumulate stats
        if (icp_iter == 0) {
            total_initial_cost = summary.initial_cost;
        }
        total_final_cost = summary.final_cost;
        total_iterations += summary.iterations.size();
        m_last_stats.num_correspondences = num_correspondences;
        
        // Check convergence
        bool converged = (translation_delta < m_config.translation_tolerance) && 
                        (rotation_delta < m_config.rotation_tolerance);
        
        if (converged) {
            break;
        }
    }
    
    // Set final results
    optimized_transform = current_transform;
    m_last_stats.num_iterations = total_iterations;
    m_last_stats.initial_cost = total_initial_cost;
    m_last_stats.final_cost = total_final_cost;
    m_last_stats.converged = true;  // If we got here, it workedl
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    m_last_stats.optimization_time_ms = duration.count();
    
    return true;
}
//查找匹配对
size_t IterativeClosestPointOptimizer::find_correspondences_loop(std::shared_ptr<database::LidarFrame> last_keyframe,
                                                        std::shared_ptr<database::LidarFrame> curr_keyframe,
                                                        DualFrameCorrespondences &correspondences)
{
    correspondences.clear();
    PointCloudPtr local_map_last = last_keyframe->get_local_map(); // Local map of last keyframe (world coordinates)

    auto local_feature_curr = get_frame_cloud(curr_keyframe);     // Current frame feature cloud (local coordinates)

    if(!local_feature_curr || !local_map_last || local_feature_curr->empty() || local_map_last->empty()) {
        spdlog::warn("[IterativeClosestPointOptimizer] Empty point clouds - curr_map: {}, matched_map: {}",
                     local_feature_curr ? local_feature_curr->size() : 0,
                     local_map_last ? local_map_last->size() : 0);
        return 0;
    }

    auto kdtree_last_ptr = last_keyframe->get_local_map_kdtree();

    if(!kdtree_last_ptr) {
        spdlog::error("[IterativeClosestPointOptimizer] Last keyframe has no KdTree - this should not happen!");
        return 0;
    }



    const int K = 5;  // Number of neighbors for plane fitting  寻找5个点

    // Find correspondences: query CURR points, find neighbors in LAST cloud

    Eigen::Matrix4f T_wl_last = last_keyframe->get_pose().matrix(); // Last keyframe pose in world coordinates
    Eigen::Matrix4f T_lw_last = T_wl_last.inverse(); // Inverse transform
    Eigen::Matrix4f T_wl_curr = curr_keyframe->get_pose().matrix(); // Current keyframe pose in world coordinates
    Eigen::Matrix4f T_lw_curr = T_wl_curr.inverse(); // Inverse transform

    for (size_t idx = 0; idx < local_feature_curr->size(); ++idx)//遍历当前帧的每个点
    {
        const auto& curr_point_local = local_feature_curr->at(idx);

        // Transform current point to world coordinates
        Eigen::Vector4f curr_point_world_h = T_wl_curr * Eigen::Vector4f(curr_point_local.x, curr_point_local.y, curr_point_local.z, 1.0f);
        Eigen::Vector3f curr_point_world(curr_point_world_h.x(), curr_point_world_h.y(), curr_point_world_h.z());

        // Find K nearest neighbors in LAST cloud
        std::vector<int> neighbor_indices(K);
        std::vector<float> neighbor_distances(K);
        util::Point3D query_point;
        query_point.x = curr_point_world.x();
        query_point.y = curr_point_world.y();
        query_point.z = curr_point_world.z();
        int found_neighbors = kdtree_last_ptr->nearestKSearch(query_point, K, neighbor_indices, neighbor_distances);
        if (found_neighbors < 5) {
            continue;
        }
        // Select points for plane fitting from LAST cloud
        std::vector<Eigen::Vector3d> selected_points_world;
        std::vector<Eigen::Vector3d> selected_points_local;
        bool non_collinear_found = false;
        //寻找对应的上一帧世界系下的点 和 feature 点
        for (int k = 0; k < found_neighbors && selected_points_world.size() < 5; ++k) {
            int neighbor_idx = neighbor_indices[k];

            // Local map points are already in world coordinates
            Eigen::Vector3d pt_world(local_map_last->at(neighbor_idx).x,
                                   local_map_last->at(neighbor_idx).y,
                                   local_map_last->at(neighbor_idx).z);

            // For local coordinates, we use the same world coordinates since local map is in world frame
            Eigen::Vector3d pt_local = T_lw_last.block<3,3>(0,0).cast<double>() * pt_world.cast<double>() + T_lw_last.block<3,1>(0,3).cast<double>();

            selected_points_world.push_back(pt_world);
            selected_points_local.push_back(pt_local);
        }

        // Check for non-collinear points
        if (selected_points_world.size() >= 3) {
            if (is_collinear(selected_points_world[0], selected_points_world[1], selected_points_world[2], 0.5)) {//太近 认为 不可用
                continue;
            }
            non_collinear_found = true;
        }
        if (!non_collinear_found) {
            continue;
        }
        
        // Fit plane to selected points using SVD
        if (selected_points_world.size() < 3) {
            continue;
        }
        
        // Compute centroid
        Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
        for (const auto& pt : selected_points_world) {
            centroid += pt;
        }
        centroid /= selected_points_world.size();
        
        // Build matrix for SVD  对世界系下的点进行分解
        Eigen::MatrixXd A(selected_points_world.size(), 3);
        for (size_t i = 0; i < selected_points_world.size(); ++i) {
            A.row(i) = (selected_points_world[i] - centroid).transpose();
        }
        
        // Compute SVD
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
        Eigen::Vector3d plane_normal = svd.matrixV().col(2);  // Last column is normal 奇异值是按照降序进行排列的
        double plane_d = -plane_normal.dot(centroid);
        // Compute point-to-plane distance (residual)
        Eigen::Vector3d curr_point_world_d(curr_point_world.x(), curr_point_world.y(), curr_point_world.z());
        double distance = plane_normal.dot(curr_point_world_d) + plane_d;
        distance = std::abs(distance);  
        // No distance thresholding for loop closure
        // Store correspondence
        correspondences.points_last.push_back(selected_points_local[0]); // Use one of the local points 局部坐标系下的点 不是local_map
        correspondences.points_curr.push_back(Eigen::Vector3d(curr_point_local.x, curr_point_local.y, curr_point_local.z));
        correspondences.normals_last.push_back(plane_normal);
        correspondences.residuals.push_back(distance);  

    }

    return correspondences.size();
}

size_t IterativeClosestPointOptimizer::find_correspondences(std::shared_ptr<database::LidarFrame> last_keyframe,
                                                  std::shared_ptr<database::LidarFrame> curr_frame,
                                                  DualFrameCorrespondences& correspondences) {
    
    correspondences.clear();
    
    // Get point clouds - use local map for keyframe, feature cloud for current frame
    // auto last_cloud = last_keyframe->get_feature_cloud_global();    // Keyframe local map (already in world coordinate
    auto last_cloud = last_keyframe->get_local_map();    // Keyframe local map (already in world coordinates)
    // if(last_cloud->empty()) {
    //     last_cloud = last_keyframe->get_feature_cloud_global();
    // }

    auto curr_cloud = get_frame_cloud(curr_frame);     // Current frame feature cloud (local coordinates)


    Eigen::Matrix4f T_wl = last_keyframe->get_pose().matrix(); // Keyframe pose in world coordinates
    Eigen::Matrix4f T_lw = T_wl.inverse(); // Inverse transform
    
    if (!last_cloud || !curr_cloud || last_cloud->empty() || curr_cloud->empty()) {
        spdlog::warn("[IterativeClosestPointOptimizer] Empty point clouds - last_map: {}, curr_features: {}", 
                     last_cloud ? last_cloud->size() : 0, 
                     curr_cloud ? curr_cloud->size() : 0);
        return 0;
    }
    
    // Transform current frame cloud to world coordinates using current pose estimate
    auto curr_pose = curr_frame->get_pose();   // Current estimate
    
    PointCloudPtr curr_world(new PointCloud());
    util::transform_point_cloud(curr_cloud, curr_world, curr_pose.matrix());
    
    // Use precomputed KD-tree from keyframe local map
    auto kdtree_ptr = last_keyframe->get_local_map_kdtree();
    if (!kdtree_ptr) {
        spdlog::error("[IterativeClosestPointOptimizer] Keyframe has no KdTree - this should not happen!");
        return 0;
    }
    util::KdTree* kdtree = kdtree_ptr.get();
    
    const int K = 5;  // Number of neighbors for plane fitting
    
    // Find correspondences: query CURR points, find neighbors in LAST cloud
    for (size_t idx = 0; idx < curr_world->size(); ++idx) {
        
        const auto& curr_point_world = curr_world->at(idx);
        
        // Find K nearest neighbors in LAST cloud
        std::vector<int> neighbor_indices(K);
        std::vector<float> neighbor_distances(K);
        
        int found_neighbors = kdtree->nearestKSearch(curr_point_world, K, neighbor_indices, neighbor_distances);
        
        if (found_neighbors < 5) {
            continue;
        }
        
        // Select points for plane fitting from LAST cloud
        std::vector<Eigen::Vector3d> selected_points_world;
        std::vector<Eigen::Vector3d> selected_points_local;
        
        bool non_collinear_found = false;
        
        for (int k = 0; k < found_neighbors && selected_points_world.size() < 5; ++k) {
            int neighbor_idx = neighbor_indices[k];
            
            // Local map points are already in world coordinates
            Eigen::Vector3d pt_world(last_cloud->at(neighbor_idx).x,
                                   last_cloud->at(neighbor_idx).y,
                                   last_cloud->at(neighbor_idx).z);
            
            // For local coordinates, we use the same world coordinates since local map is in world frame
            Eigen::Vector3d pt_local = T_lw.block<3,3>(0,0).cast<double>()*pt_world + T_lw.block<3,1>(0,3).cast<double>();
            
            if (selected_points_world.size() < 2) {
                selected_points_world.push_back(pt_world);
                selected_points_local.push_back(pt_local);
            } else if (!non_collinear_found) {
                if (is_collinear(selected_points_world[0], selected_points_world[1], pt_world, 0.5)) {
                    continue;
                } else {
                    non_collinear_found = true;
                    selected_points_world.push_back(pt_world);
                    selected_points_local.push_back(pt_local);
                }
            } else {
                selected_points_world.push_back(pt_world);
                selected_points_local.push_back(pt_local);
            }
        }
        
        if (selected_points_world.size() < 5) {
            continue;
        }
        
        // Fit plane to selected points
        size_t n_points = selected_points_world.size();
        Eigen::MatrixXd A(n_points, 3);
        Eigen::VectorXd b(n_points);
        
        for (size_t p = 0; p < n_points; ++p) {
            A(p, 0) = selected_points_world[p].x();
            A(p, 1) = selected_points_world[p].y();
            A(p, 2) = selected_points_world[p].z();
            b(p) = -1.0;
        }
        
        Eigen::Vector3d normal = A.colPivHouseholderQr().solve(b).normalized();
        Eigen::Vector3d plane_point = selected_points_world[0];
        
        // Validate plane consistency
        bool plane_valid = true;
        for (size_t p = 0; p < n_points; ++p) {
            double dist_to_plane = std::abs(normal.dot(selected_points_world[p] - plane_point));
            if (dist_to_plane > 0.4) {  // voxel_size threshold
                plane_valid = false;
                break;
            }
        }
        
        if (!plane_valid) {
            continue;
        }
        
        // Calculate residual
        Eigen::Vector3d curr_point_world_d(curr_point_world.x, curr_point_world.y, curr_point_world.z);
        double residual = std::abs(normal.dot(curr_point_world_d - plane_point));
        
        if (residual > m_config.max_correspondence_distance) {
            continue;
        }
        
        // Store correspondence
        Eigen::Vector3d last_point_local = selected_points_local[0];  // Use first selected point from LAST cloud
        Eigen::Vector3d curr_point_local(curr_cloud->at(idx).x, curr_cloud->at(idx).y, curr_cloud->at(idx).z);
        
        correspondences.points_last.push_back(last_point_local);
        correspondences.points_curr.push_back(curr_point_local);
        correspondences.normals_last.push_back(normal);  // Normal from LAST cloud (world coordinates)
        correspondences.residuals.push_back(residual);
    }
    
    return correspondences.size();
}

PointCloudConstPtr IterativeClosestPointOptimizer::get_frame_cloud(std::shared_ptr<database::LidarFrame> frame) {
    
    // Try to get feature cloud first, fall back to processed cloud
    auto feature_cloud = frame->get_feature_cloud();
    if (feature_cloud && !feature_cloud->empty()) {
        return feature_cloud;
    }
    
    auto processed_cloud = frame->get_processed_cloud();
    if (processed_cloud && !processed_cloud->empty()) {
        return processed_cloud;
    }
    
    // Last resort: use raw cloud with downsampling
    auto raw_cloud = frame->get_raw_cloud();
    if (raw_cloud && !raw_cloud->empty()) {
        auto downsampled = std::make_shared<PointCloud>();
        m_voxel_filter->setInputCloud(raw_cloud);
        m_voxel_filter->filter(*downsampled);
        return downsampled;
    }
    
    return nullptr;
}

bool IterativeClosestPointOptimizer::is_collinear(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2, 
                                         const Eigen::Vector3d& p3, double threshold) {
    Eigen::Vector3d v1 = (p2 - p1).normalized();
    Eigen::Vector3d v2 = (p3 - p1).normalized();
    
    double cross_norm = v1.cross(v2).norm();
    return cross_norm < threshold;
}

} // namespace processing
} // namespace lidar_odometry