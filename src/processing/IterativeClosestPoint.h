/**
 * @file      IterativeClosestPoint.h
 * @brief     Modern ICP implementation using Ceres optimization.
 * @author    Seungwon Choi
 * @date      2025-09-24
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#pragma once

#include <memory>
#include <vector>
#include "../util/Types.h"
#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <ceres/ceres.h>
#include "../optimization/AdaptiveMEstimator.h"

namespace lidar_odometry {
namespace processing {

// Type aliases for ICP - using our point cloud implementation
using ICPPointType = util::Point3D;
using ICPPointCloud = util::PointCloud;
using ICPPointCloudPtr = std::shared_ptr<ICPPointCloud>;
using ICPPointCloudConstPtr = std::shared_ptr<const ICPPointCloud>;
using ICPPose = Sophus::SE3f;
using ICPVector3f = Eigen::Vector3f;

/**
 * @brief Configuration for ICP algorithm
 */
struct ICPConfig {
    // Convergence criteria
    int max_iterations = 50;
    double translation_tolerance = 1e-6;  // meters
    double rotation_tolerance = 1e-6;     // radians
    
    // Correspondence parameters
    double max_correspondence_distance = 1.0;  // meters
    int min_correspondence_points = 10;
    
    // Outlier rejection
    double outlier_rejection_ratio = 0.9;  // Keep top 90% of correspondences
    bool use_robust_loss = true;
    double robust_loss_delta = 0.1;  // Huber loss delta
    
    // Performance
    bool use_kdtree = true;
    int max_kdtree_neighbors = 1;
};

/**
 * @brief Point-to-plane correspondence for ICP
 */
struct ICPPointCorrespondence {
    ICPVector3f source_point;
    ICPVector3f target_point;
    ICPVector3f plane_normal;    // Normal vector of target plane
    double distance = 0.0;
    double weight = 1.0;
    bool is_valid = false;
    
    ICPPointCorrespondence() = default;
    ICPPointCorrespondence(const ICPVector3f& src, const ICPVector3f& tgt, double dist)
        : source_point(src), target_point(tgt), distance(dist), weight(1.0), is_valid(true) {}
    ICPPointCorrespondence(const ICPVector3f& src, const ICPVector3f& tgt, const ICPVector3f& normal, double dist)
        : source_point(src), target_point(tgt), plane_normal(normal), distance(dist), weight(1.0), is_valid(true) {}
};

using ICPCorrespondenceVector = std::vector<ICPPointCorrespondence>;

/**
 * @brief ICP statistics for monitoring convergence
 */
struct ICPStatistics {
    int iterations_used = 0;
    double final_cost = 0.0;
    double initial_cost = 0.0;
    size_t correspondences_count = 0;
    size_t inlier_count = 0;
    double match_ratio = 0.0;
    bool converged = false;
    
    void reset() {
        iterations_used = 0;
        final_cost = 0.0;
        initial_cost = 0.0;
        correspondences_count = 0;
        inlier_count = 0;
        match_ratio = 0.0;
        converged = false;
    }
};

/**
 * @brief Modern ICP implementation using Ceres optimization
 * 
 * This class provides a robust ICP implementation that uses Ceres solver
 * for pose optimization with proper outlier handling and convergence criteria.
 */
class IterativeClosestPoint {
public:
    /**
     * @brief Constructor
     * @param config ICP configuration
     */
    explicit IterativeClosestPoint(const ICPConfig& config);
    
    /**
     * @brief Constructor with AdaptiveMEstimator
     * @param config ICP configuration
     * @param adaptive_estimator Pointer to AdaptiveMEstimator for robust optimization
     */
    IterativeClosestPoint(const ICPConfig& config, 
                         std::shared_ptr<optimization::AdaptiveMEstimator> adaptive_estimator);
    
    /**
     * @brief Destructor
     */
    ~IterativeClosestPoint() = default;
    
    /**
     * @brief Align source cloud to target cloud using ICP
     * @param source_cloud Source point cloud to be aligned
     * @param target_cloud Target point cloud (reference)
     * @param initial_guess Initial pose estimate
     * @param result_pose Output aligned pose
     * @return True if ICP converged successfully
     */
    bool align(ICPPointCloudConstPtr source_cloud,
               ICPPointCloudConstPtr target_cloud,
               const ICPPose& initial_guess,
               ICPPose& result_pose);
    
    /**
     * @brief Get ICP statistics from last alignment
     * @return ICP statistics
     */
    const ICPStatistics& get_statistics() const { return m_statistics; }
    
    /**
     * @brief Get correspondence distances from last alignment
     * @return Vector of correspondence distances (residuals)
     */
    std::vector<double> get_correspondence_distances() const { return m_last_correspondence_distances; }
    
    /**
     * @brief Update ICP configuration
     * @param config New configuration
     */
    void update_config(const ICPConfig& config);
    
    /**
     * @brief Get current configuration
     * @return Current ICP configuration
     */
    const ICPConfig& get_config() const { return m_config; }
    
private:
    /**
     * @brief Find point correspondences between source and target clouds
     * @param source_cloud Source point cloud
     * @param target_cloud Target point cloud
     * @param current_pose Current pose estimate
     * @param correspondences Output correspondences
     * @return Number of valid correspondences found
     */
    size_t find_correspondences(ICPPointCloudConstPtr source_cloud,
                               ICPPointCloudConstPtr target_cloud,
                               const ICPPose& current_pose,
                               ICPCorrespondenceVector& correspondences);
    
    /**
     * @brief Reject outlier correspondences
     * @param correspondences Input/output correspondences
     * @return Number of inlier correspondences
     */
    size_t reject_outliers(ICPCorrespondenceVector& correspondences);
    
    /**
     * @brief Optimize pose using Ceres solver
     * @param correspondences Point correspondences
     * @param initial_pose Initial pose estimate
     * @param optimized_pose Output optimized pose
     * @return True if optimization succeeded
     */
    bool optimize_pose(const ICPCorrespondenceVector& correspondences,
                      const ICPPose& initial_pose,
                      ICPPose& optimized_pose,
                      double normalization_scale = 1.0);
    
    /**
     * @brief Check convergence based on pose change
     * @param prev_pose Previous pose estimate
     * @param current_pose Current pose estimate
     * @return True if converged
     */
    bool check_convergence(const ICPPose& prev_pose,
                          const ICPPose& current_pose);
    
    /**
     * @brief Calculate alignment cost for statistics
     * @param correspondences Point correspondences
     * @param pose Current pose
     * @return Total alignment cost
     */
    double calculate_cost(const ICPCorrespondenceVector& correspondences,
                         const ICPPose& pose);
    
    /**
     * @brief Check if three points are collinear
     * @param p1 First point
     * @param p2 Second point
     * @param p3 Third point
     * @param threshold Collinearity threshold
     * @return True if points are collinear
     */
    bool isCollinear(const ICPVector3f& p1, const ICPVector3f& p2, const ICPVector3f& p3, float threshold);
    
    // Configuration
    ICPConfig m_config;
    
    // Target cloud KD-tree for efficient nearest neighbor search
    std::unique_ptr<util::KdTree> m_target_kdtree;
    
    // AdaptiveMEstimator for robust optimization
    std::shared_ptr<optimization::AdaptiveMEstimator> m_adaptive_estimator;
    
    // Statistics and correspondence data
    ICPStatistics m_statistics;
    std::vector<double> m_last_correspondence_distances;
};

} // namespace processing
} // namespace lidar_odometry
