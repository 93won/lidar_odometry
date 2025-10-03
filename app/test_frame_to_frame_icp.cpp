/**
 * @file      test_frame_to_frame_icp.cpp
 * @brief     Test program for Frame-to-Frame ICP using two KITTI point clouds
 * @author    Seungwon Choi
 * @date      2025-10-03
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include <iostream>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <Eigen/Dense>
#include <sophus/se3.hpp>

#include "../src/processing/IterativeClosestPoint.h"
#include "../src/processing/FeatureExtractor.h"
#include "../src/optimization/AdaptiveMEstimator.h"
#include "../src/database/LidarFrame.h"
#include "../src/util/Config.h"

using PointType = pcl::PointXYZ;
using PointCloud = pcl::PointCloud<PointType>;
using PointCloudPtr = PointCloud::Ptr;  // PCL compatible boost::shared_ptr

/**
 * @brief Load KITTI binary point cloud file
 * @param filename Path to .bin file
 * @return Loaded point cloud
 */
PointCloudPtr load_kitti_bin(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        spdlog::error("Cannot open file: {}", filename);
        return nullptr;
    }
    
    PointCloudPtr cloud(new PointCloud());
    
    float point[4]; // x, y, z, intensity
    while (file.read(reinterpret_cast<char*>(point), sizeof(point))) {
        PointType pt;
        pt.x = point[0];
        pt.y = point[1];
        pt.z = point[2];
        // Ignore intensity for now
        cloud->points.push_back(pt);
    }
    
    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = false;
    
    file.close();
    
    spdlog::info("Loaded {} points from {}", cloud->size(), filename);
    return cloud;
}

/**
 * @brief Preprocess point cloud (crop, downsample)
 * @param input_cloud Input point cloud
 * @param voxel_size Voxel grid size for downsampling
 * @param max_range Maximum range for cropping
 * @return Preprocessed point cloud
 */
PointCloudPtr preprocess_cloud(PointCloudPtr input_cloud, float voxel_size = 0.4f, float max_range = 50.0f) {
    if (!input_cloud || input_cloud->empty()) {
        return nullptr;
    }
    
    // 1. Crop box filter to remove distant points
    pcl::CropBox<PointType> crop_filter;
    crop_filter.setInputCloud(input_cloud);
    crop_filter.setMin(Eigen::Vector4f(-max_range, -max_range, -3.0, 1.0));
    crop_filter.setMax(Eigen::Vector4f(max_range, max_range, 3.0, 1.0));
    
    PointCloudPtr cropped_cloud(new PointCloud());
    crop_filter.filter(*cropped_cloud);
    
    spdlog::debug("After cropping: {} -> {} points", input_cloud->size(), cropped_cloud->size());
    
    // 2. Voxel grid downsampling
    pcl::VoxelGrid<PointType> voxel_filter;
    voxel_filter.setInputCloud(cropped_cloud);
    voxel_filter.setLeafSize(voxel_size, voxel_size, voxel_size);
    
    PointCloudPtr processed_cloud(new PointCloud());
    voxel_filter.filter(*processed_cloud);
    
    spdlog::info("After preprocessing: {} -> {} points (voxel_size: {:.2f})", 
                 input_cloud->size(), processed_cloud->size(), voxel_size);
    
    return processed_cloud;
}

/**
 * @brief Extract features from point cloud
 * @param cloud Input point cloud
 * @param extractor Feature extractor
 * @return LiDAR frame with extracted features
 */
std::shared_ptr<lidar_odometry::database::LidarFrame> extract_features(
    PointCloudPtr cloud, 
    std::shared_ptr<lidar_odometry::processing::FeatureExtractor> extractor) {
    
    if (!cloud || cloud->empty()) {
        return nullptr;
    }
    
    // Create LidarFrame
    auto frame = std::make_shared<lidar_odometry::database::LidarFrame>();
    frame->set_raw_cloud(cloud);
    frame->set_timestamp(0); // Dummy timestamp
    
    // Extract features
    extractor->extract_features(frame);
    
    spdlog::info("Feature extraction completed: {} points -> {} edge features, {} plane features",
                 cloud->size(), 
                 frame->get_edge_features().size(),
                 frame->get_plane_features().size());
    
    return frame;
}

/**
 * @brief Print ICP iteration progress
 * @param icp ICP instance
 * @param iteration Current iteration
 */
void print_icp_progress(const lidar_odometry::processing::IterativeClosestPoint& icp, int iteration) {
    const auto& stats = icp.get_statistics();
    const auto& distances = icp.get_correspondence_distances();
    
    double mean_distance = 0.0;
    if (!distances.empty()) {
        mean_distance = std::accumulate(distances.begin(), distances.end(), 0.0) / distances.size();
    }
    
    spdlog::info("Iteration {}: cost={:.6f}, correspondences={}, inliers={}, match_ratio={:.3f}, mean_dist={:.4f}",
                 iteration, stats.final_cost, stats.correspondences_count, 
                 stats.inlier_count, stats.match_ratio, mean_distance);
}

int main(int argc, char** argv) {
    // Initialize logging
    auto console = spdlog::stdout_color_mt("console");
    spdlog::set_default_logger(console);
    spdlog::set_level(spdlog::level::info);
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    
    // Parse command line arguments
    if (argc != 4) {
        spdlog::error("Usage: {} <config.yaml> <frame1_id> <frame2_id>", argv[0]);
        spdlog::info("Example: {} ../config/kitti.yaml 0 10", argv[0]);
        spdlog::info("  This will load 000000.bin and 000010.bin from the configured data directory");
        return -1;
    }
    
    std::string config_path = argv[1];
    int frame1_id = std::stoi(argv[2]);
    int frame2_id = std::stoi(argv[3]);
    
    spdlog::info("═══════════════════════════════════════════════════════════════════");
    spdlog::info("              Frame-to-Frame ICP Test Program                       ");
    spdlog::info("═══════════════════════════════════════════════════════════════════");
    spdlog::info("Configuration file: {}", config_path);
    spdlog::info("Frame 1 ID: {:06d}", frame1_id);
    spdlog::info("Frame 2 ID: {:06d}", frame2_id);
    spdlog::info("");
    
    try {
        // Load configuration
        spdlog::info("Loading configuration from: {}", config_path);
        auto config = std::make_shared<lidar_odometry::util::SystemConfig>();
        if (!config->load_from_yaml(config_path)) {
            spdlog::error("Failed to load configuration from: {}", config_path);
            return -1;
        }
        
        // Construct file paths
        std::string data_dir = config->get_data_directory();
        std::string seq = config->get_sequence();
        
        std::string file1_path = data_dir + "/sequences/" + seq + "/velodyne/" + 
                                std::to_string(frame1_id).insert(0, 6 - std::to_string(frame1_id).length(), '0') + ".bin";
        std::string file2_path = data_dir + "/sequences/" + seq + "/velodyne/" + 
                                std::to_string(frame2_id).insert(0, 6 - std::to_string(frame2_id).length(), '0') + ".bin";
        
        spdlog::info("Input files:");
        spdlog::info("  Frame 1: {}", file1_path);
        spdlog::info("  Frame 2: {}", file2_path);
        spdlog::info("");
    
    try {
        // Step 1: Load point clouds
        spdlog::info("Step 1: Loading point clouds...");
        auto cloud1 = load_kitti_bin(file1_path);
        auto cloud2 = load_kitti_bin(file2_path);
        
        if (!cloud1 || !cloud2) {
            spdlog::error("Failed to load point clouds");
            return -1;
        }
        
        // Step 2: Preprocess point clouds (using config values)
        spdlog::info("\nStep 2: Preprocessing point clouds...");
        float voxel_size = static_cast<float>(config->get_point_cloud_config().voxel_size);
        float max_range = static_cast<float>(config->get_point_cloud_config().max_range);
        
        auto processed_cloud1 = preprocess_cloud(cloud1, voxel_size, max_range);
        auto processed_cloud2 = preprocess_cloud(cloud2, voxel_size, max_range);
        
        if (!processed_cloud1 || !processed_cloud2) {
            spdlog::error("Failed to preprocess point clouds");
            return -1;
        }
        
        // Step 3: Feature extraction (using config values)
        spdlog::info("\nStep 3: Extracting features...");
        
        // Get feature extraction config from YAML
        auto feature_config_yaml = config->get_feature_extraction_config();
        
        // Configure feature extractor
        lidar_odometry::processing::FeatureExtractorConfig feature_config;
        feature_config.min_plane_points = feature_config_yaml.min_plane_points;
        feature_config.max_neighbors = feature_config_yaml.max_neighbors;
        feature_config.max_plane_distance = feature_config_yaml.max_plane_distance;
        feature_config.collinearity_threshold = feature_config_yaml.collinearity_threshold;
        feature_config.max_neighbor_distance = feature_config_yaml.max_neighbor_distance;
        feature_config.feature_quality_threshold = feature_config_yaml.feature_quality_threshold;
        
        auto feature_extractor = std::make_shared<lidar_odometry::processing::FeatureExtractor>(feature_config);
        
        auto frame1 = extract_features(processed_cloud1, feature_extractor);
        auto frame2 = extract_features(processed_cloud2, feature_extractor);
        
        if (!frame1 || !frame2) {
            spdlog::error("Failed to extract features");
            return -1;
        }
        
        // Step 4: Configure Frame-to-Frame ICP (using config values)
        spdlog::info("\nStep 4: Configuring Frame-to-Frame ICP...");
        
        // Get configs from YAML
        auto odometry_config = config->get_odometry_config();
        auto robust_config = config->get_robust_estimation_config();
        auto estimator_config = config->get_estimator_config();
        
        // ICP configuration
        lidar_odometry::processing::ICPConfig icp_config;
        icp_config.max_iterations = odometry_config.max_iterations;
        icp_config.translation_tolerance = odometry_config.translation_threshold;
        icp_config.rotation_tolerance = odometry_config.rotation_threshold;
        icp_config.max_correspondence_distance = odometry_config.max_correspondence_distance;
        icp_config.min_correspondence_points = estimator_config.min_correspondence_points;
        icp_config.outlier_rejection_ratio = 0.9;  // Default value
        icp_config.use_robust_loss = robust_config.use_adaptive_m_estimator;
        icp_config.robust_loss_delta = 0.1;  // Default value
        icp_config.max_kdtree_neighbors = feature_config_yaml.max_neighbors * 4;  // More neighbors for ICP
        
        // AdaptiveMEstimator configuration
        lidar_odometry::optimization::AdaptiveMEstimatorConfig adaptive_config;
        adaptive_config.use_adaptive_m_estimator = robust_config.use_adaptive_m_estimator;
        adaptive_config.loss_type = robust_config.pko_kernel_type;
        adaptive_config.scale_method = "PKO";
        adaptive_config.min_scale_factor = robust_config.min_scale_factor;
        adaptive_config.max_scale_factor = robust_config.max_scale_factor;
        adaptive_config.num_alpha_segments = robust_config.num_alpha_segments;
        adaptive_config.truncated_threshold = robust_config.truncated_threshold;
        adaptive_config.gmm_components = robust_config.gmm_components;
        adaptive_config.gmm_sample_size = robust_config.gmm_sample_size;
        adaptive_config.pko_kernel_type = robust_config.pko_kernel_type;
        
        spdlog::info("Configuration loaded successfully:");
        spdlog::info("  Voxel size: {:.2f}", voxel_size);
        spdlog::info("  Max range: {:.1f}", max_range);
        spdlog::info("  ICP max iterations: {}", icp_config.max_iterations);
        spdlog::info("  ICP translation tolerance: {:.2e}", icp_config.translation_tolerance);
        spdlog::info("  ICP rotation tolerance: {:.2e}", icp_config.rotation_tolerance);
        spdlog::info("  Max correspondence distance: {:.2f}", icp_config.max_correspondence_distance);
        spdlog::info("  Min correspondence points: {}", icp_config.min_correspondence_points);
        spdlog::info("  PKO kernel type: {}", adaptive_config.pko_kernel_type);
        spdlog::info("  PKO alpha range: [{:.3f}, {:.1f}]", adaptive_config.min_scale_factor, adaptive_config.max_scale_factor);
        
        auto adaptive_estimator = std::make_shared<lidar_odometry::optimization::AdaptiveMEstimator>(
            adaptive_config.use_adaptive_m_estimator,
            adaptive_config.loss_type,
            adaptive_config.min_scale_factor,
            adaptive_config.max_scale_factor,
            adaptive_config.num_alpha_segments,
            adaptive_config.truncated_threshold,
            adaptive_config.gmm_components,
            adaptive_config.gmm_sample_size,
            adaptive_config.pko_kernel_type
        );
        
        // Create ICP instance
        auto icp = std::make_shared<lidar_odometry::processing::IterativeClosestPoint>(icp_config, adaptive_estimator);
        
        // Step 5: Run Frame-to-Frame ICP
        spdlog::info("\nStep 5: Running Frame-to-Frame ICP...");
        
        // Initial poses (identity for both frames)
        lidar_odometry::processing::ICPPose initial_T_w_l1 = lidar_odometry::processing::ICPPose();  // Identity
        lidar_odometry::processing::ICPPose initial_T_w_l2 = lidar_odometry::processing::ICPPose();  // Identity
        
        // Add some initial offset to T_w_l2 to make it more challenging
        Eigen::Vector3f initial_translation(1.0f, 0.5f, 0.0f);  // 1m forward, 0.5m right
        Eigen::AngleAxisf initial_rotation(0.1f, Eigen::Vector3f::UnitZ());  // 0.1 rad around Z
        initial_T_w_l2 = lidar_odometry::processing::ICPPose(initial_rotation.toRotationMatrix(), initial_translation);
        
        spdlog::info("Initial T_w_l1 (fixed): identity");
        spdlog::info("Initial T_w_l2 (to be optimized): translation=[{:.3f}, {:.3f}, {:.3f}], rotation={:.3f} rad",
                     initial_translation.x(), initial_translation.y(), initial_translation.z(), 0.1f);
        
        lidar_odometry::processing::ICPPose result_T_w_l1, result_T_w_l2;
        
        spdlog::info("\n--- Starting Frame-to-Frame ICP Optimization ---");
        
        bool success = icp->align_frames(frame1, frame2, 
                                        initial_T_w_l1, initial_T_w_l2,
                                        result_T_w_l1, result_T_w_l2);
        
        // Step 6: Print results
        spdlog::info("\n═══════════════════════════════════════════════════════════════════");
        spdlog::info("                           RESULTS                                 ");
        spdlog::info("═══════════════════════════════════════════════════════════════════");
        
        const auto& stats = icp->get_statistics();
        
        spdlog::info("ICP Convergence: {}", success ? "SUCCESS" : "FAILED");
        spdlog::info("Iterations used: {}", stats.iterations_used);
        spdlog::info("Initial cost: {:.8f}", stats.initial_cost);
        spdlog::info("Final cost: {:.8f}", stats.final_cost);
        spdlog::info("Cost reduction: {:.2f}%", 
                     stats.initial_cost > 0 ? (1.0 - stats.final_cost / stats.initial_cost) * 100.0 : 0.0);
        spdlog::info("Final correspondences: {}", stats.correspondences_count);
        spdlog::info("Final inliers: {}", stats.inlier_count);
        spdlog::info("Match ratio: {:.3f}", stats.match_ratio);
        
        // Print final poses
        Eigen::Vector3f final_translation = result_T_w_l2.translation();
        Eigen::AngleAxisf final_rotation(result_T_w_l2.rotationMatrix());
        
        spdlog::info("\nFinal T_w_l1 (fixed): identity");
        spdlog::info("Final T_w_l2 (optimized): translation=[{:.6f}, {:.6f}, {:.6f}], rotation={:.6f} rad",
                     final_translation.x(), final_translation.y(), final_translation.z(), final_rotation.angle());
        
        // Calculate pose change
        auto pose_change = initial_T_w_l2.inverse() * result_T_w_l2;
        Eigen::Vector3f translation_change = pose_change.translation();
        Eigen::AngleAxisf rotation_change(pose_change.rotationMatrix());
        
        spdlog::info("\nPose change from initial guess:");
        spdlog::info("  Translation change: [{:.6f}, {:.6f}, {:.6f}] (norm: {:.6f} m)",
                     translation_change.x(), translation_change.y(), translation_change.z(),
                     translation_change.norm());
        spdlog::info("  Rotation change: {:.6f} rad ({:.3f} deg)",
                     rotation_change.angle(), rotation_change.angle() * 180.0f / M_PI);
        
        // Print correspondence distance statistics
        const auto& distances = icp->get_correspondence_distances();
        if (!distances.empty()) {
            double mean_dist = std::accumulate(distances.begin(), distances.end(), 0.0) / distances.size();
            double min_dist = *std::min_element(distances.begin(), distances.end());
            double max_dist = *std::max_element(distances.begin(), distances.end());
            
            spdlog::info("\nCorrespondence distance statistics:");
            spdlog::info("  Mean: {:.6f} m", mean_dist);
            spdlog::info("  Min:  {:.6f} m", min_dist);
            spdlog::info("  Max:  {:.6f} m", max_dist);
        }
        
        spdlog::info("\n═══════════════════════════════════════════════════════════════════");
        spdlog::info("                    Program completed successfully                  ");
        spdlog::info("═══════════════════════════════════════════════════════════════════");
        
        return success ? 0 : 1;
        
    } catch (const std::exception& e) {
        spdlog::error("Exception caught: {}", e.what());
        return -1;
    }
}