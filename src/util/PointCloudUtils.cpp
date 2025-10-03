/**
 * @file      PointCloudUtils.cpp
 * @brief     Native C++ point cloud utilities implementation
 * @author    Seungwon Choi
 * @date      2025-10-03
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "PointCloudUtils.h"
#include <spdlog/spdlog.h>
#include <filesystem>

namespace lidar_odometry {
namespace util {

PointCloud::Ptr load_kitti_binary(const std::string& filename) {
    auto cloud = std::make_shared<PointCloud>();
    
    // Check if file exists
    if (!std::filesystem::exists(filename)) {
        spdlog::error("KITTI binary file does not exist: {}", filename);
        return cloud;
    }
    
    // Open binary file
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        spdlog::error("Failed to open KITTI binary file: {}", filename);
        return cloud;
    }
    
    // Get file size
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // KITTI format: each point is 4 floats (x, y, z, intensity)
    // We only need x, y, z
    size_t num_points = file_size / (4 * sizeof(float));
    cloud->reserve(num_points);
    
    spdlog::debug("Loading KITTI binary file: {} ({} points)", filename, num_points);
    
    // Read points
    std::vector<float> buffer(4); // x, y, z, intensity
    for (size_t i = 0; i < num_points; ++i) {
        file.read(reinterpret_cast<char*>(buffer.data()), 4 * sizeof(float));
        
        if (file.gcount() != 4 * sizeof(float)) {
            spdlog::warn("Incomplete read at point {} in file {}", i, filename);
            break;
        }
        
        // Add point (x, y, z) - ignore intensity
        cloud->push_back(buffer[0], buffer[1], buffer[2]);
    }
    
    file.close();
    spdlog::debug("Successfully loaded {} points from {}", cloud->size(), filename);
    
    return cloud;
}

bool save_kitti_binary(const PointCloud::ConstPtr& cloud, const std::string& filename) {
    if (!cloud || cloud->empty()) {
        spdlog::error("Cannot save empty point cloud to {}", filename);
        return false;
    }
    
    // Create directory if it doesn't exist
    std::filesystem::path file_path(filename);
    std::filesystem::create_directories(file_path.parent_path());
    
    // Open binary file for writing
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        spdlog::error("Failed to open file for writing: {}", filename);
        return false;
    }
    
    // Write points in KITTI format (x, y, z, intensity)
    std::vector<float> buffer(4);
    for (size_t i = 0; i < cloud->size(); ++i) {
        const auto& point = (*cloud)[i];
        buffer[0] = point.x;
        buffer[1] = point.y;
        buffer[2] = point.z;
        buffer[3] = 0.0f; // Set intensity to 0 as we don't use it
        
        file.write(reinterpret_cast<const char*>(buffer.data()), 4 * sizeof(float));
    }
    
    file.close();
    spdlog::debug("Successfully saved {} points to {}", cloud->size(), filename);
    
    return true;
}

void transform_point_cloud(const PointCloud::ConstPtr& input,
                          PointCloud::Ptr& output,
                          const Eigen::Matrix4f& transformation) {
    if (!input) {
        spdlog::error("Input point cloud is null");
        return;
    }
    
    if (!output) {
        output = std::make_shared<PointCloud>();
    }
    
    output->clear();
    output->reserve(input->size());
    
    // Transform each point
    for (size_t i = 0; i < input->size(); ++i) {
        const auto& point = (*input)[i];
        Eigen::Vector4f homogeneous_point(point.x, point.y, point.z, 1.0f);
        Eigen::Vector4f transformed = transformation * homogeneous_point;
        
        output->push_back(transformed.x(), transformed.y(), transformed.z());
    }
}

void copy_point_cloud(const PointCloud::ConstPtr& input, PointCloud::Ptr& output) {
    if (!input) {
        spdlog::error("Input point cloud is null");
        return;
    }
    
    if (!output) {
        output = std::make_shared<PointCloud>();
    }
    
    output->clear();
    output->reserve(input->size());
    
    // Copy each point
    for (size_t i = 0; i < input->size(); ++i) {
        output->push_back((*input)[i]);
    }
}

} // namespace util
} // namespace lidar_odometry