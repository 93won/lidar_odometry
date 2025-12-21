/**
 * @file      VoxelMap.cpp
 * @brief     Implementation of 2-Level hierarchical voxel map with precomputed surfels
 * @author    Seungwon Choi
 * @date      2025-12-08
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "VoxelMap.h"
#include <algorithm>
#include <cmath>
#include "util/LogUtils.h"

namespace lidar_slam {
namespace map {

VoxelMap::VoxelMap(float voxel_size) 
    : m_voxel_size(voxel_size)
    , m_max_hit_count(10)
    , m_hierarchy_factor(3)
{
}

void VoxelMap::SetVoxelSize(float size) {
    if (size <= 0.0f) {
        throw std::invalid_argument("Voxel size must be positive");
    }
    
    if (std::abs(m_voxel_size - size) > 1e-6f) {
        m_voxel_size = size;
        Clear();
    }
}

void VoxelMap::SetHierarchyFactor(int factor) {
    if (factor <= 0 || factor % 2 == 0) {
        LOG_ERROR("[VoxelMap] Hierarchy factor must be positive and odd. Got: {}", factor);
        return;
    }
    
    if (m_hierarchy_factor != factor) {
        m_hierarchy_factor = factor;
        Clear();
    }
}

VoxelKey VoxelMap::PointToVoxelKey(const Eigen::Vector3f& point, int level) const {
    float scale = m_voxel_size;
    if (level == 1) scale *= static_cast<float>(m_hierarchy_factor);
    
    int vx = static_cast<int>(std::floor(point.x() / scale));
    int vy = static_cast<int>(std::floor(point.y() / scale));
    int vz = static_cast<int>(std::floor(point.z() / scale));
    return VoxelKey(vx, vy, vz);
}

VoxelKey VoxelMap::GetParentKey(const VoxelKey& key) const {
    int f = m_hierarchy_factor;
    return VoxelKey(
        key.x >= 0 ? key.x / f : (key.x - (f - 1)) / f,
        key.y >= 0 ? key.y / f : (key.y - (f - 1)) / f,
        key.z >= 0 ? key.z / f : (key.z - (f - 1)) / f
    );
}

Eigen::Vector3f VoxelMap::VoxelKeyToCenter(const VoxelKey& key) const {
    return Eigen::Vector3f(
        (key.x + 0.5f) * m_voxel_size,
        (key.y + 0.5f) * m_voxel_size,
        (key.z + 0.5f) * m_voxel_size
    );
}

void VoxelMap::RegisterToParent(const VoxelKey& key_L0) {
    VoxelKey parent_L1 = GetParentKey(key_L0);
    m_voxels_L1[parent_L1].occupied_children.insert(key_L0);
}

void VoxelMap::UnregisterFromParent(const VoxelKey& key_L0) {
    VoxelKey parent_L1 = GetParentKey(key_L0);
    
    auto it_L1 = m_voxels_L1.find(parent_L1);
    if (it_L1 == m_voxels_L1.end()) return;
    
    it_L1->second.occupied_children.erase(key_L0);
    
    if (it_L1->second.occupied_children.size() < 5) {
        it_L1->second.has_surfel = false;
    }
    
    if (it_L1->second.occupied_children.empty()) {
        m_voxels_L1.erase(it_L1);
    }
}

void VoxelMap::AddPoint(const Eigen::Vector3f& point) {
    VoxelKey key = PointToVoxelKey(point, 0);
    
    auto it = m_voxels_L0.find(key);
    bool was_empty = (it == m_voxels_L0.end());
    
    VoxelNode_L0& voxel = m_voxels_L0[key];
    
    int n = voxel.point_count;
    if (n == 0) {
        voxel.centroid = point;
        voxel.hit_count = m_init_hit_count;
        voxel.point_count = 1;
    } else {
        voxel.centroid = (voxel.centroid * static_cast<float>(n) + point) / static_cast<float>(n + 1);
        voxel.point_count++;
    }
    
    if (was_empty) {
        RegisterToParent(key);
    }
}

void VoxelMap::Clear() {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    m_voxels_L0.clear();
    m_voxels_L1.clear();
}

void VoxelMap::UpdateVoxelMap(const PointCloudConstPtr& new_cloud,
                               const Eigen::Vector3d& sensor_position,
                               double max_distance,
                               bool is_keyframe) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    
    if (!new_cloud || new_cloud->empty()) {
        return;
    }
    
    // Only add points for keyframes
    if (!is_keyframe) {
        return;
    }
    
    Eigen::Vector3f sensor_pos = sensor_position.cast<float>();
    float radius_sq = static_cast<float>(max_distance * max_distance);
    
    // Remove voxels outside radius from sensor position
    std::vector<VoxelKey> voxels_to_remove;
    for (const auto& [key, node] : m_voxels_L0) {
        float dist_sq = (node.centroid - sensor_pos).squaredNorm();
        if (dist_sq > radius_sq) {
            voxels_to_remove.push_back(key);
        }
    }
    
    for (const auto& key : voxels_to_remove) {
        UnregisterFromParent(key);
        m_voxels_L0.erase(key);
    }
    
    // Clean up empty L1 voxels
    std::vector<VoxelKey> L1_to_remove;
    for (const auto& [key, node] : m_voxels_L1) {
        if (node.occupied_children.empty()) {
            L1_to_remove.push_back(key);
        }
    }
    for (const auto& key : L1_to_remove) {
        m_voxels_L1.erase(key);
    }
    
    // Add new points
    ankerl::unordered_dense::set<VoxelKey, VoxelKeyHash> affected_L1;
    
    for (const auto& pt : *new_cloud) {
        Eigen::Vector3f point(pt.x, pt.y, pt.z);
        AddPoint(point);
        
        VoxelKey key_L1 = PointToVoxelKey(point, 1);
        affected_L1.insert(key_L1);
    }
    
    // Skip surfel computation if not needed (KDTree-based correspondence)
    if (!m_compute_surfels) {
        return;
    }
    
    // Update surfels for affected L1 voxels
    const int MIN_OCCUPIED_CHILDREN = 5;
    
    for (const VoxelKey& key_L1 : affected_L1) {
        auto it_L1 = m_voxels_L1.find(key_L1);
        if (it_L1 == m_voxels_L1.end()) continue;
        
        VoxelNode_L1& node_L1 = it_L1->second;
        int current_child_count = static_cast<int>(node_L1.occupied_children.size());
        
        if (current_child_count < MIN_OCCUPIED_CHILDREN) {
            node_L1.has_surfel = false;
            continue;
        }
        
        // Skip if child count didn't change (incremental update)
        if (node_L1.has_surfel && node_L1.last_child_count == current_child_count) {
            continue;
        }
        
        // Collect centroids from occupied L0 children
        std::vector<Eigen::Vector3f> centroids;
        centroids.reserve(node_L1.occupied_children.size());
        
        for (const VoxelKey& key_L0 : node_L1.occupied_children) {
            auto it_L0 = m_voxels_L0.find(key_L0);
            if (it_L0 != m_voxels_L0.end()) {
                centroids.push_back(it_L0->second.centroid);
            }
        }
        
        if (centroids.size() < 3) {
            node_L1.has_surfel = false;
            continue;
        }
        
        // Compute centroid
        Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
        for (const auto& pt : centroids) {
            centroid += pt;
        }
        centroid /= static_cast<float>(centroids.size());
        
        // Compute covariance matrix
        Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
        for (const auto& pt : centroids) {
            Eigen::Vector3f diff = pt - centroid;
            covariance += diff * diff.transpose();
        }
        covariance /= static_cast<float>(centroids.size());
        
        // SVD decomposition for plane fitting
        Eigen::JacobiSVD<Eigen::Matrix3f> svd(covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector3f singular_values = svd.singularValues();
        Eigen::Vector3f normal = svd.matrixU().col(2);
        float planarity = singular_values(2) / (singular_values(0) + 1e-6f);
        
        if (planarity > m_planarity_threshold) {
            // Not planar enough - invalidate and remove
            node_L1.has_surfel = false;
            
            for (const VoxelKey& key_L0 : node_L1.occupied_children) {
                m_voxels_L0.erase(key_L0);
            }
            m_voxels_L1.erase(it_L1);
            continue;
        }
        
        // Store surfel
        node_L1.has_surfel = true;
        node_L1.surfel_normal = normal;
        node_L1.surfel_centroid = centroid;
        node_L1.planarity_score = planarity;
        node_L1.last_child_count = current_child_count;
        
        // Gaussian Primitive: Store covariance and eigen decomposition
        node_L1.has_gaussian = true;
        node_L1.gaussian_mean = centroid;
        node_L1.gaussian_covariance = covariance;
        
        // Eigen decomposition for visualization (sorted eigenvalues)
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance);
        node_L1.eigenvalues = eigen_solver.eigenvalues();      // λ1 ≤ λ2 ≤ λ3 (sorted ascending)
        node_L1.eigenvectors = eigen_solver.eigenvectors();    // V = [v1, v2, v3]
    }
}

void VoxelMap::ApplyTransformAndRehash(const Eigen::Matrix4f& T_correction) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    
    Eigen::Matrix3f R = T_correction.block<3,3>(0,0);
    Eigen::Vector3f t = T_correction.block<3,1>(0,3);
    
    // Transform all L0 voxels
    std::vector<std::pair<VoxelKey, VoxelNode_L0>> transformed;
    transformed.reserve(m_voxels_L0.size());
    
    for (auto& [key, node] : m_voxels_L0) {
        VoxelNode_L0 new_node = node;
        new_node.centroid = R * node.centroid + t;
        
        VoxelKey new_key = PointToVoxelKey(new_node.centroid, 0);
        transformed.emplace_back(new_key, std::move(new_node));
    }
    
    // Clear and rebuild
    m_voxels_L0.clear();
    m_voxels_L1.clear();
    
    for (auto& [key, node] : transformed) {
        auto& existing = m_voxels_L0[key];
        if (existing.point_count == 0) {
            existing = std::move(node);
        } else {
            // Merge
            float n1 = static_cast<float>(existing.point_count);
            float n2 = static_cast<float>(node.point_count);
            existing.centroid = (existing.centroid * n1 + node.centroid * n2) / (n1 + n2);
            existing.point_count += node.point_count;
        }
        RegisterToParent(key);
    }
    
    // Recompute all surfels
    RecomputeAllSurfels();
}

void VoxelMap::RecomputeAllSurfels() {
    // Recompute surfels for all L1 voxels
    const int MIN_OCCUPIED_CHILDREN = 5;
    
    for (auto& [key_L1, node_L1] : m_voxels_L1) {
        int current_child_count = static_cast<int>(node_L1.occupied_children.size());
        
        if (current_child_count < MIN_OCCUPIED_CHILDREN) {
            node_L1.has_surfel = false;
            continue;
        }
        
        // Collect L0 centroids
        std::vector<Eigen::Vector3f> centroids;
        centroids.reserve(current_child_count);
        
        for (const VoxelKey& key_L0 : node_L1.occupied_children) {
            auto it = m_voxels_L0.find(key_L0);
            if (it != m_voxels_L0.end()) {
                centroids.push_back(it->second.centroid);
            }
        }
        
        if (centroids.size() < static_cast<size_t>(MIN_OCCUPIED_CHILDREN)) {
            node_L1.has_surfel = false;
            continue;
        }
        
        // Compute centroid
        Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
        for (const auto& c : centroids) {
            centroid += c;
        }
        centroid /= static_cast<float>(centroids.size());
        
        // Compute covariance matrix
        Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
        for (const auto& c : centroids) {
            Eigen::Vector3f diff = c - centroid;
            cov += diff * diff.transpose();
        }
        cov /= static_cast<float>(centroids.size());
        
        // SVD to find normal (smallest eigenvalue eigenvector)
        Eigen::JacobiSVD<Eigen::Matrix3f> svd(cov, Eigen::ComputeFullU);
        Eigen::Vector3f normal = svd.matrixU().col(2);
        Eigen::Vector3f singular_values = svd.singularValues();
        
        // Planarity check
        float planarity = singular_values(2) / (singular_values(0) + 1e-6f);
        if (planarity > m_planarity_threshold) {
            node_L1.has_surfel = false;
            node_L1.has_gaussian = false;
            continue;
        }
        
        // Store surfel
        node_L1.has_surfel = true;
        node_L1.surfel_normal = normal;
        node_L1.surfel_centroid = centroid;
        node_L1.planarity_score = planarity;
        node_L1.last_child_count = current_child_count;
        
        // Gaussian Primitive: Store covariance and eigen decomposition
        node_L1.has_gaussian = true;
        node_L1.gaussian_mean = centroid;
        node_L1.gaussian_covariance = cov;
        
        // Eigen decomposition for visualization (sorted eigenvalues)
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(cov);
        node_L1.eigenvalues = eigen_solver.eigenvalues();      // λ1 ≤ λ2 ≤ λ3 (sorted ascending)
        node_L1.eigenvectors = eigen_solver.eigenvectors();    // V = [v1, v2, v3]
    }
}

bool VoxelMap::GetSurfelAtPoint(const Eigen::Vector3f& point,
                                 Eigen::Vector3f& normal,
                                 Eigen::Vector3f& centroid) const {
    VoxelKey key_L1 = PointToVoxelKey(point, 1);
    
    auto it = m_voxels_L1.find(key_L1);
    if (it == m_voxels_L1.end()) {
        return false;
    }
    
    const VoxelNode_L1& node = it->second;
    if (!node.has_surfel) {
        return false;
    }
    
    normal = node.surfel_normal;
    centroid = node.surfel_centroid;
    return true;
}

VoxelMap::PointCloudPtr VoxelMap::GetPointCloud() const {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    
    auto cloud = std::make_shared<PointCloud>();
    cloud->reserve(m_voxels_L0.size());
    
    for (const auto& [key, node] : m_voxels_L0) {
        Point3D pt;
        pt.x = node.centroid.x();
        pt.y = node.centroid.y();
        pt.z = node.centroid.z();
        cloud->push_back(pt);
    }
    
    return cloud;
}

std::vector<std::tuple<Eigen::Vector3f, Eigen::Vector3f, float>> VoxelMap::GetL1Surfels() const {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    
    std::vector<std::tuple<Eigen::Vector3f, Eigen::Vector3f, float>> surfels;
    surfels.reserve(m_voxels_L1.size());
    
    for (const auto& [key, node] : m_voxels_L1) {
        if (node.has_surfel) {
            surfels.emplace_back(node.surfel_centroid, node.surfel_normal, node.planarity_score);
        }
    }
    
    return surfels;
}

std::vector<std::tuple<Eigen::Vector3f, Eigen::Matrix3f, Eigen::Vector3f, Eigen::Matrix3f>> VoxelMap::GetL1Gaussians() const {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    
    std::vector<std::tuple<Eigen::Vector3f, Eigen::Matrix3f, Eigen::Vector3f, Eigen::Matrix3f>> gaussians;
    gaussians.reserve(m_voxels_L1.size());
    
    for (const auto& [key, node] : m_voxels_L1) {
        if (node.has_gaussian) {
            gaussians.emplace_back(
                node.gaussian_mean,        // μ
                node.gaussian_covariance,  // Σ
                node.eigenvalues,          // λ1, λ2, λ3
                node.eigenvectors          // V = [v1, v2, v3]
            );
        }
    }
    
    return gaussians;
}

void VoxelMap::RebuildKdTree() {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    
    // Get all L0 centroids as point cloud
    m_kdtree_cloud = GetPointCloud();
    
    if (!m_kdtree_cloud || m_kdtree_cloud->empty()) {
        m_kdtree.reset();
        m_kdtree_dirty = false;
        return;
    }
    
    // Build KDTree
    m_kdtree = std::make_shared<lidar_slam::util::KdTree>();
    m_kdtree->setInputCloud(m_kdtree_cloud);
    m_kdtree_dirty = false;
    
    LOG_DEBUG("[VoxelMap] KDTree rebuilt with {} points", m_kdtree_cloud->size());
}

} // namespace map
} // namespace lidar_slam
