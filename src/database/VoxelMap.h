/**
 * @file      VoxelMap.h
 * @brief     2-Level hierarchical voxel map with precomputed surfels and fast voxel filter
 * @author    Seungwon Choi
 * @date      2025-12-08
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 * 
 * @details
 * Contains:
 * - FastVoxelFilter: Fast voxel grid filter using Morton code
 * - VoxelMap: 2-Level hierarchical voxel map with precomputed surfels
 * 
 * 2-Level Hierarchy (VoxelMap):
 * - L0: Leaf voxels (voxel_size) - stores centroid only (no raw points)
 * - L1: Parent voxels (hierarchy_factor × voxel_size) - stores precomputed surfels
 * 
 * Surfel computation:
 * - Collects L0 centroids within L1 voxel
 * - Fits plane using PCA (SVD of covariance matrix)
 * - Caches normal, centroid, planarity score
 * - Updates only when L0 child count changes (incremental)
 */

#ifndef LIDAR_ODOMETRY_VOXEL_MAP_H
#define LIDAR_ODOMETRY_VOXEL_MAP_H

#include "../util/PointCloudUtils.h"
#include "../../thirdparty/unordered_dense/unordered_dense.h"
#include <vector>
#include <mutex>
#include <Eigen/Dense>
#include <cmath>

namespace lidar_slam {
namespace map {

// ============================================================================
// FastVoxelFilter: Fast voxel grid filter using Morton code
// ============================================================================

/**
 * @brief Fast voxel grid filter using Z-order Morton code and Robin Hood hashing
 * 
 * Key optimizations:
 * - Stride-based point skipping for initial downsampling
 * - Morton code (Z-order curve) for spatial locality
 * - Robin Hood hashing (unordered_dense) for cache-friendly O(1) lookup
 * - Single-pass filtering with centroid accumulation
 */
class FastVoxelFilter {
public:
    explicit FastVoxelFilter(float voxel_size = 0.5f)
        : m_voxel_size(voxel_size)
        , m_inv_voxel_size(1.0f / voxel_size) {
    }
    
    void setVoxelSize(float voxel_size) {
        m_voxel_size = voxel_size;
        m_inv_voxel_size = 1.0f / voxel_size;
    }
    
    float getVoxelSize() const { return m_voxel_size; }
    
    /**
     * @brief Filter point cloud with stride and voxel grid
     * @param input Input point cloud
     * @param output Output filtered point cloud (cleared and filled)
     * @param stride Process every N-th point (1 = all points)
     */
    void filter(const util::PointCloud& input, util::PointCloud& output, int stride = 1) {
        output.clear();
        
        if (input.empty() || stride < 1) return;
        
        m_voxel_map.clear();
        m_voxel_map.reserve(input.size() / (stride * 8));
        
        // First pass: accumulate points into voxels with stride
        for (size_t i = 0; i < input.size(); i += stride) {
            const auto& pt = input[i];
            if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;
            
            uint64_t key = computeMortonKey(pt.x, pt.y, pt.z);
            auto& voxel = m_voxel_map[key];
            voxel.sum_x += pt.x;
            voxel.sum_y += pt.y;
            voxel.sum_z += pt.z;
            voxel.count++;
        }
        
        // Second pass: extract centroids
        output.reserve(m_voxel_map.size());
        for (const auto& [key, voxel] : m_voxel_map) {
            float inv_count = 1.0f / static_cast<float>(voxel.count);
            output.push_back(util::Point3D(
                voxel.sum_x * inv_count,
                voxel.sum_y * inv_count,
                voxel.sum_z * inv_count
            ));
        }
    }
    
    size_t getVoxelCount() const { return m_voxel_map.size(); }

private:
    struct VoxelData {
        float sum_x = 0.0f, sum_y = 0.0f, sum_z = 0.0f;
        uint32_t count = 0;
    };
    
    static inline uint64_t expandBits(uint64_t v) {
        v = v & 0x1FFFFF;
        v = (v | (v << 32)) & 0x1F00000000FFFF;
        v = (v | (v << 16)) & 0x1F0000FF0000FF;
        v = (v | (v << 8))  & 0x100F00F00F00F00F;
        v = (v | (v << 4))  & 0x10C30C30C30C30C3;
        v = (v | (v << 2))  & 0x1249249249249249;
        return v;
    }
    
    inline uint64_t computeMortonKey(float x, float y, float z) const {
        constexpr int64_t OFFSET = (1 << 20);
        int64_t ix = static_cast<int64_t>(std::floor(x * m_inv_voxel_size)) + OFFSET;
        int64_t iy = static_cast<int64_t>(std::floor(y * m_inv_voxel_size)) + OFFSET;
        int64_t iz = static_cast<int64_t>(std::floor(z * m_inv_voxel_size)) + OFFSET;
        ix = std::max<int64_t>(0, std::min<int64_t>(ix, (1 << 21) - 1));
        iy = std::max<int64_t>(0, std::min<int64_t>(iy, (1 << 21) - 1));
        iz = std::max<int64_t>(0, std::min<int64_t>(iz, (1 << 21) - 1));
        return expandBits(static_cast<uint64_t>(ix)) |
               (expandBits(static_cast<uint64_t>(iy)) << 1) |
               (expandBits(static_cast<uint64_t>(iz)) << 2);
    }
    
    float m_voxel_size;
    float m_inv_voxel_size;
    ankerl::unordered_dense::map<uint64_t, VoxelData> m_voxel_map;
};

// Backward compatibility alias
using FastVoxelGrid = FastVoxelFilter;

// ============================================================================
// VoxelMap: 2-Level hierarchical voxel map with precomputed surfels
// ============================================================================

/**
 * @brief Voxel key for spatial hashing
 */
struct VoxelKey {
    int x, y, z;
    
    VoxelKey() : x(0), y(0), z(0) {}
    VoxelKey(int x_, int y_, int z_) : x(x_), y(y_), z(z_) {}
    
    bool operator==(const VoxelKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

/**
 * @brief Z-order (Morton code) hash function for VoxelKey
 */
struct VoxelKeyHash {
private:
    static inline uint64_t ExpandBits(int32_t v) {
        uint64_t x = static_cast<uint64_t>(v + (1 << 20)) & 0x1fffff;
        x = (x | (x << 32)) & 0x1f00000000ffffULL;
        x = (x | (x << 16)) & 0x1f0000ff0000ffULL;
        x = (x | (x << 8))  & 0x100f00f00f00f00fULL;
        x = (x | (x << 4))  & 0x10c30c30c30c30c3ULL;
        x = (x | (x << 2))  & 0x1249249249249249ULL;
        return x;
    }
    
public:
    std::size_t operator()(const VoxelKey& key) const {
        uint64_t morton = ExpandBits(key.x) | (ExpandBits(key.y) << 1) | (ExpandBits(key.z) << 2);
        return static_cast<std::size_t>(morton);
    }
};

/**
 * @brief 2-Level hierarchical voxel map with precomputed surfels
 */
class VoxelMap {
public:
    using PointCloud = lidar_slam::util::PointCloud;
    using PointCloudPtr = lidar_slam::util::PointCloudPtr;
    using PointCloudConstPtr = lidar_slam::util::PointCloudConstPtr;
    using Point3D = lidar_slam::util::Point3D;

    explicit VoxelMap(float voxel_size = 0.5f);
    
    void SetVoxelSize(float size);
    void SetMaxHitCount(int max_count) { m_max_hit_count = max_count; }
    void SetInitHitCount(int count) { m_init_hit_count = count; }
    void SetHierarchyFactor(int factor);
    void SetPlanarityThreshold(float threshold) { m_planarity_threshold = threshold; }
    void SetComputeSurfels(bool compute) { m_compute_surfels = compute; }
    
    float GetVoxelSize() const { return m_voxel_size; }
    int GetHierarchyFactor() const { return m_hierarchy_factor; }
    size_t GetVoxelCount() const { return m_voxels_L0.size(); }
    size_t GetL1VoxelCount() const { return m_voxels_L1.size(); }
    bool empty() const { return m_voxels_L0.empty(); }
    bool GetComputeSurfels() const { return m_compute_surfels; }
    
    /**
     * @brief Get number of L1 voxels with valid surfels
     */
    size_t GetSurfelCount() const {
        size_t count = 0;
        for (const auto& [key, node] : m_voxels_L1) {
            if (node.has_surfel) count++;
        }
        return count;
    }
    
    /**
     * @brief Clear all voxels
     */
    void Clear();
    
    /**
     * @brief Update voxel map with new point cloud
     * @param new_cloud New points in world coordinates
     * @param sensor_position Current sensor position
     * @param max_distance Maximum distance to keep voxels
     * @param is_keyframe Whether this is a keyframe (triggers surfel update)
     */
    void UpdateVoxelMap(const PointCloudConstPtr& new_cloud,
                        const Eigen::Vector3d& sensor_position,
                        double max_distance,
                        bool is_keyframe);
    
    /**
     * @brief Apply rigid transformation and rehash all voxels (for PGO correction)
     * @param T_correction Correction transformation matrix
     */
    void ApplyTransformAndRehash(const Eigen::Matrix4f& T_correction);
    
    /**
     * @brief Get surfel at a query point (O(1) lookup)
     * @param point Query point in world coordinates
     * @param normal Output: surfel normal
     * @param centroid Output: surfel centroid
     * @return True if valid surfel found
     */
    bool GetSurfelAtPoint(const Eigen::Vector3f& point,
                          Eigen::Vector3f& normal,
                          Eigen::Vector3f& centroid) const;
    
    /**
     * @brief Get point cloud of all L0 centroids (for visualization)
     */
    PointCloudPtr GetPointCloud() const;
    
    /**
     * @brief Get KDTree of L0 centroids for correspondence search
     * @return Shared pointer to KDTree (may be nullptr if not built)
     */
    std::shared_ptr<lidar_slam::util::KdTree> GetKdTree() const { return m_kdtree; }
    
    /**
     * @brief Rebuild KDTree from current L0 centroids
     * Call this after adding new points/keyframes when using KDTree-based correspondence
     */
    void RebuildKdTree();
    
    /**
     * @brief Check if KDTree is built and ready
     */
    bool HasKdTree() const { return m_kdtree != nullptr; }
    
    /**
     * @brief Get all L1 surfels for visualization
     * @return Vector of (centroid, normal, planarity_score) tuples
     */
    std::vector<std::tuple<Eigen::Vector3f, Eigen::Vector3f, float>> GetL1Surfels() const;
    
    /**
     * @brief Get all L1 Gaussian primitives for visualization
     * @return Vector of (mean, covariance, eigenvalues, eigenvectors) tuples
     */
    std::vector<std::tuple<Eigen::Vector3f, Eigen::Matrix3f, Eigen::Vector3f, Eigen::Matrix3f>> GetL1Gaussians() const;
    
    /**
     * @brief Get number of L1 voxels with valid Gaussian primitives
     */
    size_t GetGaussianCount() const {
        size_t count = 0;
        for (const auto& [key, node] : m_voxels_L1) {
            if (node.has_gaussian) count++;
        }
        return count;
    }
    
private:
    VoxelKey PointToVoxelKey(const Eigen::Vector3f& point, int level = 0) const;
    VoxelKey GetParentKey(const VoxelKey& key) const;
    void AddPoint(const Eigen::Vector3f& point);
    void RegisterToParent(const VoxelKey& key_L0);
    void UnregisterFromParent(const VoxelKey& key_L0);
    Eigen::Vector3f VoxelKeyToCenter(const VoxelKey& key) const;
    void RecomputeAllSurfels();
    
    // Parameters
    float m_voxel_size;
    int m_max_hit_count = 10;
    int m_init_hit_count = 1;
    int m_hierarchy_factor = 3;  // L1 = 3×3×3 L0 voxels
    float m_planarity_threshold = 0.1f;  // Default: 0.1 (less strict)
    bool m_compute_surfels = true;  // If false, skip surfel computation (for KDTree-based correspondence)
    
    // Level 0: Leaf voxels (centroid only, no raw points)
    struct VoxelNode_L0 {
        Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
        int hit_count = 1;
        int point_count = 0;
        
        VoxelNode_L0() = default;
    };
    ankerl::unordered_dense::map<VoxelKey, VoxelNode_L0, VoxelKeyHash> m_voxels_L0;
    
    // Level 1: Parent voxels with precomputed surfels and Gaussian primitives
    struct VoxelNode_L1 {
        ankerl::unordered_dense::set<VoxelKey, VoxelKeyHash> occupied_children;
        
        // Surfel data (legacy)
        bool has_surfel = false;
        Eigen::Vector3f surfel_normal = Eigen::Vector3f::Zero();
        Eigen::Vector3f surfel_centroid = Eigen::Vector3f::Zero();
        float planarity_score = 1.0f;
        int last_child_count = 0;
        
        // Gaussian Primitive data (new!)
        bool has_gaussian = false;
        Eigen::Vector3f gaussian_mean = Eigen::Vector3f::Zero();       // μ
        Eigen::Matrix3f gaussian_covariance = Eigen::Matrix3f::Zero(); // Σ
        Eigen::Vector3f eigenvalues = Eigen::Vector3f::Zero();         // λ1 ≤ λ2 ≤ λ3
        Eigen::Matrix3f eigenvectors = Eigen::Matrix3f::Identity();    // V = [v1, v2, v3]
        
        VoxelNode_L1() = default;
    };
    ankerl::unordered_dense::map<VoxelKey, VoxelNode_L1, VoxelKeyHash> m_voxels_L1;
    
    // KDTree for L0 centroids (for KDTree-based correspondence search)
    std::shared_ptr<lidar_slam::util::KdTree> m_kdtree;
    PointCloudPtr m_kdtree_cloud;  // Point cloud used for KDTree
    bool m_kdtree_dirty = true;    // Flag to indicate KDTree needs rebuild
    
    mutable std::recursive_mutex m_mutex;
};

} // namespace map
} // namespace lidar_slam

#endif // LIDAR_ODOMETRY_VOXEL_MAP_H
