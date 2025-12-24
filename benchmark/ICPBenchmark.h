/**
 * @file      ICPBenchmark.h
 * @brief     Pure ICP Algorithm Benchmark Framework (Point-level, k-NN based)
 * @author    Seungwon Choi
 * @date      2025-12-22
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @details   Benchmark framework for comparing pure ICP algorithms:
 *            - Point-to-Point ICP: ||p - q||²
 *            - Point-to-Plane ICP: (n^T(p - q))²
 *            - Symmetric ICP: ((n_p + n_q)^T(p - q))²
 *            - GICP: d^T(Σ_p + RΣ_qR^T)^{-1}d
 *            - MC-ICP (Ours): d^T Ω_geo d (Manifold-Constrained)
 *
 *            Uses KITTI dataset for evaluation with ground truth poses.
 *            Multi-scale voxel downsampling for comprehensive comparison.
 *            No PKO (robust loss) - pure ICP performance only.
 */

#pragma once

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <map>
#include <unordered_map>
#include "../thirdparty/nanoflann/nanoflann.hpp"

namespace lidar_slam {

// Forward declarations
namespace util {
class PointCloud;
using PointCloudPtr = std::shared_ptr<PointCloud>;
}

namespace benchmark {

// ============================================================================
// ICP Method Types
// ============================================================================

enum class ICPMethod {
    POINT_TO_POINT,      ///< Classic point-to-point ICP: ||p - q||²
    POINT_TO_PLANE,      ///< Point-to-plane ICP: (n^T(p - q))²
    SYMMETRIC,           ///< Symmetric ICP: ((n_p + n_q)^T(p - q))²
    GICP,                ///< Generalized ICP: d^T(Σ_p + RΣ_qR^T)^{-1}d
    MC_ICP,              ///< Manifold-Constrained ICP (Ours): d^T Ω_geo d
    NDT                  ///< Normal Distributions Transform
};

inline std::string to_string(ICPMethod method) {
    switch (method) {
        case ICPMethod::POINT_TO_POINT: return "Point-to-Point";
        case ICPMethod::POINT_TO_PLANE: return "Point-to-Plane";
        case ICPMethod::SYMMETRIC: return "Symmetric";
        case ICPMethod::GICP: return "GICP";
        case ICPMethod::MC_ICP: return "MC-ICP";
        case ICPMethod::NDT: return "NDT";
        default: return "Unknown";
    }
}

// ============================================================================
// Data Structures
// ============================================================================

/**
 * @brief Point with covariance for GICP/MC-ICP methods
 */
struct PointWithCovariance {
    Eigen::Vector3f point;
    Eigen::Matrix3f covariance;
    Eigen::Vector3f eigenvalues;   // λ1 >= λ2 >= λ3 (descending)
    Eigen::Matrix3f eigenvectors;  // V = [v1, v2, v3]
    Eigen::Vector3f normal;        // Smallest eigenvector (for planes)
    
    PointWithCovariance() {
        point.setZero();
        covariance.setIdentity();
        eigenvalues.setOnes();
        eigenvectors.setIdentity();
        normal = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
    }
};

/**
 * @brief Point cloud with local covariances
 */
struct PointCloudWithCovariance {
    std::vector<PointWithCovariance> points;
    
    void clear() { points.clear(); }
    size_t size() const { return points.size(); }
    void reserve(size_t n) { points.reserve(n); }
    void push_back(const PointWithCovariance& p) { points.push_back(p); }
    
    // nanoflann adapter interface
    inline size_t kdtree_get_point_count() const { return points.size(); }
    
    inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
        if (dim == 0) return points[idx].point.x();
        else if (dim == 1) return points[idx].point.y();
        else return points[idx].point.z();
    }
    
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /*bb*/) const { return false; }
};

// KD-Tree type alias
using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, PointCloudWithCovariance>,
    PointCloudWithCovariance,
    3,  // dimension
    size_t
>;

// ============================================================================
// Correspondence Structures (Method-specific)
// ============================================================================

/**
 * @brief Simple correspondence for Point-to-Point ICP
 * Only stores source and target indices
 */
struct CorrespondenceP2P {
    int source_idx;
    int target_idx;
    float distance_sq;  // For debugging/filtering
};

/**
 * @brief Correspondence for Point-to-Plane and Symmetric ICP
 * Includes target normal for plane constraint
 */
struct CorrespondenceP2Plane {
    int source_idx;
    int target_idx;
    Eigen::Vector3f target_normal;
    Eigen::Vector3f source_normal;  // For Symmetric ICP
    float distance_sq;
};

/**
 * @brief Correspondence for GICP and MC-ICP
 * Includes full covariance/eigenvalue information
 */
struct CorrespondenceGICP {
    int source_idx;
    int target_idx;
    Eigen::Matrix3f source_covariance;
    Eigen::Matrix3f target_covariance;
    Eigen::Vector3f source_eigenvalues;   // For MC-ICP
    Eigen::Matrix3f source_eigenvectors;  // For MC-ICP
    Eigen::Vector3f target_eigenvalues;   // For MC-ICP
    Eigen::Matrix3f target_eigenvectors;  // For MC-ICP
    float distance_sq;
};

/**
 * @brief Iteration statistics for convergence analysis
 */
struct IterationStats {
    int iteration = 0;
    double cost = 0.0;               ///< Total cost value
    double translation_error = 0.0;  ///< Translation error from ground truth (m)
    double rotation_error = 0.0;     ///< Rotation error from ground truth (deg)
    double step_size = 0.0;          ///< Update step size
    double time_ms = 0.0;            ///< Cumulative time (ms)
    size_t num_correspondences = 0;  ///< Number of valid correspondences
};

/**
 * @brief Benchmark result for a single ICP run
 */
struct BenchmarkResult {
    ICPMethod method;
    std::string method_name;
    float voxel_size = 0.0f;         ///< Downsampling voxel size used
    
    // Convergence history (per iteration)
    std::vector<IterationStats> history;
    
    // Final statistics
    int total_iterations = 0;
    double final_cost = 0.0;
    double final_translation_error = 0.0;  ///< meters
    double final_rotation_error = 0.0;     ///< degrees
    double total_time_ms = 0.0;
    bool converged = false;
    
    // Initial state
    double initial_translation_error = 0.0;
    double initial_rotation_error = 0.0;
    size_t source_points = 0;
    size_t target_points = 0;
};

/**
 * @brief KITTI benchmark configuration
 */
struct KITTIBenchmarkConfig {
    // KITTI dataset
    std::string dataset_path;             ///< Path to KITTI velodyne folder
    std::string ground_truth_path;        ///< Path to ground truth poses
    int sequence = 0;                     ///< KITTI sequence number
    int frame_start = 0;                  ///< Start frame
    int frame_end = -1;                   ///< End frame (-1 = all frames)
    
    // For backward compatibility (single pair mode)
    int frame1 = -1;                      ///< First frame (source) - deprecated
    int frame2 = -1;                      ///< Second frame (target) - deprecated
    
    // Multi-scale downsampling
    std::vector<float> voxel_sizes = {0.1f, 0.2f, 0.5f, 1.0f};
    
    // ICP parameters
    int max_iterations = 10;
    double convergence_threshold = 1e-6;
    double max_correspondence_distance = 2.0;
    int k_neighbors = 20;                 ///< k-NN for covariance estimation
    
    // Output
    std::string output_dir = "./benchmark_results";
    bool verbose = true;
};

/**
 * @brief ICP algorithm configuration (no PKO)
 */
struct ICPConfig {
    int max_iterations = 10;
    double convergence_threshold = 1e-6;
    double translation_tolerance = 0.001;  // 1mm
    double rotation_tolerance = 0.001;     // ~0.057 degrees
    double max_correspondence_distance = 2.0;
    int k_neighbors = 20;
    bool verbose = false;  ///< Print per-iteration stats
};

// ============================================================================
// ICP Algorithm Base Class
// ============================================================================

/**
 * @brief Base class for ICP algorithms
 */
class ICPAlgorithm {
public:
    virtual ~ICPAlgorithm() = default;
    
    /**
     * @brief Run ICP optimization
     * @param source Source point cloud (to be aligned)
     * @param target Target point cloud (reference)
     * @param initial_transform Initial transform guess
     * @param config ICP configuration
     * @param ground_truth Ground truth transform for error computation
     * @return Benchmark result with convergence history
     */
    virtual BenchmarkResult run(
        const PointCloudWithCovariance& source,
        const PointCloudWithCovariance& target,
        const Eigen::Matrix4f& initial_transform,
        const ICPConfig& config,
        const Eigen::Matrix4f& ground_truth) = 0;
    
    virtual ICPMethod getMethod() const = 0;
    virtual std::string getName() const = 0;
    
protected:
    /**
     * @brief Compute pose error from ground truth
     * @return (translation_error_m, rotation_error_deg)
     */
    std::pair<double, double> computePoseError(
        const Eigen::Matrix4f& estimated,
        const Eigen::Matrix4f& ground_truth);
    
    /**
     * @brief Transform point cloud
     */
    void transformPoints(
        const PointCloudWithCovariance& source,
        PointCloudWithCovariance& transformed,
        const Eigen::Matrix4f& transform);
};

// ============================================================================
// ICP Algorithm Implementations
// ============================================================================

/**
 * @brief Point-to-Point ICP
 * Cost: sum_i ||p_i - q_i||²
 */
class PointToPointICP : public ICPAlgorithm {
public:
    BenchmarkResult run(
        const PointCloudWithCovariance& source,
        const PointCloudWithCovariance& target,
        const Eigen::Matrix4f& initial_transform,
        const ICPConfig& config,
        const Eigen::Matrix4f& ground_truth) override;
    
    ICPMethod getMethod() const override { return ICPMethod::POINT_TO_POINT; }
    std::string getName() const override { return "Point-to-Point"; }
    
private:
    /**
     * @brief Find correspondences for Point-to-Point ICP
     * Simple nearest neighbor with distance threshold only
     */
    std::vector<CorrespondenceP2P> findCorrespondencesP2P(
        const PointCloudWithCovariance& source,
        const PointCloudWithCovariance& target,
        const Eigen::Matrix4f& current_transform,
        double max_distance);
};

/**
 * @brief Point-to-Plane ICP
 * Cost: sum_i (n_i^T (p_i - q_i))²
 */
class PointToPlaneICP : public ICPAlgorithm {
public:
    BenchmarkResult run(
        const PointCloudWithCovariance& source,
        const PointCloudWithCovariance& target,
        const Eigen::Matrix4f& initial_transform,
        const ICPConfig& config,
        const Eigen::Matrix4f& ground_truth) override;
    
    ICPMethod getMethod() const override { return ICPMethod::POINT_TO_PLANE; }
    std::string getName() const override { return "Point-to-Plane"; }
    
private:
    /**
     * @brief Find correspondences for Point-to-Plane ICP
     * Nearest neighbor with plane fitting and residual check
     */
    std::vector<CorrespondenceP2Plane> findCorrespondencesP2Plane(
        const PointCloudWithCovariance& source,
        const PointCloudWithCovariance& target,
        const Eigen::Matrix4f& current_transform,
        double max_distance);
};

/**
 * @brief Symmetric ICP
 * Cost: sum_i ((n_p + n_q)^T (p_i - q_i))²
 * Reference: "A Symmetric Objective Function for ICP"
 */
class SymmetricICP : public ICPAlgorithm {
public:
    BenchmarkResult run(
        const PointCloudWithCovariance& source,
        const PointCloudWithCovariance& target,
        const Eigen::Matrix4f& initial_transform,
        const ICPConfig& config,
        const Eigen::Matrix4f& ground_truth) override;
    
    ICPMethod getMethod() const override { return ICPMethod::SYMMETRIC; }
    std::string getName() const override { return "Symmetric"; }
    
private:
    /**
     * @brief Find correspondences for Symmetric ICP
     * Nearest neighbor with plane fitting, includes both source and target normals
     */
    std::vector<CorrespondenceP2Plane> findCorrespondencesSymmetric(
        const PointCloudWithCovariance& source,
        const PointCloudWithCovariance& target,
        const Eigen::Matrix4f& current_transform,
        double max_distance);
};

/**
 * @brief Generalized ICP (GICP)
 * Cost: sum_i d_i^T (Σ_p + R*Σ_q*R^T)^{-1} d_i
 * Reference: "Generalized-ICP" Segal et al. 2009
 */
class GeneralizedICP : public ICPAlgorithm {
public:
    BenchmarkResult run(
        const PointCloudWithCovariance& source,
        const PointCloudWithCovariance& target,
        const Eigen::Matrix4f& initial_transform,
        const ICPConfig& config,
        const Eigen::Matrix4f& ground_truth) override;
    
    ICPMethod getMethod() const override { return ICPMethod::GICP; }
    std::string getName() const override { return "GICP"; }
    
private:
    /**
     * @brief Find correspondences for GICP
     * Nearest neighbor with plane fitting, includes covariance information
     */
    std::vector<CorrespondenceGICP> findCorrespondencesGICP(
        const PointCloudWithCovariance& source,
        const PointCloudWithCovariance& target,
        const Eigen::Matrix4f& current_transform,
        double max_distance);
};

/**
 * @brief Manifold-Constrained ICP (MC-ICP) - Our Proposed Method
 * Cost: sum_i d_i^T Ω_geo d_i
 * where Ω_geo = V * diag(c1/λ1, c2/λ2, c3/λ3) * V^T
 *       c_i = 1 - λ_i / tr(Σ)
 */
class ManifoldConstrainedICP : public ICPAlgorithm {
public:
    BenchmarkResult run(
        const PointCloudWithCovariance& source,
        const PointCloudWithCovariance& target,
        const Eigen::Matrix4f& initial_transform,
        const ICPConfig& config,
        const Eigen::Matrix4f& ground_truth) override;
    
    ICPMethod getMethod() const override { return ICPMethod::MC_ICP; }
    std::string getName() const override { return "MC-ICP"; }
    
    /**
     * @brief Compute geometry-aware information matrix
     * Ω_geo = V * diag(c1/λ1, c2/λ2, c3/λ3) * V^T
     */
    static Eigen::Matrix3f computeGeometryAwareOmega(
        const Eigen::Vector3f& eigenvalues,
        const Eigen::Matrix3f& eigenvectors);
    
private:
    /**
     * @brief Find correspondences for MC-ICP
     * Nearest neighbor with plane fitting, includes eigenvalue/eigenvector information
     */
    std::vector<CorrespondenceGICP> findCorrespondencesMCICP(
        const PointCloudWithCovariance& source,
        const PointCloudWithCovariance& target,
        const Eigen::Matrix4f& current_transform,
        double max_distance);
};

// ============================================================================
// NDT (Normal Distributions Transform)
// ============================================================================

/**
 * @brief NDT Voxel key for spatial hashing
 */
struct NDTVoxelKey {
    int x, y, z;
    
    NDTVoxelKey() : x(0), y(0), z(0) {}
    NDTVoxelKey(int x_, int y_, int z_) : x(x_), y(y_), z(z_) {}
    
    bool operator==(const NDTVoxelKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

/**
 * @brief Hash function for NDTVoxelKey
 */
struct NDTVoxelKeyHash {
    std::size_t operator()(const NDTVoxelKey& key) const {
        // Simple hash combining
        std::size_t h1 = std::hash<int>{}(key.x);
        std::size_t h2 = std::hash<int>{}(key.y);
        std::size_t h3 = std::hash<int>{}(key.z);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

/**
 * @brief NDT Voxel containing Gaussian distribution
 */
struct NDTVoxel {
    Eigen::Vector3f mean = Eigen::Vector3f::Zero();
    Eigen::Matrix3f covariance = Eigen::Matrix3f::Identity();
    Eigen::Matrix3f information = Eigen::Matrix3f::Identity();  // Inverse of covariance
    int point_count = 0;
    bool valid = false;  // Has enough points and valid covariance
    
    // For incremental updates
    Eigen::Vector3f sum = Eigen::Vector3f::Zero();
    Eigen::Matrix3f sum_sq = Eigen::Matrix3f::Zero();  // sum of (p * p^T)
};

/**
 * @brief NDT Voxel Map for target point cloud
 */
class NDTVoxelMap {
public:
    explicit NDTVoxelMap(float voxel_size = 1.0f, int min_points_per_voxel = 5);
    
    /**
     * @brief Build NDT map from point cloud
     */
    void build(const PointCloudWithCovariance& cloud);
    
    /**
     * @brief Look up voxel for a given point
     * @return Pointer to NDTVoxel if found and valid, nullptr otherwise
     */
    const NDTVoxel* lookup(const Eigen::Vector3f& point) const;
    
    /**
     * @brief Get voxel key for a point
     */
    NDTVoxelKey getVoxelKey(const Eigen::Vector3f& point) const;
    
    float getVoxelSize() const { return voxel_size_; }
    size_t getVoxelCount() const { return voxels_.size(); }
    size_t getValidVoxelCount() const;
    
private:
    float voxel_size_;
    float inv_voxel_size_;
    int min_points_per_voxel_;
    std::unordered_map<NDTVoxelKey, NDTVoxel, NDTVoxelKeyHash> voxels_;
    
    void finalizeVoxels();
};

/**
 * @brief Normal Distributions Transform (NDT)
 * Cost: sum_i -log( exp(-0.5 * d_i^T Σ^{-1} d_i) )
 *     = sum_i 0.5 * d_i^T Σ^{-1} d_i  (ignoring constants)
 * Reference: "The Normal Distributions Transform" Biber & Strasser 2003
 *            "3D NDT Scan Matching" Magnusson 2009
 */
class NDTICP : public ICPAlgorithm {
public:
    explicit NDTICP(float ndt_voxel_size = 1.0f);
    
    BenchmarkResult run(
        const PointCloudWithCovariance& source,
        const PointCloudWithCovariance& target,
        const Eigen::Matrix4f& initial_transform,
        const ICPConfig& config,
        const Eigen::Matrix4f& ground_truth) override;
    
    /**
     * @brief Run NDT with raw target for voxel map building
     * @param source Downsampled source cloud
     * @param target_raw Raw target cloud for NDT voxel map
     */
    BenchmarkResult runWithRawTarget(
        const PointCloudWithCovariance& source,
        const PointCloudWithCovariance& target_raw,
        const Eigen::Matrix4f& initial_transform,
        const ICPConfig& config,
        const Eigen::Matrix4f& ground_truth);
    
    ICPMethod getMethod() const override { return ICPMethod::NDT; }
    std::string getName() const override { return "NDT"; }
    
    void setNDTVoxelSize(float size) { ndt_voxel_size_ = size; }
    float getNDTVoxelSize() const { return ndt_voxel_size_; }
    
private:
    float ndt_voxel_size_;
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Compute local covariances for a point cloud using k-NN
 */
void computeLocalCovariances(
    PointCloudWithCovariance& cloud,
    int k_neighbors);

/**
 * @brief Convert from util::PointCloud to PointCloudWithCovariance
 */
PointCloudWithCovariance convertPointCloud(
    const util::PointCloudPtr& cloud);

/**
 * @brief Voxel downsample a point cloud
 */
PointCloudWithCovariance voxelDownsample(
    const PointCloudWithCovariance& cloud,
    float voxel_size);

// ============================================================================
// KITTI Data Loading
// ============================================================================

/**
 * @brief Load KITTI point cloud from .bin file
 */
PointCloudWithCovariance loadKITTIPointCloud(
    const std::string& bin_file_path);

/**
 * @brief Load KITTI ground truth poses
 * @return Vector of 4x4 pose matrices
 */
std::vector<Eigen::Matrix4f> loadKITTIGroundTruth(
    const std::string& pose_file_path);

/**
 * @brief Get KITTI bin file path for a frame
 */
std::string getKITTIBinPath(
    const std::string& dataset_path,
    int sequence,
    int frame);

// ============================================================================
// Benchmark Runner
// ============================================================================

/**
 * @brief Main benchmark runner for KITTI dataset
 */
class KITTIBenchmarkRunner {
public:
    explicit KITTIBenchmarkRunner(const KITTIBenchmarkConfig& config);
    
    /**
     * @brief Run benchmark on all ICP methods with multi-scale downsampling
     * @return Results organized by (voxel_size, method)
     */
    std::map<float, std::vector<BenchmarkResult>> runAll();
    
    /**
     * @brief Run benchmark on specific methods
     */
    std::map<float, std::vector<BenchmarkResult>> run(
        const std::vector<ICPMethod>& methods);
    
    /**
     * @brief Print results to console
     */
    void printResults(const std::map<float, std::vector<BenchmarkResult>>& results);
    
    /**
     * @brief Export results to CSV
     */
    void exportToCSV(
        const std::map<float, std::vector<BenchmarkResult>>& results,
        const std::string& filename);
    
    /**
     * @brief Export convergence curves to CSV (for plotting)
     */
    void exportConvergenceCurves(
        const std::map<float, std::vector<BenchmarkResult>>& results,
        const std::string& filename);

private:
    KITTIBenchmarkConfig config_;
    std::map<ICPMethod, std::unique_ptr<ICPAlgorithm>> algorithms_;
    
    void initializeAlgorithms();
    
    /**
     * @brief Run single benchmark for given voxel size
     * @param target_raw Raw target cloud for NDT (builds voxel map from raw points)
     */
    std::vector<BenchmarkResult> runForVoxelSize(
        float voxel_size,
        const std::vector<ICPMethod>& methods,
        const PointCloudWithCovariance& source,
        const PointCloudWithCovariance& target,
        const PointCloudWithCovariance& target_raw,
        const Eigen::Matrix4f& initial_transform,
        const Eigen::Matrix4f& ground_truth);
};

} // namespace benchmark
} // namespace lidar_slam
