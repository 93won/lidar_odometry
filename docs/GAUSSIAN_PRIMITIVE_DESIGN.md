# Gaussian Primitive Design for Unified LiDAR Odometry

## Overview

This document describes the design and implementation of **Gaussian Primitives** as a unified geometry representation for LiDAR odometry, replacing traditional feature-based approaches (planes, edges, corners) with a single probabilistic framework.

---

## 1. Motivation: Why Gaussian Primitives?

### 1.1 Problem with Traditional Approaches

Traditional LiDAR odometry methods (e.g., LOAM, LeGO-LOAM) explicitly classify geometric features:

```cpp
// Traditional approach: Hard classification
if (planarity > 0.8) {
    residual = point_to_plane_distance(p, normal);
} else if (linearity > 0.8) {
    residual = point_to_line_distance(p, direction);
} else {
    residual = point_to_point_distance(p, target);
}
```

**Problems:**
1. **Hard thresholds** introduce discontinuities in the optimization landscape
2. **Curved surfaces** cannot be properly represented (not plane, not edge, not corner)
3. **Multiple code paths** increase complexity and maintenance burden
4. **Information loss** - eigenvalues contain rich geometry info, but only normal vector is used

### 1.2 Key Insight: Covariance Encodes Geometry

The covariance matrix $\boldsymbol{\Sigma}$ of points within a voxel naturally encodes the local surface geometry:

| Geometry | Eigenvalue Pattern | Covariance Shape |
|----------|-------------------|------------------|
| **Plane** | $\lambda_1 \ll \lambda_2 \approx \lambda_3$ | Flat pancake ellipsoid |
| **Edge** | $\lambda_1 \approx \lambda_2 \ll \lambda_3$ | Needle-like ellipsoid |
| **Corner** | $\lambda_1 \approx \lambda_2 \approx \lambda_3$ | Sphere |
| **Curved surface** | $\lambda_1 < \lambda_2 < \lambda_3$ (all different) | General ellipsoid |

**Key insight:** Instead of classifying geometry and applying different residuals, we use the covariance matrix directly as an **anisotropic weighting matrix** in a unified optimization.

### 1.3 Proposed Solution: Gaussian Primitive

Each L1 voxel stores a Gaussian primitive:

```cpp
struct GaussianPrimitive {
    Eigen::Vector3f μ;        // Mean (centroid from L0 centroids)
    Eigen::Matrix3f Σ;        // Covariance matrix
    Eigen::Vector3f λ;        // Eigenvalues: λ1 ≤ λ2 ≤ λ3
    Eigen::Matrix3f V;        // Eigenvectors: V = [v1, v2, v3]
    bool is_valid;            // Minimum 5 points required
};
```

---

## 2. Mathematical Foundation

### 2.1 Covariance Computation

Given L0 centroids $\{\mathbf{c}_i\}_{i=1}^{N}$ within an L1 voxel:

$$\boldsymbol{\mu} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{c}_i$$

$$\boldsymbol{\Sigma} = \frac{1}{N-1} \sum_{i=1}^{N} (\mathbf{c}_i - \boldsymbol{\mu})(\mathbf{c}_i - \boldsymbol{\mu})^T$$

### 2.2 Eigen Decomposition

$$\boldsymbol{\Sigma} = \mathbf{V} \boldsymbol{\Lambda} \mathbf{V}^T$$

where $\boldsymbol{\Lambda} = \text{diag}(\lambda_1, \lambda_2, \lambda_3)$ with $\lambda_1 \leq \lambda_2 \leq \lambda_3$.

**Important:** The eigenvectors $\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3$ are the **principal axes** of the data distribution, NOT aligned with world XYZ axes.

### 2.3 Mahalanobis Distance (Unified Residual)

For a query point $\mathbf{q}$ and Gaussian primitive $\mathcal{G} = \{\boldsymbol{\mu}, \boldsymbol{\Sigma}\}$:

$$d_M(\mathbf{q}, \mathcal{G}) = \sqrt{(\mathbf{q} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{q} - \boldsymbol{\mu})}$$

**Why this works universally:**

The information matrix $\boldsymbol{\Omega} = \boldsymbol{\Sigma}^{-1}$ automatically applies direction-dependent weighting:

- **Plane:** High weight in normal direction (small eigenvalue → large inverse)
- **Edge:** High weight perpendicular to edge (two small eigenvalues)
- **Corner:** Equal weight in all directions (isotropic)
- **Curved surface:** Anisotropic weight matching principal curvatures

### 2.4 Expanded Form

$$d_M^2 = \frac{((\mathbf{q} - \boldsymbol{\mu}) \cdot \mathbf{v}_1)^2}{\lambda_1} + \frac{((\mathbf{q} - \boldsymbol{\mu}) \cdot \mathbf{v}_2)^2}{\lambda_2} + \frac{((\mathbf{q} - \boldsymbol{\mu}) \cdot \mathbf{v}_3)^2}{\lambda_3}$$

For a plane with $\lambda_1 \ll \lambda_2, \lambda_3$:
- Error along $\mathbf{v}_1$ (normal) is heavily penalized ($1/\lambda_1$ is large)
- Error along $\mathbf{v}_2, \mathbf{v}_3$ (tangent) is weakly penalized

This **automatically reproduces point-to-plane behavior** without explicit normal computation!

---

## 3. Integration with PKO (Probabilistic Kernel Optimization)

### 3.1 Two-Stage Probabilistic Framework

**Stage 1: Geometry Uncertainty (Gaussian Primitive)**
- Covariance $\boldsymbol{\Sigma}$ encodes local surface geometry
- Information matrix $\boldsymbol{\Omega} = \boldsymbol{\Sigma}^{-1}$ provides direction-dependent weighting

**Stage 2: Measurement Uncertainty (PKO)**
- GMM fitting on Mahalanobis residuals
- Jensen-Shannon divergence minimization for optimal scale $\alpha$
- Robust weight $w_{\text{PKO}}$ for outlier rejection

### 3.2 Combined Optimization

```cpp
for (auto& correspondence : correspondences) {
    // Geometry-aware residual
    Vector3f r = transformed_point - gaussian.μ;
    Matrix3f Ω = gaussian.Σ.inverse();  // Information matrix
    float mahal_dist = sqrt(r.transpose() * Ω * r);
    
    // PKO robust weight
    double w_pko = pko_estimator->calculate_weight(mahal_dist, α);
    
    // Combined weighting: Geometry + Outlier rejection
    Matrix3f Ω_weighted = w_pko * Ω;
    
    // Gauss-Newton accumulation
    H += J.transpose() * Ω_weighted * J;
    g += J.transpose() * Ω_weighted * r;
}
```

### 3.3 Complementary Roles

| Aspect | Gaussian Primitive | PKO |
|--------|-------------------|-----|
| **Purpose** | Geometry uncertainty | Measurement uncertainty |
| **Input** | Local point distribution | Residual distribution |
| **Output** | Anisotropic weight matrix | Scalar outlier weight |
| **Handles** | Planes, edges, curves | Wrong correspondences |

---

## 4. Current Implementation Status

### 4.1 Completed ✅

**VoxelMap.h:**
```cpp
struct VoxelNode_L1 {
    // ... existing surfel data ...
    
    // Gaussian Primitive data (new!)
    bool has_gaussian = false;
    Eigen::Vector3f gaussian_mean = Eigen::Vector3f::Zero();       // μ
    Eigen::Matrix3f gaussian_covariance = Eigen::Matrix3f::Zero(); // Σ
    Eigen::Vector3f eigenvalues = Eigen::Vector3f::Zero();         // λ1, λ2, λ3
    Eigen::Matrix3f eigenvectors = Eigen::Matrix3f::Identity();    // V = [v1, v2, v3]
};
```

**VoxelMap.cpp:**
- ✅ Covariance computation from L0 centroids
- ✅ Eigen decomposition for eigenvalues/eigenvectors
- ✅ Minimum 5 points requirement
- ✅ `GetL1Gaussians()` API for visualization
- ✅ Incremental update (only recompute when child count changes)

**PangolinViewer.cpp:**
- ✅ Ellipsoid wireframe rendering
- ✅ Principal axes visualization (RGB arrows)
- ✅ Color coding by geometry type:
  - Green: Plane ($e_1 < 0.1, e_2 > 0.5$)
  - Blue: Edge ($e_1 < 0.2, e_2 < 0.3$)
  - Red: Corner ($e_1 > 0.5, e_2 > 0.5$)
  - Yellow: Curved surface (intermediate)
- ✅ UI checkbox: "Show Gaussians (Ellipsoid)"

### 4.2 Not Yet Implemented ❌

**ICP Optimizer:**
- ❌ Replace point-to-plane residual with Mahalanobis distance
- ❌ Use $\boldsymbol{\Omega} = \boldsymbol{\Sigma}^{-1}$ as information matrix in Gauss-Newton
- ❌ Combine with PKO weighting

**Correspondence Finding:**
- ❌ Use Mahalanobis distance for correspondence scoring
- ❌ Probabilistic correspondence selection

---

## 5. Next Steps (TODO)

### Phase 1: Basic Mahalanobis ICP
1. Modify `IterativeClosestPointOptimizer` to use vector residual (3D instead of scalar)
2. Replace point-to-plane with Mahalanobis distance
3. Use $\boldsymbol{\Omega} = \boldsymbol{\Sigma}^{-1}$ as information matrix
4. Benchmark against current surfel-based approach

### Phase 2: PKO Integration
1. Feed Mahalanobis distances to PKO
2. Apply PKO weight to information matrix: $\boldsymbol{\Omega}_{\text{weighted}} = w_{\text{PKO}} \cdot \boldsymbol{\Omega}$
3. Verify convergence and robustness

### Phase 3: Evaluation
1. Compare accuracy on KITTI dataset
2. Test on curved/irregular environments
3. Measure computational overhead
4. Ablation study: Gaussian only vs Gaussian + PKO

### Phase 4: Paper Writing
1. **Title idea:** "Unified Geometry Representation via Gaussian Primitives for Robust LiDAR Odometry"
2. **Main contributions:**
   - Unified framework handling all geometry types without classification
   - Automatic geometry-adaptive weighting through information matrix
   - Integration with probabilistic outlier rejection (PKO)
3. **Target venue:** IEEE RA-L, ICRA, or IROS

---

## 6. Related Work

### 6.1 Similar Concepts

| Method | Key Idea | Difference from Ours |
|--------|----------|---------------------|
| **GICP** (Segal 2009) | Point + Covariance | Per-point covariance (slow), no hierarchical structure |
| **NDT** (Biber 2003) | Voxel Gaussian | Target only (asymmetric), no PKO integration |
| **LOAM** (Zhang 2014) | Edge + Plane features | Hard classification, separate residuals |
| **Surfel-LIO** (Ours) | Precomputed surfels | Plane-only, deterministic normal |

### 6.2 Our Novelty

1. **Hierarchical voxel structure** (L0/L1) for efficient covariance computation
2. **Unified residual** (Mahalanobis) for all geometry types
3. **PKO integration** for robust estimation
4. **No hard thresholds** - continuous geometry spectrum

---

## 7. File References

| File | Purpose |
|------|---------|
| `src/database/VoxelMap.h` | L1 Gaussian primitive data structure |
| `src/database/VoxelMap.cpp` | Covariance computation, eigen decomposition |
| `src/viewer/PangolinViewer.cpp` | Ellipsoid visualization |
| `src/optimization/IterativeClosestPointOptimizer.cpp` | **TODO:** Mahalanobis ICP |
| `src/optimization/AdaptiveMEstimator.cpp` | PKO implementation |

---

## 8. Quick Start

### Build and Run
```bash
cd /home/eugene/source/lidar_odometry/build
make -j8
./kitti_lidar_odometry ../config/kitti.yaml
```

### Enable Gaussian Visualization
In the Pangolin viewer, check the box: **"Show Gaussians (Ellipsoid)"**

### Color Coding
- 🟢 **Green:** Planar surfaces
- 🔵 **Blue:** Edges/Lines
- 🔴 **Red:** Corners/Spherical
- 🟡 **Yellow:** Curved surfaces

---

## 9. Contact

**Author:** Seungwon Choi  
**Repository:** https://github.com/93won/lidar_odometry

---

*Last updated: December 21, 2025*
