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

---

### 6.2 Detailed Comparison

#### 6.2.1 GICP (Generalized ICP) - Segal et al., 2009

**GICP Approach:**
```cpp
// GICP: Compute covariance for each point via k-NN
for each point p:
    neighbors = kNN(p, k=20)
    Σ_p = Cov(neighbors)  // Measurement uncertainty modeling

// Optimization: Point-to-Point with combined covariance
for each correspondence (p_source, p_target):
    r = p_source - p_target  // Point-to-point residual
    C = Σ_source + R*Σ_target*R^T  // Combined uncertainty
    Ω = C^{-1}
    cost += r^T Ω r
```

**Intent of GICP Covariance:**
- "How much **measurement uncertainty** does this point have?"
- Models sensor noise: accurate in normal direction, uncertain in tangent direction
- **Philosophy:** "Trust measurements more in directions with lower uncertainty"

**Mathematical Details:**

| Component | GICP | Ours |
|-----------|------|------|
| **Correspondence** | Point → **Point** | Point → **Gaussian** |
| **Target representation** | Individual point $\mathbf{p}_t$ | Gaussian mean $\boldsymbol{\mu}$ |
| **Residual** | $\mathbf{r} = \mathbf{p}_s - \mathbf{p}_t$ | $\mathbf{r} = \mathbf{p}_s - \boldsymbol{\mu}$ |
| **Information matrix** | $\boldsymbol{\Omega} = (\boldsymbol{\Sigma}_s + R\boldsymbol{\Sigma}_t R^T)^{-1}$ | $\boldsymbol{\Omega} = \boldsymbol{\Sigma}^{-1}$ |
| **Source covariance** | ✅ Used | ❌ Not used (L0 centroid already denoised) |
| **Target covariance** | Per-point (k-NN) | Per-voxel (L0 centroids) |

**Covariance Interpretation:**

```
GICP (plane): Σ = [0.01, 0, 0; 0, 1.0, 0; 0, 0, 1.0]
  → "Measurement noise is small in normal direction (±1cm)"
  → "Measurement noise is large in tangent direction (±100cm)"
  → Covariance represents sensor noise characteristics

Ours (plane): Σ = [0.01, 0, 0; 0, 1.0, 0; 0, 0, 1.0]  
  → "Points are distributed flat (plane geometry)"
  → "Variation is small in normal direction, large in tangent"
  → Covariance represents pure geometry shape

→ Same eigenvalues, different philosophical meaning!
```

**Key Differences from Ours:**

| Aspect | GICP | Gaussian Primitive (Ours) |
|--------|------|---------------------------|
| **Covariance meaning** | Measurement uncertainty (noise) | **Geometry shape** itself |
| **Philosophy** | "Trust reliable measurements" | "Match geometry structure" |
| **Computation target** | Raw points (noisy) | **L0 centroids** (denoised) |
| **Computation location** | Per-point (tens of thousands) | **Per-L1 voxel** (thousands) |
| **Computation timing** | Real-time per frame | **Precomputed** (incremental) |
| **Hierarchy** | None | L0 → L1 two-stage denoising |
| **Complexity** | $O(N \cdot k)$ per frame | $O(V)$ lookup only |
| **Combined covariance** | $\Sigma_s + \Sigma_t$ (both uncertainties) | Only $\Sigma_t$ (geometry only) |

**Why Combined Covariance in GICP:**
```cpp
// GICP combines both source and target uncertainty
Matrix3f C_combined = Sigma_source + R * Sigma_target * R.transpose();

// Example: Both measurements on a plane
Σ_src = [0.01, 0, 0; 0, 1, 0; 0, 0, 1]  // ±1cm normal, ±100cm tangent
Σ_tgt = [0.01, 0, 0; 0, 1, 0; 0, 0, 1]
Combined = [0.02, 0, 0; 0, 2, 0; 0, 0, 2]  // Uncertainties add up!

// Interpretation: "Both measurements are uncertain, combine uncertainties"
```

**Why We Don't Use Source Covariance:**
```cpp
// Ours: Source is L0 centroid (already averaged → low noise)
//       Target is L1 Gaussian (represents geometry)
Matrix3f Omega = gaussian.covariance.inverse();  // Only target geometry

// We could theoretically compute source covariance:
// Σ_source = Cov(points in L0 voxel)
// But it's unnecessary because:
// 1. L0 centroid already averaged (noise reduced)
// 2. Target geometry dominates the matching constraint
// 3. Simpler and faster
```

**Limitations of GICP:**
1. **Speed:** Point-wise covariance = $O(N \cdot k)$ (very slow)
2. **Noise sensitive:** Raw point based → k-NN affected by noise
3. **k selection problem:** Small k = unstable, large k = over-smoothing
4. **No robust estimation:** Vulnerable to outliers
5. **Point-to-point:** Target is individual point, not geometric primitive

---

#### 6.2.2 NDT (Normal Distributions Transform) - Biber & Straßer, 2003

**NDT Approach:**
```cpp
// NDT: Generate Gaussian from points in voxel (similar to ours!)
for each voxel:
    μ = mean(points in voxel)
    Σ = Cov(points in voxel)

// Optimization: Fit source points to target Gaussians
for each source_point p:
    find nearest target voxel
    score += exp(-0.5 * (p - μ)^T Σ^{-1} (p - μ))  // Gaussian likelihood
```

**Similarities with Ours:**
- ✅ Voxel-based Gaussian
- ✅ Covariance represents geometry
- ✅ Uses Mahalanobis-like distance
- ✅ Point-to-distribution matching

**Mathematical Details:**

| Component | NDT | Ours |
|-----------|-----|------|
| **Correspondence** | Point → **Gaussian** | Point → **Gaussian** |
| **Target representation** | Voxel Gaussian $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ | L1 Gaussian $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ |
| **Residual** | $\mathbf{r} = \mathbf{p}_s - \boldsymbol{\mu}$ | $\mathbf{r} = \mathbf{p}_s - \boldsymbol{\mu}$ |
| **Objective** | $\max \sum e^{-\frac{1}{2}\mathbf{r}^T\boldsymbol{\Omega}\mathbf{r}}$ | $\min \sum \mathbf{r}^T\boldsymbol{\Omega}\mathbf{r}$ |
| **Relationship** | $-\log(\text{NDT}) = \text{Ours}$ | $e^{-\text{Ours}} = \text{NDT}$ |
| **Optimization** | Newton on score function | Gauss-Newton on residual |

**Key Insight: Same Core, Different Wrapper**

```
NDT objective:    S = Σ exp(-½ dᵀΩd)     ← Maximize probability
Take -log:        -log(S) = Σ ½ dᵀΩd     ← Minimize negative log-likelihood
Our objective:    E = Σ dᵀΩd             ← Minimize squared Mahalanobis distance

→ Essentially the same optimization problem!
```

**Key Differences from Ours:**

| Aspect | NDT | Gaussian Primitive (Ours) |
|--------|-----|---------------------------|
| **Symmetry** | **Asymmetric** (target only Gaussian) | Symmetric (both can be Gaussian) |
| **Source handling** | Raw points as-is | **L0 centroid** (denoised) |
| **Hierarchy** | Single-level voxel | **L0 → L1** two-stage |
| **Noise filtering** | None (raw points) | **Filtered via L0 averaging** |
| **Robust estimation** | None | **PKO integration** |
| **Optimization method** | Newton (score maximization) | **Gauss-Newton** (residual minimization) |
| **Objective function** | Exponential (probabilistic) | Quadratic (geometric) |
| **Hessian** | Analytical (exact) | $J^T\Omega J$ (Gauss-Newton approx) |

**NDT's Asymmetry Problem:**
```
NDT:
  Source: raw points (noisy)              ● ● ●  ← Sensor noise
  Target: Gaussian (smoothed)                ⬭   ← Averaged
  → Source noise directly affects optimization

Ours:
  Source: L0 centroids (averaged)         ◆ ◆ ◆  ← Already denoised
  Target: L1 Gaussian (from L0 centroids)    ⬭   ← Computed from ◆◆◆
  → Both sides denoised, symmetric
```

**Optimization Method Comparison:**

```cpp
// NDT: Newton's method on score function
double score = 0;
Vector6f gradient = Vector6f::Zero();
Matrix6f hessian = Matrix6f::Zero();

for (auto& point : source_cloud) {
    auto& gaussian = findNearestVoxel(point);
    Vector3f d = point - gaussian.mean;
    Matrix3f Omega = gaussian.cov.inverse();
    
    // Probability term
    double exp_term = exp(-0.5 * d.transpose() * Omega * d);
    score += exp_term;
    
    // Analytical gradient: ∂/∂x[exp(-½dᵀΩd)] = exp(...)×(-Ωd)×∂d/∂x
    Matrix<double,3,6> J = computeJacobian(point);
    gradient += exp_term * J.transpose() * (-Omega * d);
    
    // Analytical Hessian: ∂²/∂x²[exp(-½dᵀΩd)] = exp(...)×[complex terms]
    Matrix3f H_term = -Omega + Omega * d * d.transpose() * Omega;
    hessian += exp_term * J.transpose() * H_term * J;
}

// Newton's method (exact Hessian!)
delta_x = hessian.ldlt().solve(gradient);


// Ours: Gauss-Newton on residual
Matrix6f H = Matrix6f::Zero();
Vector6f g = Vector6f::Zero();

for (auto& correspondence : correspondences) {
    Vector3f d = src_point - gaussian.mean;
    Matrix3f Omega = gaussian.cov.inverse();
    
    // Simple Jacobian of residual
    Matrix<double,3,6> J = computeJacobian(src_point);
    
    // Gauss-Newton (approximate Hessian)
    H += J.transpose() * Omega * J;
    g += J.transpose() * Omega * d;
}

// Gauss-Newton
delta_x = H.ldlt().solve(-g);
```

**Why Exponential in NDT:**

NDT views the problem as **probabilistic matching**:
- Each voxel defines probability distribution $P(\mathbf{x}) = \frac{1}{Z} e^{-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})}$
- Goal: Maximize likelihood $\prod P(\mathbf{p}_i)$ = Maximize $\sum \log P(\mathbf{p}_i)$
- Leads to exponential terms in gradient/Hessian

**Our View:**

We view it as **geometric error minimization**:
- Mahalanobis distance = geometry-aware metric
- Goal: Minimize $\sum d_M^2(\mathbf{p}_i, \mathcal{G})$
- Standard least-squares framework

**Practical Differences:**

```
NDT:
  + Exponential naturally down-weights outliers
  + Exact Hessian (better convergence near optimum)
  - More complex gradient/Hessian computation
  - Slower per iteration

Ours:
  + Simpler computation (standard Gauss-Newton)
  + Faster per iteration
  + Can add PKO for explicit outlier handling
  - Approximate Hessian (Gauss-Newton)
```

**Limitations of NDT:**
1. **Asymmetric:** Source is raw point, only target is Gaussian → information imbalance
2. **Single level:** No hierarchical denoising
3. **Score function:** More complex derivatives, slower computation
4. **No outlier handling:** Dynamic objects can corrupt the score
5. **Raw source points:** Sensor noise in source directly affects matching

**Key Differences from Ours:**

| Aspect | NDT | Gaussian Primitive (Ours) |
|--------|-----|---------------------------|
| **Symmetry** | **Asymmetric** (target only Gaussian) | Symmetric (both can be Gaussian) |
| **Source handling** | Raw points as-is | L0 centroid (denoised) |
| **Hierarchy** | Single-level voxel | **L0 → L1** two-stage |
| **Noise filtering** | None (raw points) | **Filtered via L0 averaging** |
| **Robust estimation** | None | **PKO integration** |
| **Optimization method** | Score function maximization | **Gauss-Newton with information matrix** |

**NDT's Asymmetry Problem:**
```
NDT:
  Source: raw points (noisy)
  Target: Gaussian (smoothed)
  → Source noise directly affects optimization

Ours:
  Source: L0 centroids (already averaged)
  Target: L1 Gaussian (computed from L0 centroids)
  → Both sides denoised, symmetric
```

**Limitations of NDT:**
1. **Asymmetric:** Source is raw point, only target is Gaussian → information imbalance
2. **Single level:** No hierarchical denoising
3. **Score function:** Score maximization instead of Gauss-Newton → less precise
4. **No outlier handling:** Vulnerable to dynamic objects

---

#### 6.2.3 LOAM (Lidar Odometry and Mapping) - Zhang & Singh, 2014

**LOAM Approach:**
```cpp
// LOAM: Hard feature classification
for each point p:
    curvature = compute_curvature(p)
    
    if (curvature > threshold_edge):
        edge_features.push_back(p)      // Edge point
    else if (curvature < threshold_plane):
        plane_features.push_back(p)     // Plane point

// Different residual functions
edge_residual = point_to_line_distance(p, nearest_edge)
plane_residual = point_to_plane_distance(p, nearest_plane)
```

**Key Differences from Ours:**

| Aspect | LOAM | Gaussian Primitive (Ours) |
|--------|------|---------------------------|
| **Geometry representation** | **Hard classification** (edge/plane) | **Continuous spectrum** (covariance) |
| **Residual** | Separate functions (2 types) | **Unified Mahalanobis** |
| **Curved surface handling** | ❌ Cannot handle (neither edge nor plane) | ✅ **Automatic handling** |
| **Threshold** | Manual tuning required | **No threshold** |
| **Code complexity** | 2 code paths | 1 unified path |

**Limitations of LOAM:**
1. **Hard threshold:** Forced classification "Is this edge or plane?"
2. **Cannot handle curves:** Curved surfaces are neither edge nor plane → discarded
3. **Discontinuity:** Optimization landscape discontinuous at threshold boundaries
4. **Information loss:** Rich information in eigenvalues, but only normal is used

---

#### 6.2.4 Summary: Our Contributions

```
                    Geometry        Hierarchical    Robust        Unified
Method              Representation  Denoising       Estimation    Residual
─────────────────────────────────────────────────────────────────────────
GICP                Noise model     ❌              ❌            ✅
NDT                 Voxel Gaussian  ❌              ❌            ✅
LOAM                Hard classify   ❌              ❌            ❌
─────────────────────────────────────────────────────────────────────────
Ours                Geometry repr   ✅ (L0→L1)      ✅ (PKO)      ✅
```

**Key Contributions:**
1. ✅ **Hierarchical Denoising:** L0 averaging → L1 covariance (sensor noise removal)
2. ✅ **Explicit Geometry Representation:** Covariance = geometry shape (not noise)
3. ✅ **PKO Integration:** Mahalanobis residual + GMM outlier rejection
4. ✅ **Unified Residual:** Plane, edge, corner, curved surface all in one formula
5. ✅ **Precomputed:** L1 Gaussians computed incrementally, per-frame is lookup only

---

### 6.3 Our Novelty

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
