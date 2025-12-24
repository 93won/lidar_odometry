# Voxel-Based Registration Methods: Mathematical Formulation and Implementation

## Overview

This document provides mathematical derivations and implementation details for voxel-based point cloud registration methods. All methods are compared under the same conditions:
- **Single voxel size** (no hierarchical structure for fair comparison)
- **Same correspondence search** (KD-tree nearest neighbor)
- **Same optimization framework** (Gauss-Newton with right perturbation on SE(3))

---

## Table of Contents

1. [Common Framework](#1-common-framework)
2. [Point-to-Plane ICP (Reference)](#2-point-to-plane-icp-reference)
3. [NDT (Normal Distributions Transform)](#3-ndt-normal-distributions-transform)
4. [VGICP (Voxelized GICP)](#4-vgicp-voxelized-gicp)
5. [D2D-NDT (Distribution-to-Distribution NDT)](#5-d2d-ndt-distribution-to-distribution-ndt)
6. [MC-ICP (Manifold-Constrained ICP) - Ours](#6-mc-icp-manifold-constrained-icp---ours)
7. [Comparison Summary](#7-comparison-summary)
8. [Skew-Symmetric Matrix Reference](#8-skew-symmetric-matrix-reference)

---

## 1. Common Framework

### 1.1 Pose Representation

We use SE(3) with **right perturbation** for optimization:

$$
T = \begin{bmatrix} R & \mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix} \in SE(3)
$$

For a small perturbation $\delta \boldsymbol{\xi} = [\delta \mathbf{t}^T, \delta \boldsymbol{\omega}^T]^T \in \mathfrak{se}(3)$:

$$
T' = T \cdot \exp(\delta \boldsymbol{\xi}^\wedge)
$$

**Expanding the right perturbation:**
$$
R' = R \cdot \exp([\delta\boldsymbol{\omega}]_\times) \approx R(I + [\delta\boldsymbol{\omega}]_\times) = R + R[\delta\boldsymbol{\omega}]_\times
$$
$$
\mathbf{t}' = \mathbf{t} + R \cdot \delta\mathbf{t}
$$

### 1.2 Point Transformation and its Jacobian

Given source point $\mathbf{p}_s$ and current transform $T = (R, \mathbf{t})$:

$$
\mathbf{p}' = R \mathbf{p}_s + \mathbf{t}
$$

**Perturbed point:**
$$
\mathbf{p}'' = R' \mathbf{p}_s + \mathbf{t}' = (R + R[\delta\boldsymbol{\omega}]_\times)\mathbf{p}_s + \mathbf{t} + R\delta\mathbf{t}
$$
$$
= R\mathbf{p}_s + R[\delta\boldsymbol{\omega}]_\times\mathbf{p}_s + \mathbf{t} + R\delta\mathbf{t}
$$

Using the identity $[\mathbf{a}]_\times \mathbf{b} = -[\mathbf{b}]_\times \mathbf{a}$:
$$
\mathbf{p}'' = \mathbf{p}' + R\delta\mathbf{t} - R[\mathbf{p}_s]_\times\delta\boldsymbol{\omega}
$$

**Jacobian of point transformation (3×6):**
$$
\frac{\partial \mathbf{p}'}{\partial \delta\boldsymbol{\xi}} = \begin{bmatrix} R & -R[\mathbf{p}_s]_\times \end{bmatrix}
$$

### 1.3 Gauss-Newton Update

The normal equation:

$$
\mathbf{H} \delta \boldsymbol{\xi} = -\mathbf{g}
$$

where:
- $\mathbf{H} = \sum_i \mathbf{J}_i^T \boldsymbol{\Omega}_i \mathbf{J}_i$ (Hessian approximation, 6×6)
- $\mathbf{g} = \sum_i \mathbf{J}_i^T \boldsymbol{\Omega}_i \mathbf{r}_i$ (gradient, 6×1)

### 1.4 Key Insight: Residual Dimension Determines Jacobian

| Method Type | Residual | Dimension | Jacobian Size |
|-------------|----------|-----------|---------------|
| Point-to-Plane | $r = \mathbf{n}^T(\mathbf{p}' - \mathbf{q})$ | scalar (1D) | **1×6** |
| Point-to-Distribution (NDT, MC-ICP) | $\mathbf{r} = \mathbf{p}' - \boldsymbol{\mu}$ | vector (3D) | **3×6** |
| Distribution-to-Distribution (VGICP, D2D) | $\mathbf{r} = \boldsymbol{\mu}'_s - \boldsymbol{\mu}_t$ | vector (3D) | **3×6** |

**중요:** 각 방법마다 residual 정의가 다르므로, Jacobian chain rule이 다르게 적용됩니다!

---

## 2. Point-to-Plane ICP (Reference)

Point-to-Plane은 가장 기본적인 방법이며, Jacobian 유도의 기준이 됩니다.

### 2.1 Cost Function

$$
E_{P2Pl}(T) = \sum_{i} \left( \mathbf{n}_i^T (\mathbf{p}'_i - \mathbf{q}_i) \right)^2
$$

where $\mathbf{n}_i$ is the normal at target point $\mathbf{q}_i$.

### 2.2 Residual (Scalar, 1D)

$$
r_i = \mathbf{n}_i^T (\mathbf{p}'_i - \mathbf{q}_i) = \mathbf{n}_i^T (R\mathbf{p}_{s,i} + \mathbf{t} - \mathbf{q}_i) \in \mathbb{R}
$$

### 2.3 Jacobian Derivation (1×6)

residual을 perturbation에 대해 미분:

$$
r' = \mathbf{n}^T (\mathbf{p}'' - \mathbf{q}) = \mathbf{n}^T \left( \mathbf{p}' + R\delta\mathbf{t} - R[\mathbf{p}_s]_\times\delta\boldsymbol{\omega} - \mathbf{q} \right)
$$
$$
= r + \mathbf{n}^T R\delta\mathbf{t} - \mathbf{n}^T R[\mathbf{p}_s]_\times\delta\boldsymbol{\omega}
$$

따라서:
$$
\frac{\partial r}{\partial \delta\mathbf{t}} = \mathbf{n}^T R, \quad \frac{\partial r}{\partial \delta\boldsymbol{\omega}} = -\mathbf{n}^T R[\mathbf{p}_s]_\times
$$

**Jacobian (1×6):**
$$
\mathbf{J}_i^{P2Pl} = \begin{bmatrix} \mathbf{n}_i^T R & -\mathbf{n}_i^T R[\mathbf{p}_{s,i}]_\times \end{bmatrix}
$$

### 2.4 Normal Equation

$\boldsymbol{\Omega} = 1$ (scalar residual)이므로:
$$
\mathbf{H} = \sum_i \mathbf{J}_i^T \mathbf{J}_i \in \mathbb{R}^{6 \times 6}
$$
$$
\mathbf{g} = \sum_i r_i \mathbf{J}_i^T \in \mathbb{R}^{6 \times 1}
$$

### 2.5 Implementation

```cpp
// Point-to-Plane
float residual = n.dot(p_transformed - p_target);  // scalar

Eigen::Matrix<float, 1, 6> J;
J.block<1, 3>(0, 0) = n.transpose() * R;           // 1×3
J.block<1, 3>(0, 3) = -n.transpose() * R * skew(p_source);  // 1×3

H += J.transpose() * J;      // 6×6
g += residual * J.transpose();  // 6×1
```

---

## 3. NDT (Normal Distributions Transform)

### 3.1 Concept

- **Source**: Raw points
- **Target**: Voxel grid, each voxel has a Gaussian distribution $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$

### 3.2 Cost Function (Mahalanobis Distance)

$$
E_{NDT}(T) = \sum_{i} (\mathbf{p}'_i - \boldsymbol{\mu}_i)^T \boldsymbol{\Sigma}_i^{-1} (\mathbf{p}'_i - \boldsymbol{\mu}_i)
$$

### 3.3 Residual (3D Vector)

$$
\mathbf{r}_i = \mathbf{p}'_i - \boldsymbol{\mu}_i = R\mathbf{p}_{s,i} + \mathbf{t} - \boldsymbol{\mu}_i \in \mathbb{R}^3
$$

### 3.4 Jacobian Derivation (3×6)

residual을 perturbation에 대해 미분:

$$
\mathbf{r}' = \mathbf{p}'' - \boldsymbol{\mu} = (\mathbf{p}' + R\delta\mathbf{t} - R[\mathbf{p}_s]_\times\delta\boldsymbol{\omega}) - \boldsymbol{\mu}
$$
$$
= \mathbf{r} + R\delta\mathbf{t} - R[\mathbf{p}_s]_\times\delta\boldsymbol{\omega}
$$

따라서:
$$
\frac{\partial \mathbf{r}}{\partial \delta\mathbf{t}} = R, \quad \frac{\partial \mathbf{r}}{\partial \delta\boldsymbol{\omega}} = -R[\mathbf{p}_s]_\times
$$

**Jacobian (3×6):**
$$
\mathbf{J}_i^{NDT} = \begin{bmatrix} R & -R[\mathbf{p}_{s,i}]_\times \end{bmatrix}
$$

### 3.5 Information Matrix

$$
\boldsymbol{\Omega}_i^{NDT} = \boldsymbol{\Sigma}_i^{-1}
$$

### 3.6 Normal Equation

$$
\mathbf{H} = \sum_i \mathbf{J}_i^T \boldsymbol{\Omega}_i \mathbf{J}_i \in \mathbb{R}^{6 \times 6}
$$
$$
\mathbf{g} = \sum_i \mathbf{J}_i^T \boldsymbol{\Omega}_i \mathbf{r}_i \in \mathbb{R}^{6 \times 1}
$$

### 3.7 Comparison with Point-to-Plane

| | Point-to-Plane | NDT |
|---|----------------|-----|
| Residual | $r = \mathbf{n}^T(\mathbf{p}' - \mathbf{q})$ (1D) | $\mathbf{r} = \mathbf{p}' - \boldsymbol{\mu}$ (3D) |
| Jacobian | $\mathbf{n}^T \cdot [R, -R[\mathbf{p}_s]_\times]$ (1×6) | $[R, -R[\mathbf{p}_s]_\times]$ (3×6) |
| Constraint | Normal direction only | All 3 directions with covariance weighting |

**NDT의 Jacobian은 Point-to-Plane에서 $\mathbf{n}^T$를 제거한 형태입니다.**

### 3.8 Implementation

```cpp
// NDT
Eigen::Vector3f residual = p_transformed - voxel_mean;  // 3D vector
Eigen::Matrix3f Omega = voxel_covariance.inverse();

Eigen::Matrix<float, 3, 6> J;
J.block<3, 3>(0, 0) = R;                      // 3×3
J.block<3, 3>(0, 3) = -R * skew(p_source);    // 3×3

H += J.transpose() * Omega * J;               // 6×6
g += J.transpose() * Omega * residual;        // 6×1
cost += residual.transpose() * Omega * residual;
```

---

## 4. VGICP (Voxelized GICP)

### 4.1 Concept

- **Source**: Voxelized, each voxel has Gaussian $\mathcal{N}(\boldsymbol{\mu}_s, \boldsymbol{\Sigma}_s)$
- **Target**: Voxelized, each voxel has Gaussian $\mathcal{N}(\boldsymbol{\mu}_t, \boldsymbol{\Sigma}_t)$

**Distribution-to-Distribution** 매칭: 두 Gaussian 분포 간의 거리를 최소화

### 4.2 Cost Function

$$
E_{VGICP}(T) = \sum_{i} \mathbf{d}_i^T \boldsymbol{\Omega}_i \mathbf{d}_i
$$

where:
- $\mathbf{d}_i = R\boldsymbol{\mu}_{s,i} + \mathbf{t} - \boldsymbol{\mu}_{t,i}$ (transformed source mean - target mean)
- $\boldsymbol{\Omega}_i = (R\boldsymbol{\Sigma}_{s,i}R^T + \boldsymbol{\Sigma}_{t,i})^{-1}$ (combined information)

### 4.3 Residual (3D Vector)

$$
\mathbf{r}_i = R\boldsymbol{\mu}_{s,i} + \mathbf{t} - \boldsymbol{\mu}_{t,i} \in \mathbb{R}^3
$$

### 4.4 Jacobian Derivation (3×6)

**핵심 차이**: Source가 voxel centroid $\boldsymbol{\mu}_s$이므로, Jacobian에서 $\mathbf{p}_s$ 대신 $\boldsymbol{\mu}_s$를 사용:

$$
\mathbf{r}' = R'\boldsymbol{\mu}_s + \mathbf{t}' - \boldsymbol{\mu}_t
$$

Right perturbation 적용:
$$
\mathbf{r}' = (R + R[\delta\boldsymbol{\omega}]_\times)\boldsymbol{\mu}_s + \mathbf{t} + R\delta\mathbf{t} - \boldsymbol{\mu}_t
$$
$$
= \mathbf{r} + R\delta\mathbf{t} - R[\boldsymbol{\mu}_s]_\times\delta\boldsymbol{\omega}
$$

**Jacobian (3×6):**
$$
\mathbf{J}_i^{VGICP} = \begin{bmatrix} R & -R[\boldsymbol{\mu}_{s,i}]_\times \end{bmatrix}
$$

### 4.5 Information Matrix (R-dependent!)

$$
\boldsymbol{\Omega}_i = (R\boldsymbol{\Sigma}_{s,i}R^T + \boldsymbol{\Sigma}_{t,i})^{-1}
$$

**중요**: $\boldsymbol{\Omega}$가 $R$에 의존하므로 매 iteration마다 재계산해야 합니다. 
Gauss-Newton에서는 이를 상수로 취급하고 iteration 내에서만 고정합니다.

### 4.6 Implementation

```cpp
// VGICP
Eigen::Vector3f mu_s = source_voxel_mean;  // Source voxel centroid
Eigen::Vector3f mu_t = target_voxel_mean;  // Target voxel centroid

Eigen::Vector3f residual = R * mu_s + t - mu_t;  // 3D vector

// Combined covariance (R-dependent!)
Eigen::Matrix3f C_combined = R * source_cov * R.transpose() + target_cov;
Eigen::Matrix3f Omega = C_combined.inverse();

// Jacobian: use source voxel mean, not raw point!
Eigen::Matrix<float, 3, 6> J;
J.block<3, 3>(0, 0) = R;                     // 3×3
J.block<3, 3>(0, 3) = -R * skew(mu_s);       // 3×3 (mu_s, not p_source!)

H += J.transpose() * Omega * J;              // 6×6
g += J.transpose() * Omega * residual;       // 6×1
cost += residual.transpose() * Omega * residual;
```

---

## 5. D2D-NDT (Distribution-to-Distribution NDT)

### 5.1 Concept

- **Source**: Voxelized Gaussians $\mathcal{N}(\boldsymbol{\mu}_s, \boldsymbol{\Sigma}_s)$
- **Target**: Voxelized Gaussians $\mathcal{N}(\boldsymbol{\mu}_t, \boldsymbol{\Sigma}_t)$

**VGICP와 수학적으로 동일**하지만, correspondence 방식이 다릅니다:
- VGICP: nearest neighbor로 correspondence
- D2D-NDT: voxel grid 기반 correspondence (같은 voxel 위치)

### 5.2 Cost Function (L2 Distance between Gaussians)

$$
E_{D2D}(T) = \sum_{i} (\boldsymbol{\mu}'_{s,i} - \boldsymbol{\mu}_{t,i})^T (\boldsymbol{\Sigma}'_{s,i} + \boldsymbol{\Sigma}_{t,i})^{-1} (\boldsymbol{\mu}'_{s,i} - \boldsymbol{\mu}_{t,i})
$$

where $\boldsymbol{\mu}'_s = R\boldsymbol{\mu}_s + \mathbf{t}$ and $\boldsymbol{\Sigma}'_s = R\boldsymbol{\Sigma}_s R^T$.

### 5.3 Jacobian (3×6)

VGICP와 동일:
$$
\mathbf{J}_i^{D2D} = \begin{bmatrix} R & -R[\boldsymbol{\mu}_{s,i}]_\times \end{bmatrix}
$$

### 5.4 Implementation

```cpp
// D2D-NDT (same math as VGICP, different correspondence)
Eigen::Vector3f residual = R * source_voxel_mean + t - target_voxel_mean;
Eigen::Matrix3f C_combined = R * source_voxel_cov * R.transpose() + target_voxel_cov;
Eigen::Matrix3f Omega = C_combined.inverse();

Eigen::Matrix<float, 3, 6> J;
J.block<3, 3>(0, 0) = R;
J.block<3, 3>(0, 3) = -R * skew(source_voxel_mean);

H += J.transpose() * Omega * J;
g += J.transpose() * Omega * residual;
```

---

## 6. MC-ICP (Manifold-Constrained ICP) - Ours

### 6.1 Concept

- **Source**: Raw points
- **Target**: Voxel Gaussians with **geometry-aware information matrix**

**Point-to-Distribution** 매칭이지만, information matrix에 geometry-aware weighting 적용

### 6.2 Geometry-Aware Information Matrix

Given voxel covariance $\boldsymbol{\Sigma}$ with eigendecomposition:

$$
\boldsymbol{\Sigma} = \mathbf{V} \text{diag}(\lambda_1, \lambda_2, \lambda_3) \mathbf{V}^T
$$

**NDT** uses:
$$
\boldsymbol{\Omega}^{NDT} = \boldsymbol{\Sigma}^{-1} = \mathbf{V} \text{diag}\left(\frac{1}{\lambda_1}, \frac{1}{\lambda_2}, \frac{1}{\lambda_3}\right) \mathbf{V}^T
$$

**MC-ICP** uses:
$$
\boldsymbol{\Omega}^{MC} = \mathbf{V} \text{diag}\left(\frac{c_1}{\lambda_1}, \frac{c_2}{\lambda_2}, \frac{c_3}{\lambda_3}\right) \mathbf{V}^T
$$

where the **constraint strength** is:
$$
c_i = 1 - \frac{\lambda_i}{\text{tr}(\boldsymbol{\Sigma})} = 1 - \frac{\lambda_i}{\lambda_1 + \lambda_2 + \lambda_3}
$$

### 6.3 Intuition

| Geometry | Eigenvalues | Constraint Strength |
|----------|-------------|---------------------|
| **Plane** | $\lambda_1 \ll \lambda_2 \approx \lambda_3$ | $c_1 \approx 1$ (normal direction strong) |
| **Edge** | $\lambda_1 \approx \lambda_2 \ll \lambda_3$ | $c_1, c_2 \approx 1$, $c_3 \approx 0$ |
| **Sphere** | $\lambda_1 \approx \lambda_2 \approx \lambda_3$ | $c_1 \approx c_2 \approx c_3 \approx 0.67$ |

### 6.4 Residual (3D Vector)

NDT와 동일:
$$
\mathbf{r}_i = \mathbf{p}'_i - \boldsymbol{\mu}_i = R\mathbf{p}_{s,i} + \mathbf{t} - \boldsymbol{\mu}_i \in \mathbb{R}^3
$$

### 6.5 Jacobian Derivation (3×6)

NDT와 동일:
$$
\mathbf{J}_i^{MC} = \begin{bmatrix} R & -R[\mathbf{p}_{s,i}]_\times \end{bmatrix}
$$

**핵심 차이점**: Jacobian은 같지만, **Information matrix가 다릅니다!**

### 6.6 Combined Information (Source + Target)

Source와 target 모두에서 geometry-aware information을 결합:

$$
\boldsymbol{\Sigma}_{combined} = R \boldsymbol{\Sigma}_s^{MC} R^T + \boldsymbol{\Sigma}_t^{MC}
$$
$$
\boldsymbol{\Omega}_{combined}^{MC} = \boldsymbol{\Sigma}_{combined}^{-1}
$$

where $\boldsymbol{\Sigma}^{MC} = (\boldsymbol{\Omega}^{MC})^{-1}$.

### 6.7 Implementation

```cpp
// Geometry-aware information matrix
Eigen::Matrix3f computeGeometryAwareOmega(
    const Eigen::Vector3f& eigenvalues,
    const Eigen::Matrix3f& eigenvectors)
{
    float trace = eigenvalues.sum();
    
    // Constraint strength: c_i = 1 - λ_i / tr(Σ)
    Eigen::Vector3f c;
    c[0] = 1.0f - eigenvalues[0] / trace;
    c[1] = 1.0f - eigenvalues[1] / trace;
    c[2] = 1.0f - eigenvalues[2] / trace;
    
    // Information weight: c_i / λ_i
    const float eps = 1e-4f * trace;
    Eigen::Vector3f info;
    info[0] = c[0] / (eigenvalues[0] + eps);
    info[1] = c[1] / (eigenvalues[1] + eps);
    info[2] = c[2] / (eigenvalues[2] + eps);
    
    // Ω = V * diag(info) * V^T
    return eigenvectors * info.asDiagonal() * eigenvectors.transpose();
}

// MC-ICP main loop
Eigen::Vector3f residual = p_transformed - target_mean;  // 3D vector

// Geometry-aware information matrices
Eigen::Matrix3f Omega_s = computeGeometryAwareOmega(source_eigenvalues, source_eigenvectors);
Eigen::Matrix3f Omega_t = computeGeometryAwareOmega(target_eigenvalues, target_eigenvectors);

// Combine (covariance addition, then invert)
Eigen::Matrix3f Cov_s = Omega_s.inverse();
Eigen::Matrix3f Cov_t = Omega_t.inverse();
Eigen::Matrix3f Cov_combined = R * Cov_s * R.transpose() + Cov_t;
Eigen::Matrix3f Omega = Cov_combined.inverse();

// Jacobian: raw source point (not voxel mean)
Eigen::Matrix<float, 3, 6> J;
J.block<3, 3>(0, 0) = R;                      // 3×3
J.block<3, 3>(0, 3) = -R * skew(p_source);    // 3×3

H += J.transpose() * Omega * J;               // 6×6
g += J.transpose() * Omega * residual;        // 6×1
```

---

---

## 7. Comparison Summary

### 7.1 Jacobian Comparison

| Method | Residual | Jacobian | Size | Point in Jacobian |
|--------|----------|----------|------|-------------------|
| **Point-to-Plane** | $r = \mathbf{n}^T(\mathbf{p}' - \mathbf{q})$ (scalar) | $[\mathbf{n}^T R, -\mathbf{n}^T R[\mathbf{p}_s]_\times]$ | **1×6** | source point $\mathbf{p}_s$ |
| **NDT** | $\mathbf{r} = \mathbf{p}' - \boldsymbol{\mu}$ (3D) | $[R, -R[\mathbf{p}_s]_\times]$ | **3×6** | source point $\mathbf{p}_s$ |
| **VGICP** | $\mathbf{r} = R\boldsymbol{\mu}_s + \mathbf{t} - \boldsymbol{\mu}_t$ (3D) | $[R, -R[\boldsymbol{\mu}_s]_\times]$ | **3×6** | source voxel mean $\boldsymbol{\mu}_s$ |
| **D2D-NDT** | $\mathbf{r} = R\boldsymbol{\mu}_s + \mathbf{t} - \boldsymbol{\mu}_t$ (3D) | $[R, -R[\boldsymbol{\mu}_s]_\times]$ | **3×6** | source voxel mean $\boldsymbol{\mu}_s$ |
| **MC-ICP** | $\mathbf{r} = \mathbf{p}' - \boldsymbol{\mu}$ (3D) | $[R, -R[\mathbf{p}_s]_\times]$ | **3×6** | source point $\mathbf{p}_s$ |

### 7.2 Information Matrix Comparison

| Method | Source | Target | Information Matrix $\boldsymbol{\Omega}$ |
|--------|--------|--------|------------------------------------------|
| **Point-to-Plane** | point | point | $\boldsymbol{\Omega} = 1$ (scalar) |
| **NDT** | point | voxel Gaussian | $\boldsymbol{\Sigma}_t^{-1}$ |
| **VGICP** | voxel Gaussian | voxel Gaussian | $(R\boldsymbol{\Sigma}_sR^T + \boldsymbol{\Sigma}_t)^{-1}$ |
| **D2D-NDT** | voxel Gaussian | voxel Gaussian | $(R\boldsymbol{\Sigma}_sR^T + \boldsymbol{\Sigma}_t)^{-1}$ |
| **MC-ICP** | point | voxel Gaussian | $\mathbf{V}\text{diag}(\frac{c_i}{\lambda_i})\mathbf{V}^T$ |

### 7.3 Key Differences

| | Point-to-Plane | NDT | VGICP/D2D | MC-ICP |
|---|----------------|-----|-----------|--------|
| Source representation | point | point | **voxel** | point |
| Target representation | point | voxel | **voxel** | voxel |
| Residual dimension | **1D** | 3D | 3D | 3D |
| Geometry weighting | normal only | covariance | covariance | **geometry-aware** |

### 7.4 MC-ICP vs NDT (핵심 차이)

Both use **point → voxel Gaussian** matching with 3D residual:

| | NDT | MC-ICP |
|---|-----|--------|
| Information | $\frac{1}{\lambda_i}$ | $\frac{c_i}{\lambda_i}$ where $c_i = 1 - \frac{\lambda_i}{\text{tr}(\boldsymbol{\Sigma})}$ |
| Effect | High weight for small variance | **Down-weights ambiguous directions** |

The constraint strength $c_i$ removes **correspondence ambiguity** in directions where the Gaussian has high variance.

---

## 8. Skew-Symmetric Matrix Reference

$$
[\mathbf{p}]_\times = \begin{bmatrix} 0 & -p_z & p_y \\ p_z & 0 & -p_x \\ -p_y & p_x & 0 \end{bmatrix}
$$

Property: $[\mathbf{a}]_\times \mathbf{b} = \mathbf{a} \times \mathbf{b} = -[\mathbf{b}]_\times \mathbf{a}$

---

*Last updated: 2025-12-23*
