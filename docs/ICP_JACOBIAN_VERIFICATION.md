# ICP Methods: Mathematical Formulation and Jacobian Verification

## Notation

- $p_i \in \mathbb{R}^3$: source point in its **local (body) frame**
- $q_i \in \mathbb{R}^3$: target point in **world frame**
- $T = [R \mid t] \in SE(3)$: transformation (R ∈ SO(3), t ∈ ℝ³)
- $\tilde{p}_i = Rp_i + t$: transformed source point in world frame
- $\delta\xi = [\delta t; \delta\omega] \in \mathbb{R}^6$: pose perturbation (translation + rotation)

## SE(3) Right Perturbation (Lie Group Optimization)

**Convention:** $T' = T \oplus \delta\xi = T \cdot \exp(\hat{\delta\xi})$

### Derivation

Homogeneous transformation matrix:
$$T = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix}$$

For small perturbations, the exponential map:
$$\exp(\hat{\delta\xi}) \approx \begin{bmatrix} I + [\delta\omega]_\times & \delta t \\ 0 & 1 \end{bmatrix}$$

Matrix multiplication (right perturbation):
$$T' = T \cdot \exp(\hat{\delta\xi}) = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix} \begin{bmatrix} I + [\delta\omega]_\times & \delta t \\ 0 & 1 \end{bmatrix}$$

$$= \begin{bmatrix} R(I + [\delta\omega]_\times) & R\delta t + t \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} R + R[\delta\omega]_\times & t + R\delta t \\ 0 & 1 \end{bmatrix}$$

**Transformed point under perturbation:**
$$\tilde{p}' = R'p + t' = (R + R[\delta\omega]_\times)p + (t + R\delta t)$$
$$= Rp + t + R[\delta\omega]_\times p + R\delta t = \tilde{p} + R\delta t + R[\delta\omega]_\times p$$

Using the skew-symmetric property $[\delta\omega]_\times p = -[p]_\times \delta\omega$:
$$\boxed{\tilde{p}' = \tilde{p} + R\delta t - R[p]_\times \delta\omega}$$

**Jacobian of transformed point w.r.t. perturbation:**
$$\boxed{\frac{\partial \tilde{p}'}{\partial \delta\xi} = \begin{bmatrix} \frac{\partial \tilde{p}'}{\partial \delta t} & \frac{\partial \tilde{p}'}{\partial \delta\omega} \end{bmatrix} = \begin{bmatrix} R & -R[p]_\times \end{bmatrix} \in \mathbb{R}^{3 \times 6}}$$

**⚠️ Critical:** $[p]_\times$ uses the **original source point $p$** in local frame, NOT the transformed point!

---

## 1. Point-to-Point ICP

**Residual:** $r_i = \|\tilde{p}_i - q_i\|^2$

**Cost:** $E = \sum_i \|Rp_i + t - q_i\|^2$

**Optimization:** Closed-form SVD solution (Umeyama/Horn algorithm)
1. Compute centroids: $\bar{p} = \frac{1}{N}\sum_i p_i$, $\bar{q} = \frac{1}{N}\sum_i q_i$
2. Build cross-covariance: $H = \sum_i (p_i - \bar{p})(q_i - \bar{q})^T$
3. SVD: $H = U\Sigma V^T$
4. Rotation: $R^* = VU^T$ (if $\det(R^*) = -1$, negate last column of $V$)
5. Translation: $t^* = \bar{q} - R^*\bar{p}$

**Why SVD works:** The cost is quadratic in R and t, with a unique global minimum.

**Status:** ✅ Correct (closed-form, no iterative Jacobian needed)

---

## 2. Point-to-Plane ICP

**Residual (scalar):** $r_i = n_i^T(\tilde{p}_i - q_i)$ where $n_i$ is the normal at target point $q_i$

**Cost:** $E = \sum_i [n_i^T(Rp_i + t - q_i)]^2$

**Jacobian derivation:**
$$\frac{\partial r_i}{\partial \delta\xi} = n_i^T \cdot \frac{\partial \tilde{p}_i'}{\partial \delta\xi} = n_i^T \begin{bmatrix} R & -R[p_i]_\times \end{bmatrix}$$

$$\boxed{J_i = \begin{bmatrix} n_i^T R & -n_i^T R[p_i]_\times \end{bmatrix} \in \mathbb{R}^{1 \times 6}}$$

**Gauss-Newton:**
$$H = \sum_i J_i^T J_i, \quad g = \sum_i r_i J_i^T, \quad \delta\xi^* = -H^{-1}g$$

**Implementation:**
```cpp
J.block<1, 3>(0, 0) = n.transpose() * R;           // ✅
J.block<1, 3>(0, 3) = -n.transpose() * R * p_skew; // ✅ (p_skew = [p_source]_×)
```

**Status:** ✅ Correct

---

## 3. Symmetric ICP

**Residual (scalar):** $r_i = n_{sym,i}^T(\tilde{p}_i - q_i)$

where $n_{sym,i} = R \cdot n_{p,i} + n_{q,i}$ (WITHOUT normalization)
- $n_{p,i}$: normal at source point (in local frame)
- $n_{q,i}$: normal at target point (in world frame)

**Cost:** $E = \sum_i [(R n_{p,i} + n_{q,i})^T(Rp_i + t - q_i)]^2$

**Approximation:** Treat $n_{sym,i}$ as constant within each Gauss-Newton iteration.

**Jacobian (under approximation):**
$$\frac{\partial r_i}{\partial \delta\xi} \approx n_{sym,i}^T \cdot \frac{\partial \tilde{p}_i'}{\partial \delta\xi}$$

$$\boxed{J_i \approx \begin{bmatrix} n_{sym,i}^T R & -n_{sym,i}^T R[p_i]_\times \end{bmatrix} \in \mathbb{R}^{1 \times 6}}$$

**Implementation:**
```cpp
Eigen::Vector3f n_sym = R * n_source + n_target;       // ✅
J.block<1, 3>(0, 0) = n_sym.transpose() * R;           // ✅
J.block<1, 3>(0, 3) = -n_sym.transpose() * R * p_skew; // ✅
```

**Status:** ✅ Correct (with linearization approximation)

---

## 4. GICP (Generalized ICP)

**Residual (3D vector):** $d_i = \tilde{p}_i - q_i = Rp_i + t - q_i$

**Cost (Mahalanobis distance):**
$$E = \sum_i d_i^T \Omega_i d_i$$

where $\Omega_i = (\Sigma_{q,i} + R \Sigma_{p,i} R^T)^{-1}$ is the information matrix
- $\Sigma_{p,i}$: covariance at source point
- $\Sigma_{q,i}$: covariance at target point

**Jacobian:**
$$\frac{\partial d_i}{\partial \delta\xi} = \frac{\partial \tilde{p}_i'}{\partial \delta\xi} = \begin{bmatrix} R & -R[p_i]_\times \end{bmatrix}$$

$$\boxed{J_i = \begin{bmatrix} R & -R[p_i]_\times \end{bmatrix} \in \mathbb{R}^{3 \times 6}}$$

**Gauss-Newton (weighted):**
$$H = \sum_i J_i^T \Omega_i J_i, \quad g = \sum_i J_i^T \Omega_i d_i, \quad \delta\xi^* = -H^{-1}g$$

**Implementation:**
```cpp
J.block<3, 3>(0, 0) = R;           // ✅
J.block<3, 3>(0, 3) = -R * p_skew; // ✅ (p_skew = [p_source]_×)
H += J.transpose() * Omega * J;    // ✅
g += J.transpose() * Omega * d;    // ✅
```

**Status:** ✅ Correct

---

## 5. MC-ICP (Manifold-Constrained ICP)

**Residual (3D vector):** $d_i = \tilde{p}_i - q_i = Rp_i + t - q_i$

**Cost (Geometry-aware Mahalanobis distance):**
$$E = \sum_i d_i^T \Omega_{geo,i} d_i$$

where $\Omega_{geo,i}$ is the **geometry-aware information matrix**:
$$\Omega_{geo,i} = V_i \cdot \text{diag}\left(\frac{c_1}{\lambda_1}, \frac{c_2}{\lambda_2}, \frac{c_3}{\lambda_3}\right) \cdot V_i^T$$

- $\Sigma_i = V_i \cdot \text{diag}(\lambda_1, \lambda_2, \lambda_3) \cdot V_i^T$: eigendecomposition
- $c_j = 1 - \frac{\lambda_j}{\text{tr}(\Sigma_i)}$: constraint strength ($0 \le c_j \le 1$)

**Geometry Intuition:**
- **Plane:** $\lambda_3 \ll \lambda_2 \approx \lambda_1$ → strong constraint in normal direction ($c_3 \approx 1$)
- **Line:** $\lambda_2, \lambda_3 \ll \lambda_1$ → strong constraint perpendicular to line
- **Scatter:** $\lambda_1 \approx \lambda_2 \approx \lambda_3$ → weak constraint (all $c_j \approx 0$)

**Jacobian:** Same as GICP
$$\boxed{J_i = \begin{bmatrix} R & -R[p_i]_\times \end{bmatrix} \in \mathbb{R}^{3 \times 6}}$$

**Implementation:**
```cpp
J.block<3, 3>(0, 0) = R;           // ✅
J.block<3, 3>(0, 3) = -R * p_skew; // ✅
H += J.transpose() * Omega_geo * J;
g += J.transpose() * Omega_geo * d;
```

**Status:** ✅ Correct

---

## Summary Table

| Method | Residual | Jacobian | Notes |
|--------|----------|----------|-------|
| Point-to-Point | $d_i \in \mathbb{R}^3$ | SVD (closed-form) | No iteration needed |
| Point-to-Plane | $n_i^T d_i \in \mathbb{R}$ | $[n_i^T R \mid -n_i^T R[p_i]_\times]$ | Normal from target |
| Symmetric | $n_{sym}^T d_i \in \mathbb{R}$ | $[n_{sym}^T R \mid -n_{sym}^T R[p_i]_\times]$ | $n_{sym} = Rn_p + n_q$ |
| GICP | $d_i \in \mathbb{R}^3$ | $[R \mid -R[p_i]_\times]$ | Weighted by $\Omega$ |
| MC-ICP | $d_i \in \mathbb{R}^3$ | $[R \mid -R[p_i]_\times]$ | Weighted by $\Omega_{geo}$ |

**Key Point:** All use $[p_i]_\times$ where $p_i$ is the **original source point in local frame**, NOT the transformed point!

---

## ✅ Final Verification: All Jacobians are Correct!

All implementations follow the right perturbation convention on SE(3) manifold:
- Translation part: $\frac{\partial \tilde{p}'}{\partial \delta t} = R$
- Rotation part: $\frac{\partial \tilde{p}'}{\partial \delta\omega} = -R[p]_\times$

