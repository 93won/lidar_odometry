/**
 * @file      PangolinViewer.h
 * @brief     Pangolin-based 3D viewer for LiDAR odometry visualization.
 * @author    Seungwon Choi
 * @date      2025-09-24
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#pragma once

#include <pangolin/display/display.h>
#include <pangolin/display/view.h>
#include <pangolin/handler/handler.h>
#include <pangolin/gl/glinclude.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/var/var.h>
#include <pangolin/var/varextra.h>
#include <pangolin/display/widgets.h>
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <mutex>
#include <thread>
#include <atomic>
#include "../util/MathUtils.h"
#include "../util/PointCloudUtils.h"

// Forward declarations
namespace lidar_slam {
namespace database {
class LidarFrame;
}
namespace map {
class VoxelMap;
}
}

namespace lidar_slam {
namespace viewer {

using PointType = util::Point3D;  // Use our point type
using PointCloud = util::PointCloud;
using PointCloudPtr = PointCloud::Ptr;
using PointCloudConstPtr = PointCloud::ConstPtr;
using Vector3f = Eigen::Vector3f;
using Matrix4f = Eigen::Matrix4f;

/**
 * @brief Pangolin-based 3D viewer for LiDAR odometry visualization
 * 
 * This viewer displays:
 * - Current LiDAR frame point cloud (gray)
 * - Extracted plane features (red, larger points)
 * - Camera trajectory
 * - Coordinate frames and axes
 */
class PangolinViewer {
public:
    PangolinViewer();
    ~PangolinViewer();

    // ===== Initialization and Control =====
    
    /**
     * @brief Initialize the viewer window and start render thread
     * @param width Window width
     * @param height Window height
     * @return True if initialization succeeded
     */
    bool initialize(int width = 1280, int height = 960);
    
    /**
     * @brief Shutdown the viewer and stop render thread
     */
    void shutdown();
    
    /**
     * @brief Check if viewer should close
     * @return True if close requested
     */
    bool should_close() const;
    
    /**
     * @brief Check if viewer is ready
     * @return True if initialized and ready
     */
    bool is_ready() const;
    
    /**
     * @brief Main render loop - removed from public (now runs in thread)
     */
    
    /**
     * @brief Reset camera to default view
     */
    void reset_camera();

    // ===== Data Updates =====
    
    /**
     * @brief Update current LiDAR frame  
     * @param lidar_frame Current LiDAR frame
     */
    void update_current_frame(std::shared_ptr<database::LidarFrame> lidar_frame);
    
    /**
     * @brief Update map points for visualization
     * @param map_cloud Map point cloud in world coordinates
     */
    void update_map_points(PointCloudConstPtr map_cloud);
    
    /**
     * @brief Update ICP debug clouds for visualization
     * @param pre_icp_cloud LiDAR features before ICP (world coordinates)
     * @param post_icp_cloud LiDAR features after ICP (world coordinates)
     */
    void update_icp_debug_clouds(PointCloudConstPtr pre_icp_cloud, 
                                PointCloudConstPtr post_icp_cloud);
    
    /**
     * @brief Add keyframe for visualization (uses get_pose() for dynamic update)
     * @param keyframe Frame object to add as keyframe
     */
    void add_keyframe(std::shared_ptr<database::LidarFrame> keyframe);
    
    /**
     * @brief Clear all keyframes
     */
    void clear_keyframes();
    
    /**
     * @brief Update last keyframe for local map visualization
     * @param last_keyframe Last keyframe pointer (will use get_local_map())
     */
    void update_last_keyframe(std::shared_ptr<database::LidarFrame> last_keyframe);
    
    /**
     * @brief Update VoxelMap pointer for on-demand visualization
     * @param voxel_map Pointer to VoxelMap (will extract points on-demand during rendering)
     */
    void update_voxel_map(map::VoxelMap* voxel_map);
    
    /**
     * @brief Add frame to trajectory (will use get_pose() for dynamic update)
     * @param frame Frame to add to trajectory
     */
    void add_trajectory_frame(std::shared_ptr<database::LidarFrame> frame);
    
    /**
     * @brief Update optimized trajectory from pose graph optimization
     * @param optimized_poses Map of keyframe ID to optimized pose
     */
    void update_optimized_trajectory(const std::map<int, Matrix4f>& optimized_poses);
    
    // ===== UI Controls =====
    
    /**
     * @brief Check if auto mode is enabled
     * @return True if auto mode enabled
     */
    bool is_auto_mode_enabled() const;
    
    /**
     * @brief Check if finish was requested
     * @return True if finish button pressed
     */
    bool is_finish_requested() const;
    
    /**
     * @brief Check if step forward was requested
     * @return True if step forward button was pressed
     */
    bool is_step_forward_requested();
    
    /**
     * @brief Process keyboard input
     * @param auto_play Auto play mode flag
     * @param step_mode Step mode flag
     * @param advance_frame Advance frame flag
     */
    void process_keyboard_input(bool& auto_play, bool& step_mode, bool& advance_frame);

private:
    // ===== Pangolin Components =====
    pangolin::OpenGlRenderState m_cam_state;
    pangolin::View m_display_3d;
    pangolin::View m_display_panel;
    
    // ===== Data Storage =====
    std::shared_ptr<database::LidarFrame> m_current_frame;  ///< Current LiDAR frame
    std::vector<std::shared_ptr<database::LidarFrame>> m_trajectory_frames; ///< All frames for trajectory (use get_pose() for dynamic update)
    std::vector<Matrix4f> m_optimized_trajectory;           ///< PGO optimized trajectory (for debugging)
    
    // ===== Keyframe Data =====
    std::vector<std::shared_ptr<database::LidarFrame>> m_keyframes; ///< Keyframe objects (use get_pose() for dynamic update)
    
    // ===== New Visualization Data =====
    PointCloudConstPtr m_map_cloud;                        ///< Map points (world coordinate) - Gray
    PointCloudConstPtr m_pre_icp_cloud;                    ///< Pre-ICP lidar features (world) - Blue  
    PointCloudConstPtr m_post_icp_cloud;                   ///< Post-ICP lidar features (world) - Red
    std::shared_ptr<database::LidarFrame> m_last_keyframe; ///< Last keyframe for local map visualization
    map::VoxelMap* m_voxel_map;                            ///< VoxelMap pointer for on-demand visualization
    
    // ===== Thread Safety =====
    mutable std::mutex m_data_mutex;
    
    // ===== Thread Management =====
    std::thread m_render_thread;                       ///< Render thread
    std::atomic<bool> m_should_stop;                   ///< Thread stop flag
    std::atomic<bool> m_thread_ready;                  ///< Thread ready flag
    
    // ===== UI Variables =====
    pangolin::Var<bool> m_auto_mode;                   ///< Auto mode checkbox
    pangolin::Var<bool> m_show_point_cloud;            ///< Show LiDAR point cloud (Red-Blue)
    pangolin::Var<bool> m_show_features;               ///< Show LiDAR features (Mint)
    pangolin::Var<bool> m_show_trajectory;             ///< Show estimated trajectory checkbox
    pangolin::Var<bool> m_show_keyframes;              ///< Show keyframes checkbox
    pangolin::Var<bool> m_show_keyframe_map;           ///< Show last keyframe map checkbox
    pangolin::Var<bool> m_show_surfels;                ///< Show surfel discs checkbox
    pangolin::Var<bool> m_show_gaussians;              ///< Show Gaussian primitive ellipsoids checkbox
    pangolin::Var<bool> m_show_coordinate_frame;       ///< Show coordinate frame checkbox
    pangolin::Var<bool> m_top_view_follow;             ///< Top-down view follow mode checkbox
    pangolin::Var<bool> m_step_forward_button;         ///< Step forward button
    pangolin::Var<bool> m_finish_button;               ///< Finish button
    pangolin::Var<int> m_frame_id;                     ///< Current frame ID
    pangolin::Var<int> m_total_points;                 ///< Total points in frame
    pangolin::Var<int> m_feature_count;                ///< Number of features
    
    // ===== Display Settings =====
    float m_point_size;                                 ///< Point cloud point size
    float m_feature_point_size;                        ///< Feature point size
    float m_trajectory_width;                          ///< Trajectory line width
    float m_coordinate_frame_size;                     ///< Coordinate frame axis length
    float m_coordinate_frame_width;                    ///< Coordinate frame line width
    
    // ===== Memory Management =====
    static constexpr size_t MAX_TRAJECTORY_FRAMES = 1000; ///< Maximum trajectory frames to keep
    
    // ===== State =====
    bool m_initialized;                                 ///< Initialization state
    bool m_finish_pressed;                             ///< Finish button state
    bool m_step_forward_pressed;                       ///< Step forward button state
    
    // ===== Internal Methods =====
    
    /**
     * @brief Setup UI panels
     */
    void setup_panels();
    
    /**
     * @brief Main render loop - runs in separate thread
     */
    void render_loop();
    
    /**
     * @brief Draw 3D grid
     */
    void draw_grid();
    
    /**
     * @brief Draw coordinate axes
     */
    void draw_coordinate_axes();
    
    /**
     * @brief Draw map points (world coordinate) in gray
     */
    void draw_map_points();
    
    /**
     * @brief Draw pre-ICP LiDAR features (world coordinate) in blue
     */
    void draw_pre_icp_features();
    
    /**
     * @brief Draw post-ICP LiDAR features (world coordinate) in red
     */
    void draw_post_icp_features();
    
    /**
     * @brief Draw point cloud
     */
    void draw_point_cloud();
    
    /**
     * @brief Draw point cloud with specific frame data
     * @param frame LiDAR frame containing point cloud data
     */
    void draw_point_cloud_with_frame(std::shared_ptr<database::LidarFrame> frame);
    
    /**
     * @brief Draw plane features
     */
    void draw_plane_features();
    
    /**
     * @brief Draw trajectory
     */
    void draw_trajectory();
    
    /**
     * @brief Draw trajectory with specific trajectory data
     * @param trajectory Vector of poses forming the trajectory
     */
    void draw_trajectory_with_data(const std::vector<Matrix4f>& trajectory);
    
    /**
     * @brief Draw trajectory with frame objects (uses get_pose() for dynamic update)
     * @param frames Vector of frames forming the trajectory
     */
    void draw_trajectory_with_frames(const std::vector<std::shared_ptr<database::LidarFrame>>& frames);
    
    /**
     * @brief Draw keyframes as coordinate axes
     */
    void draw_keyframes();
    
    /**
     * @brief Draw last keyframe map points as large points
     */
    void draw_last_keyframe_map();
    
    /**
     * @brief Draw VoxelMap points (extracts points on-demand from VoxelMap)
     */
    void draw_voxel_map();
    
    /**
     * @brief Draw L1 surfels as transparent discs
     */
    void draw_surfels();
    
    /**
     * @brief Draw L1 Gaussian primitives as transparent ellipsoids
     */
    void draw_gaussians();
    
    /**
     * @brief Draw current pose
     */
    void draw_current_pose();
    
    /**
     * @brief Draw current pose with specific frame data
     * @param frame LiDAR frame containing pose data
     */
    void draw_current_pose_with_frame(std::shared_ptr<database::LidarFrame> frame);
    
    /**
     * @brief Convert Point3D to Eigen vector
     * @param point Point3D point
     * @return Eigen vector
     */
    Vector3f pcl_to_eigen(const PointType& point) const;
};

} // namespace viewer
} // namespace lidar_slam
