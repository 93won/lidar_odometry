/**
 * @file      PangolinViewer.cpp
 * @brief     Implementation of Pangolin-based 3D viewer for LiDAR odometry.
 * @author    Seungwon Choi
 * @date      2025-09-24
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "PangolinViewer.h"
#include "../database/LidarFrame.h"
#include "../util/ConfigUtils.h"
#include "../database/VoxelMap.h"
#include "util/LogUtils.h"
#include <iostream>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace lidar_slam {
namespace viewer {

PangolinViewer::PangolinViewer()
    : m_should_stop(false)
    , m_thread_ready(false)
    , m_auto_mode("ui.1. Auto Mode", true, true)
    , m_show_point_cloud("ui.3. Show Point Cloud (Red-Blue)", true, true)
    , m_show_features("ui.4. Show Features (Mint)", true, true)
    , m_show_trajectory("ui.5. Show Trajectory", true, true)
    , m_show_keyframes("ui.6. Show Keyframes", true, true)
    , m_show_keyframe_map("ui.7. Show Keyframe Map (Gray)", true, true)
    , m_show_surfels("ui.8. Show Surfels (Disc)", false, true)
    , m_show_gaussians("ui.8b. Show Gaussians (Ellipsoid)", true, true)
    , m_show_coordinate_frame("ui.9. Show Coordinate Frame", true, true)
    , m_top_view_follow("ui.10. Top View Follow", false, true)
    , m_step_forward_button("ui.11. Step Forward", false, false)
    , m_finish_button("ui.12. Finish & Exit", false, false)
    , m_frame_id("info.Frame ID", 0)
    , m_total_points("info.Total Points", 0)
    , m_feature_count("info.Feature Count", 0)
    , m_point_size(1.0f)
    , m_feature_point_size(5.0f)
    , m_trajectory_width(2.0f)
    , m_coordinate_frame_size(3.0f)
    , m_coordinate_frame_width(4.0f)
    , m_initialized(false)
    , m_finish_pressed(false)
    , m_step_forward_pressed(false)
    , m_voxel_map(nullptr)
{
    // Apply initial config settings
    try {
        const auto& config = util::ConfigManager::instance().get_config();
        m_top_view_follow = config.top_view_follow;
        
        // Apply coordinate frame settings
        m_coordinate_frame_size = static_cast<float>(config.coordinate_frame_size);
        m_coordinate_frame_width = static_cast<float>(config.coordinate_frame_width);
        
        LOG_INFO("[PangolinViewer] Config applied - top_view_follow: {}, coord_size: {}, coord_width: {}", 
                    config.top_view_follow, m_coordinate_frame_size, m_coordinate_frame_width);
        
        // Debug: Check if settings are properly applied
        LOG_INFO("[PangolinViewer] UI state - top_view_follow: {}", 
                    m_top_view_follow.Get());
    } catch (const std::exception& e) {
        LOG_WARN("[PangolinViewer] Could not load config settings: {}", e.what());
    }
}

PangolinViewer::~PangolinViewer() {
    shutdown();
}

bool PangolinViewer::initialize(int width, int height) {
    LOG_INFO("[PangolinViewer] Starting initialization with window size {}x{}", width, height);
    
    // Start render thread
    m_should_stop = false;
    m_thread_ready = false;
    
    m_render_thread = std::thread([this, width, height]() {
        try {
            // Create OpenGL window with Pangolin (must be done in render thread)
            pangolin::CreateWindowAndBind("LiDAR Odometry with Probabilistic Kernel Optimization (PKO)", width, height);
            glEnable(GL_DEPTH_TEST);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

            // Setup camera for 3D navigation
            float fx = width * 0.7f;
            float fy = height * 0.7f;
            m_cam_state = pangolin::OpenGlRenderState(
                pangolin::ProjectionMatrix(width, height, fx, fy, width/2, height/2, 0.1, 1000),
                pangolin::ModelViewLookAt(-5, -5, 5, 0, 0, 0, pangolin::AxisZ)
            );

            // Setup display panels
            setup_panels();

            // Set clear color to dark background
            glClearColor(0.1f, 0.1f, 0.15f, 1.0f);

            m_initialized = true;
            m_thread_ready = true;
            LOG_INFO("[PangolinViewer] Render thread initialized successfully");

            // Run render loop
            render_loop();
            
        } catch (const std::exception& e) {
            LOG_ERROR("[PangolinViewer] Exception in render thread: {}", e.what());
            m_initialized = false;
            m_thread_ready = false;
            return;
        }
        
        // Cleanup
        try {
            pangolin::DestroyWindow("LiDAR Odometry with Probabilistic Kernel Optimization (PKO)");
        } catch (const std::exception& e) {
            LOG_WARN("[PangolinViewer] Exception during window cleanup: {}", e.what());
        }
        m_initialized = false;
        LOG_INFO("[PangolinViewer] Render thread finished");
    });
    
    // Wait for thread to be ready with timeout
    auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!m_thread_ready && !m_should_stop && std::chrono::steady_clock::now() < timeout) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    
    if (!m_thread_ready) {
        LOG_ERROR("[PangolinViewer] Failed to initialize render thread within timeout");
        m_should_stop = true;
        if (m_render_thread.joinable()) {
            m_render_thread.join();
        }
        return false;
    }
    
    LOG_INFO("[PangolinViewer] Initialized successfully with window size {}x{}", width, height);
    return m_thread_ready;
}

void PangolinViewer::setup_panels() {
    // UI panel width (similar to the working example - 0.2 means 20% of screen width)
    float ui_panel_ratio = 0.2f;
    
    // Create main 3D view (takes remaining space on the right side)
    m_display_3d = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Frac(ui_panel_ratio), 1.0)
        .SetHandler(new pangolin::Handler3D(m_cam_state));
    
    // Panel (UI) area - exactly like the example
    pangolin::View& d_panel = pangolin::CreatePanel("ui")
        .SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Frac(ui_panel_ratio));
    
    LOG_INFO("[PangolinViewer] UI panel and 3D display created successfully");
}

void PangolinViewer::shutdown() {
    if (m_initialized || m_thread_ready) {
        LOG_INFO("[PangolinViewer] Shutting down viewer...");
        m_should_stop = true;
        
        if (m_render_thread.joinable()) {
            m_render_thread.join();
        }
        
        m_initialized = false;
        m_thread_ready = false;
        LOG_INFO("[PangolinViewer] Viewer shutdown completed");
    }
}

bool PangolinViewer::should_close() const {
    return m_should_stop || m_finish_pressed;
}

bool PangolinViewer::is_ready() const {
    return m_thread_ready;
}

void PangolinViewer::render_loop() {
    LOG_INFO("[PangolinViewer] Starting render loop");
    
    while (!m_should_stop) {
        // Check if window should quit
        if (pangolin::ShouldQuit()) {
            LOG_INFO("[PangolinViewer] Pangolin quit requested");
            m_should_stop = true;
            break;
        }

        // Clear screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Activate 3D view
        m_display_3d.Activate(m_cam_state);

        // Copy data once with single lock to avoid multiple locks during rendering
        std::shared_ptr<database::LidarFrame> current_frame;
        std::vector<std::shared_ptr<database::LidarFrame>> trajectory_frames_copy;
        std::vector<Matrix4f> optimized_trajectory_copy;
        PointCloudConstPtr map_cloud_copy;
        PointCloudConstPtr pre_icp_cloud_copy;
        PointCloudConstPtr post_icp_cloud_copy;

        // LOG_INFO("[PangolinViewer] Acquiring data lock for rendering");
        
        {
            std::lock_guard<std::mutex> lock(m_data_mutex);
            current_frame = m_current_frame;
            trajectory_frames_copy = m_trajectory_frames;
            optimized_trajectory_copy = m_optimized_trajectory;
            map_cloud_copy = m_map_cloud;
            pre_icp_cloud_copy = m_pre_icp_cloud;
            post_icp_cloud_copy = m_post_icp_cloud;
        }


        // LOG_INFO("[PangolinViewer] After acquiring data lock for rendering");

        // Draw 3D content - no more locks needed
        // Grid drawing disabled
        // if (m_show_grid.Get()) {
        //     draw_grid();
        // }

        if (m_show_coordinate_frame.Get()) {
            draw_coordinate_axes();
        }

        // // Draw map points (Gray) - using local copy
        // if (m_show_map_points.Get() && map_cloud_copy) {
        //     glPointSize(m_point_size);
        //     glColor3f(0.5f, 0.5f, 0.5f); // Gray
        //     glBegin(GL_POINTS);
        //     for (const auto& point : *map_cloud_copy) {
        //         glVertex3f(point.x, point.y, point.z);
        //     }
        //     glEnd();
        // }

        // Draw current frame point cloud - using local copy
        if (m_show_point_cloud.Get() && current_frame) {
            draw_point_cloud_with_frame(current_frame);
        }

        // Draw feature points as mint colored points - using local copy
        if (m_show_features.Get() && current_frame) {
            auto feature_cloud = current_frame->get_feature_cloud();
            if (feature_cloud && !feature_cloud->empty()) {
                // Transform features to world coordinates
                Eigen::Matrix4f transform_matrix = current_frame->get_pose().Matrix().cast<float>();
                
                glPointSize(7.0f); // Large mint points
                glColor3f(0.4f, 0.9f, 0.8f); // Mint color
                glBegin(GL_POINTS);
                for (const auto& point : *feature_cloud) {
                    // Transform point to world coordinates
                    Eigen::Vector4f local_point(point.x, point.y, point.z, 1.0f);
                    Eigen::Vector4f world_point = transform_matrix * local_point;
                    glVertex3f(world_point.x(), world_point.y(), world_point.z());
                }
                glEnd();
                glPointSize(1.0f);
            }
        }

        // Blue and red ICP debug clouds disabled
        // if (m_show_pre_icp_features.Get() && pre_icp_cloud_copy) { ... }
        // if (m_show_post_icp_features.Get() && post_icp_cloud_copy) { ... }

        // Draw trajectory - using local copy
        if (m_show_trajectory.Get() && trajectory_frames_copy.size() > 1) {
            draw_trajectory_with_frames(trajectory_frames_copy);
        }
        
        // // Draw optimized trajectory (PGO debug) - GREEN and THICK - DISABLED
        // if (optimized_trajectory_copy.size() > 1) {
        //     glLineWidth(m_trajectory_width * 2.0f);  // Thicker line
        //     glColor3f(0.0f, 1.0f, 0.0f);  // Bright green
        //     
        //     glBegin(GL_LINE_STRIP);
        //     for (const auto& pose : optimized_trajectory_copy) {
        //         Vector3f position = pose.block<3, 1>(0, 3);
        //         glVertex3f(position.x(), position.y(), position.z());
        //     }
        //     glEnd();
        //     
        //     glLineWidth(1.0f);
        // }

        // Draw keyframes
        if (m_show_keyframes.Get()) {
            draw_keyframes();
        }

        // Draw last keyframe map
        if (m_show_keyframe_map.Get()) {
            draw_last_keyframe_map();
        }
        
        // Draw surfels as transparent discs
        if (m_show_surfels.Get()) {
            draw_surfels();
        }
        
        // Draw Gaussian primitives as transparent ellipsoids
        if (m_show_gaussians.Get()) {
            draw_gaussians();
        }

        // Draw current pose - using local copy
        if (current_frame) {
            draw_current_pose_with_frame(current_frame);
        }

        // Handle follow camera modes - using local copy
        if (current_frame) {
            Vector3f pos = current_frame->get_pose().Translation();
            
            if (m_top_view_follow.Get()) {
                // Top-down view follow mode with zoomable camera
                static float base_height = 50.0f;
                static Eigen::Vector3f last_pos = pos;
                
                // Smooth position following
                float follow_speed = 0.1f;
                Eigen::Vector3f smooth_pos = last_pos + follow_speed * (pos - last_pos);
                last_pos = smooth_pos;
                
                // Use Follow method which allows user zoom/pan while following
                pangolin::OpenGlMatrix follow_matrix = pangolin::OpenGlMatrix::Translate(
                    smooth_pos.x(), smooth_pos.y(), smooth_pos.z()
                );
                
                // Set camera to follow the position (this allows zoom with mouse wheel)
                m_cam_state.Follow(follow_matrix, true); // true = follow rotation too
                
                // Optionally set a good default top-down view on first activation
                static bool first_activation = true;
                if (first_activation) {
                    pangolin::OpenGlMatrix initial_view = pangolin::ModelViewLookAt(
                        smooth_pos.x(), smooth_pos.y(), smooth_pos.z() + base_height,
                        smooth_pos.x(), smooth_pos.y(), smooth_pos.z(),
                        0, 1, 0
                    );
                    m_cam_state.SetModelViewMatrix(initial_view);
                    first_activation = false;
                }
            }
        }

        // Check UI button states and changes
        if (m_finish_button.Get()) {
            LOG_INFO("[PangolinViewer] Finish button pressed");
            m_finish_pressed = true;
            m_should_stop = true;
        }
        
        // Check Step Forward button
        if (Pushed(m_step_forward_button)) {
            LOG_INFO("[PangolinViewer] Step Forward button pressed");
            m_step_forward_pressed = true;
        }

        // Process UI variable changes and handle mode transitions
        static int ui_update_counter = 0;
        if (++ui_update_counter % 10 == 1) { // Check more frequently (every 10 frames)
            // Access all UI variables to ensure they're rendered in the panel
            bool auto_mode = m_auto_mode.Get();
            bool show_point_cloud = m_show_point_cloud.Get();
            bool show_features = m_show_features.Get();
            bool show_trajectory = m_show_trajectory.Get();
            bool show_coord = m_show_coordinate_frame.Get();
            
            // Handle auto mode changes (removed follow camera handling)
            static bool prev_auto = auto_mode;
            
            if (auto_mode != prev_auto) {
                prev_auto = auto_mode;
            }
        }

        // Swap frames and process events
        pangolin::FinishFrame();
        
        // Maintain higher FPS for smoother camera movement (60 FPS)
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }
    
    LOG_INFO("[PangolinViewer] Render loop finished");
}

void PangolinViewer::reset_camera() {
    m_cam_state.SetModelViewMatrix(
        pangolin::ModelViewLookAt(-5, -5, 5, 0, 0, 0, pangolin::AxisZ)
    );
}

// ===== Data Update Methods =====

void PangolinViewer::update_current_frame(std::shared_ptr<database::LidarFrame> lidar_frame) {

    std::lock_guard<std::mutex> lock(m_data_mutex);
    m_current_frame = lidar_frame;

    
    if (lidar_frame) {
        // Update frame info
        m_frame_id = static_cast<int>(lidar_frame->get_frame_id());
        
        auto point_cloud = lidar_frame->get_processed_cloud();
        m_total_points = point_cloud ? static_cast<int>(point_cloud->size()) : 0;
        
        auto feature_cloud = lidar_frame->get_feature_cloud();
        m_feature_count = feature_cloud ? static_cast<int>(feature_cloud->size()) : 0;
        
        
        // Add current pose to trajectory
        // add_trajectory_pose(lidar_frame->get_pose().Matrix());
    }

}

void PangolinViewer::update_map_points(PointCloudConstPtr map_cloud) {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    m_map_cloud = map_cloud;
}

void PangolinViewer::update_icp_debug_clouds(PointCloudConstPtr pre_icp_cloud, 
                                            PointCloudConstPtr post_icp_cloud) {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    m_pre_icp_cloud = pre_icp_cloud;
    m_post_icp_cloud = post_icp_cloud;
}

void PangolinViewer::add_trajectory_frame(std::shared_ptr<database::LidarFrame> frame) {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    m_trajectory_frames.push_back(frame);
    
    // Apply sliding window: keep only last MAX_TRAJECTORY_FRAMES frames
    if (m_trajectory_frames.size() > MAX_TRAJECTORY_FRAMES) {
        size_t excess = m_trajectory_frames.size() - MAX_TRAJECTORY_FRAMES;
        m_trajectory_frames.erase(m_trajectory_frames.begin(), 
                                  m_trajectory_frames.begin() + excess);
        LOG_DEBUG("[PangolinViewer] Trajectory sliding window: removed {} old frames, keeping {}", 
                     excess, m_trajectory_frames.size());
    }
}

void PangolinViewer::update_optimized_trajectory(const std::map<int, Matrix4f>& optimized_poses) {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    m_optimized_trajectory.clear();
    
    // Sort by keyframe ID and extract poses
    for (const auto& [id, pose] : optimized_poses) {
        m_optimized_trajectory.push_back(pose);
    }
}

// ===== UI Control Methods =====

bool PangolinViewer::is_auto_mode_enabled() const {
    return m_auto_mode.Get();  // Get current UI state directly from Pangolin variable
}

bool PangolinViewer::is_finish_requested() const {
    return m_finish_pressed;
}

bool PangolinViewer::is_step_forward_requested() {
    // Check if Step Forward button was pressed and reset the flag
    if (m_step_forward_pressed) {
        m_step_forward_pressed = false; // Reset flag after reading
        return true;
    }
    return false;
}

void PangolinViewer::process_keyboard_input(bool& auto_play, bool& step_mode, bool& advance_frame) {
    // Get current auto mode state from UI
    bool ui_auto_mode = m_auto_mode.Get();
    
    // If UI checkbox changed, update the modes accordingly
    if (ui_auto_mode && !auto_play) {
        // UI enabled auto mode - switch from step to auto
        auto_play = true;
        step_mode = false;
    } else if (!ui_auto_mode && auto_play) {
        // UI disabled auto mode - switch from auto to step
        auto_play = false;
        step_mode = true;
    }
    
    // In step mode, check if step forward was requested
    if (step_mode) {
        // Check if Step Forward button was pressed
        advance_frame = is_step_forward_requested();
    }
}

// ===== Drawing Methods =====

void PangolinViewer::draw_grid() {
    const float grid_size = 20.0f;  // Larger grid for LiDAR scale
    const float step = 1.0f;
    
    glLineWidth(1.0f);
    glBegin(GL_LINES);
    
    // Grid lines in light gray
    glColor3f(0.6f, 0.6f, 0.6f);
    for (float i = -grid_size; i <= grid_size; i += step) {
        // X direction lines
        glVertex3f(i, -grid_size, 0.0f);
        glVertex3f(i, grid_size, 0.0f);
        
        // Y direction lines
        glVertex3f(-grid_size, i, 0.0f);
        glVertex3f(grid_size, i, 0.0f);
    }
    
    // Axis lines in different colors
    glColor3f(1.0f, 0.0f, 0.0f); // X axis in red
    glVertex3f(-grid_size, 0.0f, 0.0f);
    glVertex3f(grid_size, 0.0f, 0.0f);
    
    glColor3f(0.0f, 1.0f, 0.0f); // Y axis in green
    glVertex3f(0.0f, -grid_size, 0.0f);
    glVertex3f(0.0f, grid_size, 0.0f);
    
    glEnd();
    glLineWidth(1.0f);
}

void PangolinViewer::draw_coordinate_axes() {
    glLineWidth(m_coordinate_frame_width);
    glBegin(GL_LINES);
    
    // X axis - Red
    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(m_coordinate_frame_size, 0.0f, 0.0f);
    
    // Y axis - Green
    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, m_coordinate_frame_size, 0.0f);
    
    // Z axis - Blue
    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, m_coordinate_frame_size);
    
    glEnd();
    glLineWidth(1.0f);
}

void PangolinViewer::draw_point_cloud() {
    if (!m_current_frame) {
        return;
    }
    
    auto point_cloud = m_current_frame->get_processed_cloud();
    if (!point_cloud || point_cloud->empty()) {
        return;
    }
    
    // Get the pose transformation matrix to transform points to world coordinates
    Eigen::Matrix4f transform_matrix = m_current_frame->get_pose().Matrix().cast<float>();
    
    // Draw point cloud with angle-based coloring, transformed to world coordinates
    glPointSize(m_point_size);
    
    // Draw with angle-based coloring (0-360 degrees: blue to red)
    glBegin(GL_POINTS);
    for (const auto& point : *point_cloud) {
        // Calculate angle in XY plane using local coordinates (0 to 360 degrees)
        float angle = atan2(point.y, point.x); // Returns -π to π
        angle = angle < 0 ? angle + 2.0f * M_PI : angle;      // Convert to 0 to 2π
        angle += M_PI; // Add 180 degree offset
        if (angle > 2.0f * M_PI) angle -= 2.0f * M_PI; // Keep in 0-2π range
        float angle_normalized = angle / (2.0f * M_PI);        // Normalize to [0,1]
        
        // Blue to red coloring: blue (0°) to red (360°)
        glColor3f(angle_normalized, 0.2f, 1.0f - angle_normalized);
        
        // Transform point to world coordinates for rendering
        Eigen::Vector4f local_point(point.x, point.y, point.z, 1.0f);
        Eigen::Vector4f world_point = transform_matrix * local_point;
        glVertex3f(world_point.x(), world_point.y(), world_point.z());
    }
    glEnd();
    
    glPointSize(1.0f);
}

void PangolinViewer::draw_point_cloud_with_frame(std::shared_ptr<database::LidarFrame> frame) {
    if (!frame) {
        return;
    }
    
    // Use the processed cloud for visualization
    auto point_cloud = frame->get_processed_cloud();
    if (!point_cloud || point_cloud->empty()) {
        return;
    }
    
    // Get the pose transformation matrix to transform points to world coordinates
    Eigen::Matrix4f transform_matrix = frame->get_pose().Matrix().cast<float>();
    
    // Draw point cloud with angle-based coloring, transformed to world coordinates
    glPointSize(5.0f); // Larger than map cloud points
    
    // Enable alpha blending for transparency
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    // Draw with angle-based coloring (0-360 degrees: blue to red)
    glBegin(GL_POINTS);
    for (const auto& point : *point_cloud) {
        // Calculate angle in XY plane using local coordinates (0 to 360 degrees)
        float angle = atan2(point.y, point.x); // Returns -π to π
        angle = angle < 0 ? angle + 2.0f * M_PI : angle;      // Convert to 0 to 2π
        angle += M_PI; // Add 180 degree offset
        if (angle > 2.0f * M_PI) angle -= 2.0f * M_PI; // Keep in 0-2π range
        float angle_normalized = angle / (2.0f * M_PI);        // Normalize to [0,1]
        
        // Blue to red coloring: blue (0°) to red (360°)
        glColor4f(angle_normalized, 0.2f, 1.0f - angle_normalized, 0.7f); // Alpha = 0.7 for transparency
        
        // Transform point to world coordinates for rendering
        Eigen::Vector4f local_point(point.x, point.y, point.z, 1.0f);
        Eigen::Vector4f world_point = transform_matrix * local_point;
        glVertex3f(world_point.x(), world_point.y(), world_point.z());
    }
    glEnd();
    
    glPointSize(1.0f);
}

void PangolinViewer::draw_plane_features() {
    if (!m_current_frame) {
        return;
    }
    
    auto feature_cloud = m_current_frame->get_feature_cloud();
    if (!feature_cloud || feature_cloud->empty()) {
        return;
    }
    
    glPointSize(m_feature_point_size);
    glColor3f(1.0f, 0.5f, 0.0f); // Orange for plane features
    
    glBegin(GL_POINTS);
    for (const auto& point : *feature_cloud) {
        glVertex3f(point.x, point.y, point.z);
    }
    glEnd();
    
    glPointSize(1.0f);
}

void PangolinViewer::draw_trajectory() {
    if (m_trajectory_frames.size() < 2) {
        return;
    }
    
    glLineWidth(m_trajectory_width);
    glColor3f(1.0f, 1.0f, 0.0f); // Yellow trajectory
    
    glBegin(GL_LINE_STRIP);
    for (const auto& frame : m_trajectory_frames) {
        if (frame) {
            Vector3f position = frame->get_pose().Translation();
            glVertex3f(position.x(), position.y(), position.z());
        }
    }
    glEnd();
    
    glLineWidth(1.0f);
}

void PangolinViewer::draw_trajectory_with_data(const std::vector<Matrix4f>& trajectory) {
    if (trajectory.size() < 2) {
        return;
    }
    
    glLineWidth(m_trajectory_width);
    glColor3f(1.0f, 1.0f, 0.0f); // Yellow trajectory
    
    glBegin(GL_LINE_STRIP);
    for (const auto& pose : trajectory) {
        Vector3f position = pose.block<3, 1>(0, 3);
        glVertex3f(position.x(), position.y(), position.z());
    }
    glEnd();
    
    glLineWidth(1.0f);
}

void PangolinViewer::draw_trajectory_with_frames(const std::vector<std::shared_ptr<database::LidarFrame>>& frames) {
    if (frames.size() < 2) {
        return;
    }
    
    glLineWidth(m_trajectory_width);
    glColor3f(1.0f, 1.0f, 0.0f); // Yellow trajectory
    
    glBegin(GL_LINE_STRIP);
    for (const auto& frame : frames) {
        if (frame) {
            Vector3f position = frame->get_pose().Translation();
            glVertex3f(position.x(), position.y(), position.z());
        }
    }
    glEnd();
    
    glLineWidth(1.0f);
}

void PangolinViewer::draw_current_pose() {
    Vector3f position = m_current_frame->get_pose().Translation();
    Eigen::Matrix3f rotation = m_current_frame->get_pose().RotationMatrix();
    
    // Draw current position as large red point
    glPointSize(10.0f);
    glColor3f(1.0f, 0.0f, 0.0f);
    glBegin(GL_POINTS);
    glVertex3f(position.x(), position.y(), position.z());
    glEnd();
    glPointSize(1.0f);
    
    // Draw coordinate frame at current pose
    const float axis_length = 3.0;
    glLineWidth(10.0f);
    glBegin(GL_LINES);
    
    // X-axis (Red)
    glColor3f(1.0f, 0.0f, 0.0f);
    Vector3f x_axis = position + rotation.col(0) * axis_length;
    glVertex3f(position.x(), position.y(), position.z());
    glVertex3f(x_axis.x(), x_axis.y(), x_axis.z());
    
    // Y-axis (Green)
    glColor3f(0.0f, 1.0f, 0.0f);
    Vector3f y_axis = position + rotation.col(1) * axis_length;
    glVertex3f(position.x(), position.y(), position.z());
    glVertex3f(y_axis.x(), y_axis.y(), y_axis.z());
    
    // Z-axis (Blue)
    glColor3f(0.0f, 0.0f, 1.0f);
    Vector3f z_axis = position + rotation.col(2) * axis_length;
    glVertex3f(position.x(), position.y(), position.z());
    glVertex3f(z_axis.x(), z_axis.y(), z_axis.z());
    
    glEnd();
    glLineWidth(10.0f);
}

void PangolinViewer::draw_current_pose_with_frame(std::shared_ptr<database::LidarFrame> frame) {
    if (!frame) {
        return;
    }
    
    Vector3f position = frame->get_pose().Translation();
    Eigen::Matrix3f rotation = frame->get_pose().RotationMatrix();
    
    // Draw current position as large red point
    glPointSize(10.0f);
    glColor3f(1.0f, 0.0f, 0.0f);
    glBegin(GL_POINTS);
    glVertex3f(position.x(), position.y(), position.z());
    glEnd();
    glPointSize(1.0f);
    
    // Draw coordinate frame at current pose
    const float axis_length = 3.0f;
    glLineWidth(6.0f);
    glBegin(GL_LINES);
    
    // X-axis (Red)
    glColor3f(1.0f, 0.0f, 0.0f);
    Vector3f x_axis = position + rotation.col(0) * axis_length;
    glVertex3f(position.x(), position.y(), position.z());
    glVertex3f(x_axis.x(), x_axis.y(), x_axis.z());
    
    // Y-axis (Green)
    glColor3f(0.0f, 1.0f, 0.0f);
    Vector3f y_axis = position + rotation.col(1) * axis_length;
    glVertex3f(position.x(), position.y(), position.z());
    glVertex3f(y_axis.x(), y_axis.y(), y_axis.z());
    
    // Z-axis (Blue)
    glColor3f(0.0f, 0.0f, 1.0f);
    Vector3f z_axis = position + rotation.col(2) * axis_length;
    glVertex3f(position.x(), position.y(), position.z());
    glVertex3f(z_axis.x(), z_axis.y(), z_axis.z());
    
    glEnd();
    glLineWidth(1.0f);
}

// ===== Utility Methods =====

Vector3f PangolinViewer::pcl_to_eigen(const PointType& point) const {
    return Vector3f(point.x, point.y, point.z);
}

// ===== New Drawing Functions =====

void PangolinViewer::draw_map_points() {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    if (!m_map_cloud) return;
    
    glPointSize(m_point_size);
    glColor3f(0.5f, 0.5f, 0.5f); // Gray
    glBegin(GL_POINTS);
    for (const auto& point : *m_map_cloud) {
        glVertex3f(point.x, point.y, point.z);
    }
    glEnd();
}

void PangolinViewer::draw_pre_icp_features() {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    if (!m_pre_icp_cloud) return;
    
    glPointSize(m_feature_point_size);
    glColor3f(0.0f, 0.5f, 1.0f); // Blue
    glBegin(GL_POINTS);
    for (const auto& point : *m_pre_icp_cloud) {
        glVertex3f(point.x, point.y, point.z);
    }
    glEnd();
}

void PangolinViewer::draw_post_icp_features() {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    if (!m_post_icp_cloud) return;
    
    glPointSize(m_feature_point_size);
    glColor3f(1.0f, 0.2f, 0.2f); // Red
    glBegin(GL_POINTS);
    for (const auto& point : *m_post_icp_cloud) {
        glVertex3f(point.x, point.y, point.z);
    }
    glEnd();
}

void PangolinViewer::add_keyframe(std::shared_ptr<database::LidarFrame> keyframe) {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    
    m_keyframes.push_back(keyframe);
    
    if (keyframe) {
        Vector3f position = keyframe->get_pose().Translation();
        LOG_DEBUG("[PangolinViewer] Added keyframe {} at position ({:.2f}, {:.2f}, {:.2f})",
                      keyframe->get_keyframe_id(), position.x(), position.y(), position.z());
    }
}

void PangolinViewer::clear_keyframes() {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    m_keyframes.clear();
    LOG_DEBUG("[PangolinViewer] Cleared all keyframes");
}

void PangolinViewer::draw_keyframes() {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    
    if (m_keyframes.empty()) return;
    
    glLineWidth(m_coordinate_frame_width);
    
    for (const auto& keyframe : m_keyframes) {
        if (!keyframe) continue;
        
        // Get pose dynamically (updates after PGO)
        Matrix4f pose = keyframe->get_pose().Matrix();
        
        // Extract position and rotation from pose matrix
        Vector3f position = pose.block<3,1>(0,3);
        Matrix3f rotation = pose.block<3,3>(0,0);
        
        // Scale for keyframe coordinate axes (smaller than current pose)
        float axis_length = m_coordinate_frame_size * 0.4f;
        
        // X-axis (Red)
        Vector3f x_axis = rotation.col(0) * axis_length;
        glColor3f(1.0f, 0.0f, 0.0f);
        glBegin(GL_LINES);
        glVertex3f(position.x(), position.y(), position.z());
        glVertex3f(position.x() + x_axis.x(), position.y() + x_axis.y(), position.z() + x_axis.z());
        glEnd();
        
        // Y-axis (Green)
        Vector3f y_axis = rotation.col(1) * axis_length;
        glColor3f(0.0f, 1.0f, 0.0f);
        glBegin(GL_LINES);
        glVertex3f(position.x(), position.y(), position.z());
        glVertex3f(position.x() + y_axis.x(), position.y() + y_axis.y(), position.z() + y_axis.z());
        glEnd();
        
        // Z-axis (Blue)
        Vector3f z_axis = rotation.col(2) * axis_length;
        glColor3f(0.0f, 0.0f, 1.0f);
        glBegin(GL_LINES);
        glVertex3f(position.x(), position.y(), position.z());
        glVertex3f(position.x() + z_axis.x(), position.y() + z_axis.y(), position.z() + z_axis.z());
        glEnd();
        
        // Draw keyframe ID as a small sphere/point
        glPointSize(8.0f);
        glColor3f(1.0f, 1.0f, 0.0f); // Yellow
        glBegin(GL_POINTS);
        glVertex3f(position.x(), position.y(), position.z());
        glEnd();
    }
}

void PangolinViewer::update_last_keyframe(std::shared_ptr<database::LidarFrame> last_keyframe) {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    m_last_keyframe = last_keyframe;
    
    if (last_keyframe) {
        LOG_DEBUG("[PangolinViewer] Updated last keyframe (ID: {})", last_keyframe->get_keyframe_id());
    } else {
        LOG_DEBUG("[PangolinViewer] Cleared last keyframe");
    }
}

void PangolinViewer::update_voxel_map(map::VoxelMap* voxel_map) {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    m_voxel_map = voxel_map;
}

void PangolinViewer::draw_last_keyframe_map() {
    std::lock_guard<std::mutex> lock(m_data_mutex);

    if (!m_last_keyframe) return;
    
    auto local_map = m_last_keyframe->get_local_map();
    if (!local_map || local_map->empty()) return;
    
    glPointSize(m_feature_point_size/3.0f);
    glColor3f(0.5f, 0.5f, 0.5f); // Gray
    
    
    glBegin(GL_POINTS);
    for (const auto& point : *local_map) {
        // Transform point to world coordinates
        Eigen::Vector4f world_point(point.x, point.y, point.z, 1.0f);
        glVertex3f(world_point.x(), world_point.y(), world_point.z());
    }
    glEnd();
}

void PangolinViewer::draw_surfels() {
    if (!m_voxel_map) return;
    
    // Get all L1 surfels
    auto surfels = m_voxel_map->GetL1Surfels();
    if (surfels.empty()) return;
    
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    // Surfel disc radius = L1 voxel size / 2
    float l1_voxel_size = m_voxel_map->GetVoxelSize() * m_voxel_map->GetHierarchyFactor();
    float disc_radius = l1_voxel_size / 2.0f;
    const int num_segments = 16;  // Circle resolution
    
    for (const auto& surfel_data : surfels) {
        const Eigen::Vector3f& centroid = std::get<0>(surfel_data);
        const Eigen::Vector3f& normal = std::get<1>(surfel_data);
        float planarity = std::get<2>(surfel_data);
        
        // Color intensity based on planarity: more planar = more intense
        float intensity = 1.0f - (planarity / 0.1f);
        intensity = std::max(0.3f, std::min(1.0f, intensity));
        
        // Find two perpendicular vectors to the normal
        Eigen::Vector3f u_axis, v_axis;
        if (std::abs(normal.x()) < 0.9f) {
            u_axis = Eigen::Vector3f(1, 0, 0).cross(normal).normalized();
        } else {
            u_axis = Eigen::Vector3f(0, 1, 0).cross(normal).normalized();
        }
        v_axis = normal.cross(u_axis).normalized();
        
        // Draw filled disc (mint color, semi-transparent)
        glColor4f(0.2f * intensity, 0.9f * intensity, 0.7f * intensity, 0.15f);
        glBegin(GL_TRIANGLE_FAN);
        glVertex3f(centroid.x(), centroid.y(), centroid.z());  // Center vertex
        
        for (int i = 0; i <= num_segments; ++i) {
            float angle = 2.0f * M_PI * i / num_segments;
            float cos_a = std::cos(angle);
            float sin_a = std::sin(angle);
            
            Eigen::Vector3f circle_point = centroid + disc_radius * (cos_a * u_axis + sin_a * v_axis);
            glVertex3f(circle_point.x(), circle_point.y(), circle_point.z());
        }
        glEnd();
        
        // Draw circle outline (border)
        glLineWidth(1.5f);
        glColor4f(0.0f, 0.8f * intensity, 0.6f * intensity, 0.4f);
        glBegin(GL_LINE_LOOP);
        for (int i = 0; i < num_segments; ++i) {
            float angle = 2.0f * M_PI * i / num_segments;
            float cos_a = std::cos(angle);
            float sin_a = std::sin(angle);
            
            Eigen::Vector3f circle_point = centroid + disc_radius * (cos_a * u_axis + sin_a * v_axis);
            glVertex3f(circle_point.x(), circle_point.y(), circle_point.z());
        }
        glEnd();
    }
    
    glDisable(GL_BLEND);
    glLineWidth(1.0f);
}

void PangolinViewer::draw_gaussians() {
    if (!m_voxel_map) return;
    
    // Get all L1 Gaussian primitives
    auto gaussians = m_voxel_map->GetL1Gaussians();
    if (gaussians.empty()) return;
    
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    // Ellipsoid resolution
    const int lat_segments = 12;   // Latitude divisions
    const int lon_segments = 16;   // Longitude divisions
    const float scale_factor = 2.0f;  // 2-sigma ellipsoid
    
    for (const auto& gaussian_data : gaussians) {
        const Eigen::Vector3f& mean = std::get<0>(gaussian_data);
        // const Eigen::Matrix3f& covariance = std::get<1>(gaussian_data);  // Not directly used
        const Eigen::Vector3f& eigenvalues = std::get<2>(gaussian_data);
        const Eigen::Matrix3f& eigenvectors = std::get<3>(gaussian_data);
        
        // Ellipsoid radii = sqrt(eigenvalues) * scale_factor
        Eigen::Vector3f radii = eigenvalues.cwiseSqrt() * scale_factor;
        
        // Clamp radii to reasonable range for visualization
        float max_radius = 1.0f;  // Max visualization radius
        float min_radius = 0.02f;  // Min visualization radius
        radii = radii.cwiseMax(min_radius).cwiseMin(max_radius);
        
        // Color based on geometry type (eigenvalue ratios)
        float e1 = eigenvalues[0] / (eigenvalues[2] + 1e-6f);  // λ1/λ3
        float e2 = eigenvalues[1] / (eigenvalues[2] + 1e-6f);  // λ2/λ3
        
        float r, g, b;
        if (e1 < 0.1f && e2 > 0.5f) {
            // Plane: λ1 << λ2 ≈ λ3 → Green
            r = 0.2f; g = 0.9f; b = 0.3f;
        } else if (e1 < 0.2f && e2 < 0.3f) {
            // Edge/Line: λ1 ≈ λ2 << λ3 → Blue
            r = 0.3f; g = 0.5f; b = 0.9f;
        } else if (e1 > 0.5f && e2 > 0.5f) {
            // Sphere/Corner: λ1 ≈ λ2 ≈ λ3 → Red
            r = 0.9f; g = 0.3f; b = 0.3f;
        } else {
            // Curved surface: intermediate → Yellow/Orange
            r = 0.9f; g = 0.7f; b = 0.2f;
        }
        
        // Draw ellipsoid as wireframe with transparency
        glLineWidth(1.0f);
        glColor4f(r, g, b, 0.6f);
        
        // Draw latitude circles
        for (int i = 1; i < lat_segments; ++i) {
            float lat = M_PI * i / lat_segments - M_PI / 2.0f;
            float cos_lat = std::cos(lat);
            float sin_lat = std::sin(lat);
            
            glBegin(GL_LINE_LOOP);
            for (int j = 0; j < lon_segments; ++j) {
                float lon = 2.0f * M_PI * j / lon_segments;
                float cos_lon = std::cos(lon);
                float sin_lon = std::sin(lon);
                
                // Unit sphere point
                Eigen::Vector3f unit_point(
                    cos_lat * cos_lon,
                    cos_lat * sin_lon,
                    sin_lat
                );
                
                // Scale by radii
                Eigen::Vector3f scaled = radii.cwiseProduct(unit_point);
                
                // Rotate by eigenvectors and translate by mean
                Eigen::Vector3f world_point = mean + eigenvectors * scaled;
                
                glVertex3f(world_point.x(), world_point.y(), world_point.z());
            }
            glEnd();
        }
        
        // Draw longitude lines
        for (int j = 0; j < lon_segments; ++j) {
            float lon = 2.0f * M_PI * j / lon_segments;
            float cos_lon = std::cos(lon);
            float sin_lon = std::sin(lon);
            
            glBegin(GL_LINE_STRIP);
            for (int i = 0; i <= lat_segments; ++i) {
                float lat = M_PI * i / lat_segments - M_PI / 2.0f;
                float cos_lat = std::cos(lat);
                float sin_lat = std::sin(lat);
                
                Eigen::Vector3f unit_point(
                    cos_lat * cos_lon,
                    cos_lat * sin_lon,
                    sin_lat
                );
                
                Eigen::Vector3f scaled = radii.cwiseProduct(unit_point);
                Eigen::Vector3f world_point = mean + eigenvectors * scaled;
                
                glVertex3f(world_point.x(), world_point.y(), world_point.z());
            }
            glEnd();
        }
        
        // Draw principal axes as arrows (optional - shows orientation)
        glLineWidth(2.0f);
        float arrow_scale = 0.5f;
        
        // v1 (smallest eigenvalue) - Red (typically normal direction)
        glColor4f(1.0f, 0.0f, 0.0f, 0.8f);
        glBegin(GL_LINES);
        Eigen::Vector3f axis1 = mean + arrow_scale * radii[0] * eigenvectors.col(0);
        glVertex3f(mean.x(), mean.y(), mean.z());
        glVertex3f(axis1.x(), axis1.y(), axis1.z());
        glEnd();
        
        // v2 (middle eigenvalue) - Green
        glColor4f(0.0f, 1.0f, 0.0f, 0.8f);
        glBegin(GL_LINES);
        Eigen::Vector3f axis2 = mean + arrow_scale * radii[1] * eigenvectors.col(1);
        glVertex3f(mean.x(), mean.y(), mean.z());
        glVertex3f(axis2.x(), axis2.y(), axis2.z());
        glEnd();
        
        // v3 (largest eigenvalue) - Blue
        glColor4f(0.0f, 0.0f, 1.0f, 0.8f);
        glBegin(GL_LINES);
        Eigen::Vector3f axis3 = mean + arrow_scale * radii[2] * eigenvectors.col(2);
        glVertex3f(mean.x(), mean.y(), mean.z());
        glVertex3f(axis3.x(), axis3.y(), axis3.z());
        glEnd();
    }
    
    glDisable(GL_BLEND);
    glLineWidth(1.0f);
}

} // namespace viewer
} // namespace lidar_slam
