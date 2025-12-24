/**
 * @file      icp_benchmark.cpp
 * @brief     Pure ICP Algorithm Benchmark Application
 * @author    Seungwon Choi
 * @date      2025-12-22
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 * 
 * Usage:
 *   ./icp_benchmark --dataset_path /path/to/kitti --sequence 0 --frame_start 0 --frame_end 100
 */

#include "ICPBenchmark.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <sstream>
#include <filesystem>
#include <fstream>
#include <map>

namespace fs = std::filesystem;

void printUsage(const char* program_name)
{
    std::cout << "Pure ICP Algorithm Benchmark\n";
    std::cout << "=============================\n\n";
    std::cout << "Usage: " << program_name << " [options]\n\n";
    std::cout << "Required options:\n";
    std::cout << "  --dataset_path <path>  Path to KITTI velodyne data\n";
    std::cout << "  --pose_path <path>     Path to KITTI ground truth poses\n";
    std::cout << "  --sequence <n>         Sequence number (0-10)\n\n";
    std::cout << "Optional options:\n";
    std::cout << "  --frame_start <n>      Start frame index (default: 0)\n";
    std::cout << "  --frame_end <n>        End frame index (default: -1 = all frames)\n";
    std::cout << "  --voxel_sizes <sizes>  Comma-separated voxel sizes (default: 0.5)\n";
    std::cout << "  --max_iterations <n>   Maximum ICP iterations (default: 50)\n";
    std::cout << "  --verbose              Enable verbose output\n";
}

std::vector<float> parseVoxelSizes(const std::string& str)
{
    std::vector<float> sizes;
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, ',')) {
        sizes.push_back(std::stof(item));
    }
    return sizes;
}

int countFrames(const std::string& dataset_path, int sequence)
{
    std::ostringstream velodyne_dir;
    velodyne_dir << dataset_path << "/"
                 << std::setfill('0') << std::setw(2) << sequence << "/velodyne";
    
    int count = 0;
    for (const auto& entry : fs::directory_iterator(velodyne_dir.str())) {
        if (entry.path().extension() == ".bin") {
            count++;
        }
    }
    return count;
}

struct AggregatedStats {
    int total_pairs = 0;
    int converged_count = 0;
    double sum_trans_error = 0.0;
    double sum_rot_error = 0.0;
    double sum_iterations = 0.0;
    double sum_time = 0.0;
    
    void add(const lidar_slam::benchmark::BenchmarkResult& result) {
        total_pairs++;
        if (result.converged) converged_count++;
        sum_trans_error += result.final_translation_error;
        sum_rot_error += result.final_rotation_error;
        sum_iterations += result.total_iterations;
        sum_time += result.total_time_ms;
    }
    
    double avgTransError() const { return total_pairs > 0 ? sum_trans_error / total_pairs : 0; }
    double avgRotError() const { return total_pairs > 0 ? sum_rot_error / total_pairs : 0; }
    double avgIterations() const { return total_pairs > 0 ? sum_iterations / total_pairs : 0; }
    double avgTime() const { return total_pairs > 0 ? sum_time / total_pairs : 0; }
    double convergeRate() const { return total_pairs > 0 ? 100.0 * converged_count / total_pairs : 0; }
};

int main(int argc, char** argv)
{
    lidar_slam::benchmark::KITTIBenchmarkConfig config;
    config.voxel_sizes = {0.5f};  // 0.5m downsampling for Point-to-Plane, GICP, MC-ICP
    config.verbose = false;
    std::string output_file = "icp_benchmark_results.csv";
    std::string pose_path;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        }
        else if (arg == "--dataset_path" && i + 1 < argc) {
            config.dataset_path = argv[++i];
        }
        else if (arg == "--pose_path" && i + 1 < argc) {
            pose_path = argv[++i];
        }
        else if (arg == "--sequence" && i + 1 < argc) {
            config.sequence = std::stoi(argv[++i]);
        }
        else if (arg == "--frame_start" && i + 1 < argc) {
            config.frame_start = std::stoi(argv[++i]);
        }
        else if (arg == "--frame_end" && i + 1 < argc) {
            config.frame_end = std::stoi(argv[++i]);
        }
        else if (arg == "--frame1" && i + 1 < argc) {
            config.frame1 = std::stoi(argv[++i]);
        }
        else if (arg == "--frame2" && i + 1 < argc) {
            config.frame2 = std::stoi(argv[++i]);
        }
        else if (arg == "--voxel_sizes" && i + 1 < argc) {
            config.voxel_sizes = parseVoxelSizes(argv[++i]);
        }
        else if (arg == "--max_iterations" && i + 1 < argc) {
            config.max_iterations = std::stoi(argv[++i]);
        }
        else if (arg == "--verbose") {
            config.verbose = true;
        }
    }
    
    if (config.dataset_path.empty() || pose_path.empty()) {
        std::cerr << "Error: --dataset_path and --pose_path are required\n";
        return 1;
    }
    
    std::ostringstream gt_path;
    gt_path << pose_path << "/" << std::setfill('0') << std::setw(2) << config.sequence << ".txt";
    config.ground_truth_path = gt_path.str();
    
    int total_frames = countFrames(config.dataset_path, config.sequence);
    
    int frame_start, frame_end;
    if (config.frame1 >= 0 && config.frame2 >= 0) {
        frame_start = config.frame1;
        frame_end = config.frame2;
    } else {
        frame_start = config.frame_start;
        frame_end = (config.frame_end < 0) ? total_frames - 1 : config.frame_end;
    }
    
    int num_pairs = frame_end - frame_start;
    
    std::cout << "\nICP Benchmark Configuration\n";
    std::cout << "===========================\n";
    std::cout << "Sequence: " << config.sequence << " (" << total_frames << " frames)\n";
    std::cout << "Frame range: " << frame_start << " -> " << frame_end << " (" << num_pairs << " pairs)\n";
    std::cout << "Voxel size: " << config.voxel_sizes[0] << "m\n";
    std::cout << "Convergence: delta_t < 0.001m AND delta_r < 0.001rad\n\n";
    
    std::map<lidar_slam::benchmark::ICPMethod, AggregatedStats> method_stats;
    
    // Open per-frame CSV file
    std::string detailed_csv_file = "icp_benchmark_detailed.csv";
    std::ofstream detailed_csv(detailed_csv_file);
    detailed_csv << "frame_pair,method,iterations,trans_error_cm,rot_error_deg,time_ms,converged\n";
    
    std::vector<lidar_slam::benchmark::ICPMethod> method_order = {
        // lidar_slam::benchmark::ICPMethod::POINT_TO_POINT,  // Skip Point-to-Point
        lidar_slam::benchmark::ICPMethod::POINT_TO_PLANE,
        lidar_slam::benchmark::ICPMethod::SYMMETRIC,
        lidar_slam::benchmark::ICPMethod::GICP,
        lidar_slam::benchmark::ICPMethod::MC_ICP
        // lidar_slam::benchmark::ICPMethod::NDT  // NDT excluded - needs tuning
    };
    
    auto printIntermediateResults = [&](int current_pairs) {
        std::cout << "\n\n--- Intermediate Results (" << current_pairs << " pairs) ---\n";
        std::cout << std::left << std::setw(16) << "Method"
                  << std::right << std::setw(8) << "Iter"
                  << std::setw(12) << "Trans(cm)"
                  << std::setw(12) << "Rot(deg)"
                  << std::setw(10) << "Time(ms)"
                  << std::setw(10) << "Conv%"
                  << "\n";
        for (auto method : method_order) {
            if (method_stats.find(method) == method_stats.end()) continue;
            const auto& stats = method_stats[method];
            std::cout << std::left << std::setw(16) << lidar_slam::benchmark::to_string(method)
                      << std::right << std::fixed
                      << std::setw(8) << std::setprecision(1) << stats.avgIterations()
                      << std::setw(12) << std::setprecision(2) << stats.avgTransError() * 100.0
                      << std::setw(12) << std::setprecision(4) << stats.avgRotError()
                      << std::setw(10) << std::setprecision(1) << stats.avgTime()
                      << std::setw(9) << std::setprecision(1) << stats.convergeRate() << "%"
                      << "\n";
        }
        std::cout << "\n";
    };
    
    for (int frame = frame_start; frame < frame_end; ++frame) {
        config.frame1 = frame;
        config.frame2 = frame + 1;
        
        int current_pair = frame - frame_start + 1;
        std::cout << "\r[" << current_pair << "/" << num_pairs << "] Frame " 
                  << frame << " -> " << (frame + 1) << "   " << std::flush;
        
        lidar_slam::benchmark::KITTIBenchmarkRunner runner(config);
        auto results = runner.runAll();
        
        for (const auto& [voxel_size, method_results] : results) {
            for (const auto& result : method_results) {
                method_stats[result.method].add(result);
                
                // Write per-frame result to detailed CSV
                detailed_csv << frame << "->" << (frame+1) << ","
                             << result.method_name << ","
                             << result.total_iterations << ","
                             << result.final_translation_error * 100.0 << ","
                             << result.final_rotation_error << ","
                             << result.total_time_ms << ","
                             << (result.converged ? 1 : 0) << "\n";
            }
        }
        
        // Print intermediate results every 10 pairs
        if (current_pair % 10 == 0) {
            printIntermediateResults(current_pair);
        }
    }
    
    detailed_csv.close();
    std::cout << "\nPer-frame results exported to: " << detailed_csv_file << "\n";
    std::cout << "================================================================================\n";
    std::cout << "ICP BENCHMARK FINAL RESULTS (Aggregated over " << num_pairs << " frame pairs)\n";
    std::cout << "================================================================================\n\n";
    
    std::cout << std::left << std::setw(18) << "Method"
              << std::right << std::setw(10) << "Avg Iter"
              << std::setw(14) << "Trans(cm)"
              << std::setw(14) << "Rot(deg)"
              << std::setw(12) << "Time(ms)"
              << std::setw(12) << "Conv %"
              << "\n";
    std::cout << std::string(80, '-') << "\n";
    
    std::ofstream csv(output_file);
    csv << "Method,AvgIter,AvgTrans_cm,AvgRot_deg,AvgTime_ms,ConvRate\n";
    
    for (auto method : method_order) {
        if (method_stats.find(method) == method_stats.end()) continue;
        
        const auto& stats = method_stats[method];
        std::string name = lidar_slam::benchmark::to_string(method);
        
        std::cout << std::left << std::setw(18) << name
                  << std::right << std::fixed
                  << std::setw(10) << std::setprecision(1) << stats.avgIterations()
                  << std::setw(14) << std::setprecision(2) << stats.avgTransError() * 100.0
                  << std::setw(14) << std::setprecision(4) << stats.avgRotError()
                  << std::setw(12) << std::setprecision(1) << stats.avgTime()
                  << std::setw(11) << std::setprecision(1) << stats.convergeRate() << "%"
                  << "\n";
        
        csv << name << "," << stats.avgIterations() << "," << stats.avgTransError() * 100.0 
            << "," << stats.avgRotError() << "," << stats.avgTime() << "," << stats.convergeRate() << "\n";
    }
    
    std::cout << std::string(80, '=') << "\n";
    csv.close();
    
    std::cout << "\nResults exported to: " << output_file << "\n";
    
    return 0;
}
