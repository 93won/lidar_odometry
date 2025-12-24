#include "ICPBenchmark.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <sstream>
#include <filesystem>
#include <fstream>
#include <map>
#include <cmath>
#include <algorithm>
#include <thread>
#include <mutex>
#include <atomic>

namespace fs = std::filesystem;

std::atomic<int> g_progress{0};
std::mutex g_print_mutex;

void printUsage(const char* program_name)
{
    std::cout << "ICP Benchmark for KITTI - Method Parallel\n";
    std::cout << "Usage: " << program_name << " [options]\n";
    std::cout << "Required: --dataset_path <path> --pose_path <path> --sequence <n>\n";
    std::cout << "Optional: --output_dir <path> --voxel_size <f> --verbose\n";
}

int countFrames(const std::string& dataset_path, int sequence)
{
    std::ostringstream velodyne_dir;
    velodyne_dir << dataset_path << "/" << std::setfill('0') << std::setw(2) << sequence << "/velodyne";
    int count = 0;
    for (const auto& entry : fs::directory_iterator(velodyne_dir.str())) {
        if (entry.path().extension() == ".bin") count++;
    }
    return count;
}

struct SingleResult {
    int frame1, frame2;
    float voxel_size;
    std::string method;
    int iterations;
    double trans_error_m, rot_error_deg, time_ms;
    bool converged;
};

void runBenchmarkForMethod(
    lidar_slam::benchmark::ICPMethod method,
    float voxel_size,
    const lidar_slam::benchmark::KITTIBenchmarkConfig& base_config,
    const std::vector<int>& frames,
    const std::string& output_dir,
    int method_idx)  // method index for display
{
    std::string method_name = lidar_slam::benchmark::to_string(method);
    lidar_slam::benchmark::KITTIBenchmarkConfig config = base_config;
    config.voxel_sizes = {voxel_size};
    
    std::ostringstream filename;
    filename << output_dir << "icp_results_" << method_name << ".csv";
    std::ofstream csv_out(filename.str());
    csv_out << "frame1,frame2,voxel_size,method,iterations,trans_error_m,rot_error_deg,time_ms,converged\n";
    
    std::vector<SingleResult> results;
    int local_processed = 0;
    int total_local = static_cast<int>(frames.size());
    int conv_count = 0;
    
    // Print start message
    {
        std::lock_guard<std::mutex> lock(g_print_mutex);
        std::cout << "[" << method_name << "] Starting... (" << total_local << " pairs)\n" << std::flush;
    }
    
    for (int frame : frames) {
        config.frame1 = frame;
        config.frame2 = frame + 1;
        
        lidar_slam::benchmark::KITTIBenchmarkRunner runner(config);
        std::vector<lidar_slam::benchmark::ICPMethod> single_method = {method};
        auto run_results = runner.runAll();  // Use runAll like old benchmark
        
        for (const auto& kv : run_results) {
            for (const auto& result : kv.second) {
                // Filter only the method we're responsible for
                if (result.method != method) continue;
                
                SingleResult sr;
                sr.frame1 = frame;
                sr.frame2 = frame + 1;
                sr.voxel_size = voxel_size;
                sr.method = method_name;
                sr.iterations = result.total_iterations;
                sr.trans_error_m = result.final_translation_error;
                sr.rot_error_deg = result.final_rotation_error;
                sr.time_ms = result.total_time_ms;
                sr.converged = result.converged;
                
                csv_out << sr.frame1 << "," << sr.frame2 << "," << sr.voxel_size << ","
                        << sr.method << "," << sr.iterations << "," << sr.trans_error_m << ","
                        << sr.rot_error_deg << "," << sr.time_ms << "," << (sr.converged ? 1 : 0) << "\n";
                results.push_back(sr);
                if (sr.converged) conv_count++;
            }
        }
        
        local_processed++;
        
        // Print progress every 50 frames
        if (local_processed % 50 == 0) {
            double pct = 100.0 * local_processed / total_local;
            double conv_rate = 100.0 * conv_count / local_processed;
            std::lock_guard<std::mutex> lock(g_print_mutex);
            std::cout << "[" << method_name << "] " << local_processed << "/" << total_local 
                      << " (" << std::fixed << std::setprecision(1) << pct << "%)"
                      << " conv=" << std::setprecision(1) << conv_rate << "%\n" << std::flush;
        }
    }
    csv_out.close();
    
    int n = static_cast<int>(results.size());
    if (n > 0) {
        double sum_time = 0;
        int conv_count = 0;
        for (const auto& r : results) {
            sum_time += r.time_ms;
            if (r.converged) conv_count++;
        }
        double avg_time = sum_time / n;
        double conv_rate = 100.0 * conv_count / n;
        
        std::vector<double> trans_conv, rot_conv;
        for (const auto& r : results) {
            if (r.converged) {
                trans_conv.push_back(r.trans_error_m);
                rot_conv.push_back(r.rot_error_deg);
            }
        }
        double median_trans = 0, median_rot = 0;
        if (!trans_conv.empty()) {
            std::sort(trans_conv.begin(), trans_conv.end());
            std::sort(rot_conv.begin(), rot_conv.end());
            int nc = static_cast<int>(trans_conv.size());
            median_trans = trans_conv[nc/2];
            median_rot = rot_conv[nc/2];
        }
        
        std::lock_guard<std::mutex> lock(g_print_mutex);
        std::cout << "\n[DONE] " << method_name << " | Conv: " << conv_count << "/" << n 
                  << " (" << std::fixed << std::setprecision(1) << conv_rate << "%)"
                  << " | Median: " << std::setprecision(4) << median_trans*100 << "cm, " << median_rot << "deg"
                  << " | Time: " << std::setprecision(1) << avg_time << "ms" << std::endl;
    }
}

int main(int argc, char** argv)
{
    lidar_slam::benchmark::KITTIBenchmarkConfig config;
    config.max_iterations = 10;
    config.verbose = false;
    std::string pose_path, output_dir = "./";
    float voxel_size = 0.5f;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { printUsage(argv[0]); return 0; }
        else if (arg == "--dataset_path" && i+1 < argc) config.dataset_path = argv[++i];
        else if (arg == "--pose_path" && i+1 < argc) pose_path = argv[++i];
        else if (arg == "--output_dir" && i+1 < argc) { output_dir = argv[++i]; if (output_dir.back() != '/') output_dir += '/'; }
        else if (arg == "--sequence" && i+1 < argc) config.sequence = std::stoi(argv[++i]);
        else if (arg == "--voxel_size" && i+1 < argc) voxel_size = std::stof(argv[++i]);
        else if (arg == "--verbose") config.verbose = true;
    }
    
    if (config.dataset_path.empty() || pose_path.empty()) {
        std::cerr << "Error: --dataset_path and --pose_path required\n";
        return 1;
    }
    
    fs::create_directories(output_dir);
    std::ostringstream gt_path;
    gt_path << pose_path << "/" << std::setfill('0') << std::setw(2) << config.sequence << ".txt";
    config.ground_truth_path = gt_path.str();
    
    int total_frames = countFrames(config.dataset_path, config.sequence);
    int num_pairs = total_frames - 1;
    
    std::vector<int> frames;
    for (int i = 0; i < num_pairs; ++i) frames.push_back(i);
    
    std::vector<lidar_slam::benchmark::ICPMethod> methods = {
        lidar_slam::benchmark::ICPMethod::POINT_TO_PLANE,
        lidar_slam::benchmark::ICPMethod::SYMMETRIC,
        lidar_slam::benchmark::ICPMethod::GICP,
        lidar_slam::benchmark::ICPMethod::MC_ICP
    };
    
    std::cout << "\nICP Benchmark: Seq " << config.sequence << " (" << num_pairs << " pairs), voxel=" << voxel_size << "m\n";
    std::cout << "Parallel: " << methods.size() << " threads (one per method)\n\n";
    
    int total_tasks = num_pairs * static_cast<int>(methods.size());
    
    std::vector<std::thread> threads;
    for (const auto& method : methods) {
        threads.emplace_back(runBenchmarkForMethod, method, voxel_size, std::cref(config), std::cref(frames), output_dir, total_tasks);
    }
    for (auto& t : threads) t.join();
    
    std::cout << "\n=== ALL DONE ===\n\n";
    
    std::ofstream merged(output_dir + "icp_benchmark_all_results.csv");
    merged << "frame1,frame2,voxel_size,method,iterations,trans_error_m,rot_error_deg,time_ms,converged\n";
    for (const auto& method : methods) {
        std::string name = lidar_slam::benchmark::to_string(method);
        std::ifstream in(output_dir + "icp_results_" + name + ".csv");
        std::string line;
        std::getline(in, line);
        while (std::getline(in, line)) if (!line.empty()) merged << line << "\n";
    }
    merged.close();
    
    std::cout << "Merged: " << output_dir << "icp_benchmark_all_results.csv\n";
    return 0;
}
