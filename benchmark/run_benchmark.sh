#!/bin/bash

# ICP Benchmark Runner Script
# Usage: ./run_benchmark.sh [sequence] [frame1] [frame2]

# Default values
SEQUENCE=${1:-5}
FRAME1=${2:-0}
FRAME2=${3:-1}

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../build"
DATASET_PATH="/home/eugene/data/KITTI/Velodyne"
POSE_PATH="/home/eugene/data/KITTI/kitti_odometry_poses/dataset/poses"

# Check if executable exists
if [ ! -f "${BUILD_DIR}/icp_benchmark" ]; then
    echo "Building icp_benchmark..."
    cd "${BUILD_DIR}" && make icp_benchmark -j8
fi

# Run benchmark
echo "Running ICP Benchmark"
echo "====================="
echo "Sequence: ${SEQUENCE}"
echo "Frames: ${FRAME1} -> ${FRAME2}"
echo ""

cd "${BUILD_DIR}" && ./icp_benchmark \
    --dataset_path "${DATASET_PATH}" \
    --pose_path "${POSE_PATH}" \
    --sequence ${SEQUENCE} \
    --frame1 ${FRAME1} \
    --frame2 ${FRAME2} \
    --verbose

echo ""
echo "Results saved to:"
echo "  - icp_benchmark_results.csv"
echo "  - icp_benchmark_convergence.csv"
