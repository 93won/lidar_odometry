#!/usr/bin/env python3
"""
Plot ICP benchmark results as box plots.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

def main():
    # Load data
    csv_path = "/home/eugene/data/KITTI/benchmark/icp_benchmark_all_results.csv"
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} rows")
    print(f"Methods: {df['method'].unique()}")
    print(f"Voxel sizes: {sorted(df['voxel_size'].unique())}")
    
    # Filter converged only
    df_conv = df[df['converged'] == 1]
    print(f"Converged: {len(df_conv)} / {len(df)}")
    
    # Create figure with 2 rows: translation error and rotation error
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    methods = ['Point-to-Plane', 'Symmetric', 'GICP', 'MC-ICP']
    voxel_sizes = sorted(df['voxel_size'].unique())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Prepare data for box plots
    for ax_idx, (metric, ylabel, title) in enumerate([
        ('trans_error_m', 'Translation Error (m)', 'Translation Error by Method and Voxel Size'),
        ('rot_error_deg', 'Rotation Error (deg)', 'Rotation Error by Method and Voxel Size')
    ]):
        ax = axes[ax_idx]
        
        # Group positions
        n_methods = len(methods)
        n_voxels = len(voxel_sizes)
        width = 0.18
        
        positions = []
        data_list = []
        color_list = []
        labels = []
        
        for v_idx, voxel in enumerate(voxel_sizes):
            for m_idx, method in enumerate(methods):
                subset = df_conv[(df_conv['voxel_size'] == voxel) & (df_conv['method'] == method)]
                data = subset[metric].values
                
                if len(data) > 0:
                    pos = v_idx * (n_methods + 1) + m_idx
                    positions.append(pos)
                    data_list.append(data)
                    color_list.append(colors[m_idx])
                    if v_idx == 0:
                        labels.append(method)
        
        # Create box plots
        bp = ax.boxplot(data_list, positions=positions, widths=width*3, patch_artist=True)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], color_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Set x-axis labels
        xtick_positions = [(i * (n_methods + 1) + (n_methods - 1) / 2) for i in range(n_voxels)]
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels([f'{v:.1f}m' for v in voxel_sizes])
        
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Add legend
        if ax_idx == 0:
            legend_patches = [plt.Rectangle((0,0),1,1, fc=c, alpha=0.7) for c in colors]
            ax.legend(legend_patches, methods, loc='upper right')
    
    plt.tight_layout()
    
    # Save
    output_path = "/home/eugene/data/KITTI/benchmark/icp_benchmark_boxplot.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    
    # Also create convergence rate plot
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    conv_rates = []
    for voxel in voxel_sizes:
        rates = []
        for method in methods:
            subset = df[(df['voxel_size'] == voxel) & (df['method'] == method)]
            rate = subset['converged'].mean() * 100
            rates.append(rate)
        conv_rates.append(rates)
    
    x = np.arange(len(voxel_sizes))
    width = 0.2
    
    for i, method in enumerate(methods):
        rates = [conv_rates[v][i] for v in range(len(voxel_sizes))]
        ax2.bar(x + i*width, rates, width, label=method, color=colors[i], alpha=0.8)
    
    ax2.set_xlabel('Voxel Size')
    ax2.set_ylabel('Convergence Rate (%)')
    ax2.set_title('Convergence Rate by Method and Voxel Size')
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels([f'{v:.1f}m' for v in voxel_sizes])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 105)
    
    output_path2 = "/home/eugene/data/KITTI/benchmark/icp_benchmark_convergence.png"
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path2}")
    
    # Print summary stats
    print("\n" + "="*60)
    print("SUMMARY STATISTICS (converged only)")
    print("="*60)
    
    for voxel in voxel_sizes:
        print(f"\nVoxel Size: {voxel:.1f}m")
        print("-" * 50)
        for method in methods:
            subset = df_conv[(df_conv['voxel_size'] == voxel) & (df_conv['method'] == method)]
            if len(subset) > 0:
                trans_med = subset['trans_error_m'].median()
                rot_med = subset['rot_error_deg'].median()
                time_med = subset['time_ms'].median()
                n = len(subset)
                total = len(df[(df['voxel_size'] == voxel) & (df['method'] == method)])
                print(f"  {method:15s}: trans={trans_med:.4f}m, rot={rot_med:.4f}°, time={time_med:.1f}ms, conv={n}/{total}")

if __name__ == "__main__":
    main()
