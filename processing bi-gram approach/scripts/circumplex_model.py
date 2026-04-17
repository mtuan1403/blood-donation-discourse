#!/usr/bin/env python3
"""
Circumplex Emotional Profiling Visualization Script

This script creates 2D visualizations of emotional data using the Circumplex Model,
plotting valence (x-axis) vs arousal (y-axis) with color-coded quadrants and smooth transitions.

Author: Generated for emotional profiling analysis
Date: September 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import glob
from pathlib import Path
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

def create_custom_colormap():
    """
    Create a custom colormap for the four quadrants of the circumplex model:
    - Top-right (positive valence, positive arousal): Green pastel
    - Top-left (negative valence, positive arousal): Yellow pastel  
    - Bottom-left (negative valence, negative arousal): Red pastel
    - Bottom-right (positive valence, negative arousal): Blue pastel
    """
    # Define colors for each quadrant (in RGBA format)
    colors = {
        'green_pastel': [0.6, 0.9, 0.6, 0.8],    # Top-right quadrant
        'yellow_pastel': [0.9, 0.9, 0.6, 0.8],   # Top-left quadrant
        'red_pastel': [0.9, 0.6, 0.6, 0.8],      # Bottom-left quadrant
        'blue_pastel': [0.6, 0.6, 0.9, 0.8]      # Bottom-right quadrant
    }
    return colors

def get_point_color(valence, arousal, intensity_factor=1.0):
    """
    Calculate color for a point based on valence and arousal values.
    
    Args:
        valence (float): Valence value (-1 to 1)
        arousal (float): Arousal value (-1 to 1)
        intensity_factor (float): Factor to control color intensity
    
    Returns:
        tuple: RGBA color values
    """
    colors = create_custom_colormap()
    
    # Normalize values to 0-1 range for easier calculation
    norm_valence = (valence + 1) / 2  # Convert from [-1,1] to [0,1]
    norm_arousal = (arousal + 1) / 2   # Convert from [-1,1] to [0,1]
    
    # Calculate distance from center (0,0) for intensity
    distance_from_center = np.sqrt(valence**2 + arousal**2)
    max_distance = np.sqrt(2)  # Maximum possible distance
    intensity = min(distance_from_center / max_distance * intensity_factor, 1.0)
    
    # Determine base color based on quadrant
    if valence >= 0 and arousal >= 0:  # Top-right: Green
        base_color = colors['green_pastel']
    elif valence < 0 and arousal >= 0:  # Top-left: Yellow
        base_color = colors['yellow_pastel']
    elif valence < 0 and arousal < 0:   # Bottom-left: Red
        base_color = colors['red_pastel']
    else:  # Bottom-right: Blue
        base_color = colors['blue_pastel']
    
    # Apply intensity to make colors lighter/darker based on distance from center
    final_color = [
        base_color[0] * (0.3 + 0.7 * intensity),  # R
        base_color[1] * (0.3 + 0.7 * intensity),  # G
        base_color[2] * (0.3 + 0.7 * intensity),  # B
        base_color[3] * (0.5 + 0.5 * intensity)   # A
    ]
    
    return final_color

def load_and_process_data(csv_file):
    """
    Load CSV file and process emotional data.
    
    Args:
        csv_file (str): Path to CSV file
    
    Returns:
        tuple: (processed_df, stats_dict)
    """
    df = pd.read_csv(csv_file)
    
    # Check if required columns exist
    required_cols = ['valence', 'arousal', 'dominance']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Calculate statistics
    total_rows = len(df)
    
    # Count N/A values in any of the three columns
    na_mask = (df['valence'] == 'N/A') | (df['arousal'] == 'N/A') | (df['dominance'] == 'N/A')
    na_count = na_mask.sum()
    
    # Filter out N/A values
    clean_df = df[~na_mask].copy()
    
    # Convert to numeric
    clean_df['valence'] = pd.to_numeric(clean_df['valence'], errors='coerce')
    clean_df['arousal'] = pd.to_numeric(clean_df['arousal'], errors='coerce')
    clean_df['dominance'] = pd.to_numeric(clean_df['dominance'], errors='coerce')
    
    # Remove any remaining NaN values after conversion
    clean_df = clean_df.dropna(subset=['valence', 'arousal', 'dominance'])
    
    processed_count = len(clean_df)
    success_count = processed_count  # All processed rows are successful
    
    stats = {
        'total_words': total_rows,
        'ignored_words': na_count,
        'processed_words': processed_count,
        'successful_words': success_count,
        'file_name': os.path.basename(csv_file)
    }
    
    return clean_df, stats

def create_quadrant_colormap():
    """Create custom colormaps for each quadrant."""
    # Define the base colors for each quadrant
    colors = {
        'excited': '#90EE90',    # Light green for positive valence, positive arousal
        'alarmed': '#FFD700',    # Gold/yellow for negative valence, positive arousal
        'lethargic': '#FF6B6B',  # Light red for negative valence, negative arousal
        'serene': '#87CEEB'      # Sky blue for positive valence, negative arousal
    }
    return colors

def create_density_surface(valence_vals, arousal_vals, grid_size=100):
    """
    Create density surface for contour plotting using 2D histogram.
    
    Args:
        valence_vals (array): Valence values
        arousal_vals (array): Arousal values  
        grid_size (int): Resolution of the grid
    
    Returns:
        tuple: (X, Y, Z) meshgrid and density values
    """
    # Define the range for the grid
    x_min, x_max = valence_vals.min() - 0.1, valence_vals.max() + 0.1
    y_min, y_max = arousal_vals.min() - 0.1, arousal_vals.max() + 0.1
    
    # Create 2D histogram for density estimation
    H, xedges, yedges = np.histogram2d(valence_vals, arousal_vals, 
                                      bins=grid_size, 
                                      range=[[x_min, x_max], [y_min, y_max]])
    
    # Create coordinate meshgrid
    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2
    X, Y = np.meshgrid(x_centers, y_centers)
    
    # Transpose H to match meshgrid orientation
    Z = H.T
    
    # Apply simple smoothing by convolution with a 2D Gaussian-like kernel
    # Create a simple smoothing kernel
    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    
    # Apply simple smoothing manually
    Z_smooth = np.copy(Z)
    for i in range(1, Z.shape[0]-1):
        for j in range(1, Z.shape[1]-1):
            Z_smooth[i,j] = np.mean(Z[i-1:i+2, j-1:j+2])
    
    return X, Y, Z_smooth

def create_circumplex_plot(df, output_path, file_name, stats):
    """
    Create the circumplex emotional profiling density plot.
    
    Args:
        df (DataFrame): Processed data
        output_path (str): Path to save the plot
        file_name (str): Name of the input file
        stats (dict): Statistics dictionary
    """
    # Create figure with circular layout
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection=None)
    
    # Prepare data for plotting
    valence_vals = df['valence'].values
    arousal_vals = df['arousal'].values
    
    # Create density surface
    X, Y, Z = create_density_surface(valence_vals, arousal_vals)
    
    # Get quadrant colors
    quad_colors = create_quadrant_colormap()
    
    # Create the main density contour plot
    levels = np.linspace(Z.min(), Z.max(), 15)
    
    # Create filled contours with custom colors based on quadrants
    contour_filled = ax.contourf(X, Y, Z, levels=levels, alpha=0.6, extend='both')
    
    # Create contour lines
    contour_lines = ax.contour(X, Y, Z, levels=levels[::2], colors='white', alpha=0.4, linewidths=0.8)
    
    # Customize the colormap based on quadrants
    # This creates a smooth transition between quadrant colors
    colors_list = []
    for i in range(len(levels)-1):
        # Sample points to determine dominant quadrant
        mid_level = (levels[i] + levels[i+1]) / 2
        mask = (Z >= levels[i]) & (Z < levels[i+1])
        if np.any(mask):
            # Find average position of this density level
            y_pos, x_pos = np.where(mask)
            avg_x = X[y_pos, x_pos].mean()
            avg_y = Y[y_pos, x_pos].mean()
            
            # Determine quadrant and set color
            if avg_x >= 0 and avg_y >= 0:  # Excited
                colors_list.append(quad_colors['excited'])
            elif avg_x < 0 and avg_y >= 0:  # Alarmed
                colors_list.append(quad_colors['alarmed'])
            elif avg_x < 0 and avg_y < 0:   # Lethargic
                colors_list.append(quad_colors['lethargic'])
            else:  # Serene
                colors_list.append(quad_colors['serene'])
        else:
            colors_list.append('#FFFFFF')
    
    # Apply custom colors to contours
    for i, collection in enumerate(contour_filled.collections):
        if i < len(colors_list):
            collection.set_facecolor(colors_list[i])
    
    # Add central neutral range overlay (like in reference image)
    neutral_circle = plt.Circle((0, 0), 0.15, color='gray', alpha=0.3, zorder=10)
    ax.add_patch(neutral_circle)
    ax.text(0, 0, 'Neutral\nRange', ha='center', va='center', fontsize=10, 
            fontweight='bold', color='white', zorder=11)
    
    # Create circular boundary (like in reference image)
    circle_boundary = plt.Circle((0, 0), max(abs(valence_vals.max()), abs(valence_vals.min()),
                                             abs(arousal_vals.max()), abs(arousal_vals.min())) * 1.1, 
                                fill=False, color='black', linewidth=2, alpha=0.7)
    ax.add_patch(circle_boundary)
    
    # Add quadrant lines
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1.5, zorder=5)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=1.5, zorder=5)
    
    # Set equal aspect ratio for circular appearance
    ax.set_aspect('equal')
    
    # Set axis limits
    max_range = max(abs(valence_vals.max()), abs(valence_vals.min()),
                   abs(arousal_vals.max()), abs(arousal_vals.min())) * 1.2
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    
    # Add quadrant labels around the circle
    label_radius = max_range * 0.85
    
    # Top (Aroused)
    ax.text(0, label_radius, 'Aroused', ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Right (Positive)
    ax.text(label_radius, 0, 'Positive', ha='center', va='center', fontsize=14, fontweight='bold', rotation=90)
    
    # Bottom (Lethargic)
    ax.text(0, -label_radius, 'Lethargic', ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Left (Negative)
    ax.text(-label_radius, 0, 'Negative', ha='center', va='center', fontsize=14, fontweight='bold', rotation=90)
    
    # Add diagonal quadrant labels
    diag_radius = label_radius * 0.7
    
    # Top-right (Excited)
    ax.text(diag_radius * 0.7, diag_radius * 0.7, 'Excited', ha='center', va='center', 
            fontsize=12, fontweight='bold', color='darkgreen')
    
    # Top-left (Alarmed)
    ax.text(-diag_radius * 0.7, diag_radius * 0.7, 'Alarmed', ha='center', va='center', 
            fontsize=12, fontweight='bold', color='darkorange')
    
    # Bottom-left (Bored)
    ax.text(-diag_radius * 0.7, -diag_radius * 0.7, 'Bored', ha='center', va='center', 
            fontsize=12, fontweight='bold', color='darkred')
    
    # Bottom-right (Serene)
    ax.text(diag_radius * 0.7, -diag_radius * 0.7, 'Serene', ha='center', va='center', 
            fontsize=12, fontweight='bold', color='darkblue')
    
    # Remove axis ticks and labels for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add title
    ax.set_title(f'Circumplex Emotional Profiling\n{file_name}', 
                fontsize=16, fontweight='bold', pad=30)
    
    # Add statistics text box
    stats_text = f"""Statistics:
Total words: {stats['total_words']:,}
Ignored (N/A): {stats['ignored_words']:,}
Processed: {stats['processed_words']:,}
Successful: {stats['successful_words']:,}"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9),
            fontsize=10, fontfamily='monospace', zorder=15)
    
    # Make the background clean
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved visualization: {output_path}")

def save_statistics_report(stats_list, output_dir):
    """
    Save statistics report to CSV file.
    
    Args:
        stats_list (list): List of statistics dictionaries
        output_dir (str): Output directory path
    """
    df_stats = pd.DataFrame(stats_list)
    report_path = os.path.join(output_dir, 'circumplex_visualization_report.csv')
    df_stats.to_csv(report_path, index=False)
    print(f"✓ Saved statistics report: {report_path}")
    return report_path

def process_single_file(csv_file, results_dir):
    """
    Process a single CSV file and create visualization.
    
    Args:
        csv_file (str): Path to CSV file
        results_dir (str): Base results directory
    
    Returns:
        dict: Statistics for this file
    """
    try:
        # Load and process data
        df, stats = load_and_process_data(csv_file)
        
        if len(df) == 0:
            print(f"⚠ Warning: No valid data found in {csv_file}")
            return stats
        
        # Create output directory structure
        file_stem = Path(csv_file).stem
        output_dir = os.path.join(results_dir, 'emotional_profiling', 'emotional_visualisation', file_stem)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create visualization
        output_path = os.path.join(output_dir, f'{file_stem}_circumplex.png')
        create_circumplex_plot(df, output_path, stats['file_name'], stats)
        
        return stats
        
    except Exception as e:
        print(f"✗ Error processing {csv_file}: {str(e)}")
        return None

def main():
    """Main function to run the circumplex visualization."""
    parser = argparse.ArgumentParser(description='Create Circumplex Emotional Profiling Visualizations')
    parser.add_argument('input_files', nargs='+', 
                       help='CSV file(s) to process (can use wildcards)')
    parser.add_argument('--results-dir', 
                       default='/Users/ntuan0314/Desktop/processing bi-gram approach/results',
                       help='Base results directory (default: current project results)')
    
    args = parser.parse_args()
    
    # Expand wildcards in input files
    all_files = []
    for pattern in args.input_files:
        if '*' in pattern or '?' in pattern:
            all_files.extend(glob.glob(pattern))
        else:
            all_files.append(pattern)
    
    # Remove duplicates and filter existing files
    csv_files = list(set([f for f in all_files if os.path.exists(f) and f.endswith('.csv')]))
    
    if not csv_files:
        print("✗ No valid CSV files found!")
        return
    
    print(f"📊 Processing {len(csv_files)} CSV file(s)...")
    print("-" * 60)
    
    # Process each file
    stats_list = []
    for csv_file in csv_files:
        print(f"🔄 Processing: {os.path.basename(csv_file)}")
        stats = process_single_file(csv_file, args.results_dir)
        if stats:
            stats_list.append(stats)
    
    # Save overall statistics report
    if stats_list:
        output_dir = os.path.join(args.results_dir, 'emotional_profiling', 'emotional_visualisation')
        save_statistics_report(stats_list, output_dir)
        
        # Print summary
        print("\n" + "="*60)
        print("📈 PROCESSING SUMMARY")
        print("="*60)
        total_words = sum(s['total_words'] for s in stats_list)
        total_ignored = sum(s['ignored_words'] for s in stats_list)
        total_processed = sum(s['processed_words'] for s in stats_list)
        
        print(f"Files processed: {len(stats_list)}")
        print(f"Total words: {total_words:,}")
        print(f"Words ignored (N/A): {total_ignored:,} ({total_ignored/total_words*100:.1f}%)")
        print(f"Words processed: {total_processed:,} ({total_processed/total_words*100:.1f}%)")
        print("="*60)

if __name__ == "__main__":
    main()
