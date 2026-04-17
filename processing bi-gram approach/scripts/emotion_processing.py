#!/usr/bin/env python3
"""
Emotion Processing Script

This script processes CSV files containing emotional data and counts the occurrences 
of each emotion (value 1) across the 10 emotion columns: anger, fear, anticipation, 
trust, surprise, sadness, joy, disgust, negative, positive.

Author: Generated for emotional profiling analysis
Date: September 2025
"""

import pandas as pd
import numpy as np
import os
import argparse
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def process_emotion_data(csv_file):
    """
    Process a single CSV file and count emotion occurrences.
    
    Args:
        csv_file (str): Path to CSV file
    
    Returns:
        dict: Statistics and emotion counts
    """
    # Define the 10 emotion columns
    emotion_columns = ['anger', 'fear', 'anticipation', 'trust', 'surprise', 
                      'sadness', 'joy', 'disgust', 'negative', 'positive']
    
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # Check if required columns exist
    missing_cols = [col for col in emotion_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required emotion columns: {missing_cols}")
    
    # Calculate basic statistics
    total_words = len(df)
    
    # Count rows with N/A values in any of the emotion columns
    na_mask = df[emotion_columns].isin(['N/A']).any(axis=1)
    ignored_words = na_mask.sum()
    
    # Filter out rows with N/A values
    clean_df = df[~na_mask].copy()
    
    # Convert emotion columns to numeric (should be 0 or 1)
    for col in emotion_columns:
        clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')
    
    # Remove any remaining NaN values after conversion
    clean_df = clean_df.dropna(subset=emotion_columns)
    
    processed_words = len(clean_df)
    successful_words = processed_words  # All processed words are successful
    
    # Count occurrences of value 1 for each emotion
    emotion_counts = {}
    emotion_proportions = {}
    
    for emotion in emotion_columns:
        count = (clean_df[emotion] == 1).sum()
        emotion_counts[emotion] = count
        # Calculate proportion (count / total processed words)
        emotion_proportions[emotion] = count / processed_words if processed_words > 0 else 0.0
    
    # Prepare statistics dictionary
    stats = {
        'file_name': os.path.basename(csv_file),
        'total_words': total_words,
        'ignored_words': ignored_words,
        'processed_words': processed_words,
        'successful_words': successful_words,
        'emotion_counts': emotion_counts,
        'emotion_proportions': emotion_proportions
    }
    
    return stats, clean_df

def create_emotion_report(stats_list, output_dir):
    """
    Create comprehensive emotion analysis report.
    
    Args:
        stats_list (list): List of statistics dictionaries
        output_dir (str): Output directory path
    
    Returns:
        str: Path to the saved report
    """
    # Prepare data for the report
    report_data = []
    
    for stats in stats_list:
        row = {
            'file_name': stats['file_name'],
            'total_words': stats['total_words'],
            'ignored_words': stats['ignored_words'],
            'processed_words': stats['processed_words'],
            'successful_words': stats['successful_words']
        }
        
        # Add emotion counts
        for emotion, count in stats['emotion_counts'].items():
            row[f'{emotion}_count'] = count
        
        # Add emotion proportions
        for emotion, proportion in stats['emotion_proportions'].items():
            row[f'{emotion}_proportion'] = round(proportion, 4)
        
        report_data.append(row)
    
    # Create DataFrame and save
    df_report = pd.DataFrame(report_data)
    report_path = os.path.join(output_dir, 'emotion_analysis_report.csv')
    df_report.to_csv(report_path, index=False)
    
    return report_path

def create_summary_visualization(stats_list, output_dir):
    """
    Create a summary visualization of emotion distributions.
    
    Args:
        stats_list (list): List of statistics dictionaries
        output_dir (str): Output directory path
    """
    import matplotlib.pyplot as plt
    
    emotion_columns = ['anger', 'fear', 'anticipation', 'trust', 'surprise', 
                      'sadness', 'joy', 'disgust', 'negative', 'positive']
    
    # Aggregate data across all files
    total_counts = {emotion: 0 for emotion in emotion_columns}
    total_processed = 0
    
    for stats in stats_list:
        for emotion in emotion_columns:
            total_counts[emotion] += stats['emotion_counts'][emotion]
        total_processed += stats['processed_words']
    
    # Calculate overall proportions
    proportions = {emotion: count / total_processed if total_processed > 0 else 0 
                  for emotion, count in total_counts.items()}
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Emotion Counts
    emotions = list(total_counts.keys())
    counts = list(total_counts.values())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
              '#DDA0DD', '#98D8C8', '#F7DC6F', '#F1948A', '#85C1E9']
    
    bars1 = ax1.bar(emotions, counts, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
    ax1.set_title('Total Emotion Counts Across All Files', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Emotions', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, count in zip(bars1, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                f'{count:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: Emotion Proportions
    prop_values = list(proportions.values())
    bars2 = ax2.bar(emotions, prop_values, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
    ax2.set_title('Emotion Proportions Across All Files', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('Emotions', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Proportion', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_ylim(0, max(prop_values) * 1.1)
    
    # Add value labels on bars
    for bar, prop in zip(bars2, prop_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(prop_values)*0.01,
                f'{prop:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add grid for better readability
    ax1.grid(True, alpha=0.3, axis='y')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add summary statistics
    summary_text = f"""Summary Statistics:
Total Files Processed: {len(stats_list)}
Total Words Processed: {total_processed:,}
Total Emotion Instances: {sum(total_counts.values()):,}"""
    
    fig.text(0.02, 0.02, summary_text, fontsize=10, fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the visualization
    viz_path = os.path.join(output_dir, 'emotion_summary_visualization.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return viz_path

def process_single_file(csv_file, results_dir):
    """
    Process a single CSV file and create individual reports.
    
    Args:
        csv_file (str): Path to CSV file
        results_dir (str): Base results directory
    
    Returns:
        dict: Statistics for this file
    """
    try:
        # Process the file
        stats, clean_df = process_emotion_data(csv_file)
        
        if stats['processed_words'] == 0:
            print(f"⚠ Warning: No valid data found in {csv_file}")
            return stats
        
        # Create output directory structure
        file_stem = Path(csv_file).stem
        output_dir = os.path.join(results_dir, 'emotional_profiling', 'emotions', file_stem)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save individual file report
        individual_report_data = [{
            'file_name': stats['file_name'],
            'total_words': stats['total_words'],
            'ignored_words': stats['ignored_words'],
            'processed_words': stats['processed_words'],
            'successful_words': stats['successful_words'],
            **{f'{emotion}_count': count for emotion, count in stats['emotion_counts'].items()},
            **{f'{emotion}_proportion': round(prop, 4) for emotion, prop in stats['emotion_proportions'].items()}
        }]
        
        individual_df = pd.DataFrame(individual_report_data)
        individual_report_path = os.path.join(output_dir, f'{file_stem}_emotion_report.csv')
        individual_df.to_csv(individual_report_path, index=False)
        
        print(f"✓ Processed: {os.path.basename(csv_file)}")
        print(f"  - Total words: {stats['total_words']:,}")
        print(f"  - Processed: {stats['processed_words']:,}")
        print(f"  - Ignored (N/A): {stats['ignored_words']:,}")
        print(f"  - Report saved: {individual_report_path}")
        
        return stats
        
    except Exception as e:
        print(f"✗ Error processing {csv_file}: {str(e)}")
        return None

def main():
    """Main function to run the emotion processing."""
    parser = argparse.ArgumentParser(description='Process Emotion Data from CSV Files')
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
    
    print(f"📊 Processing {len(csv_files)} CSV file(s) for emotion analysis...")
    print("=" * 70)
    
    # Process each file
    stats_list = []
    for csv_file in csv_files:
        print(f"\n🔄 Processing: {os.path.basename(csv_file)}")
        stats = process_single_file(csv_file, args.results_dir)
        if stats:
            stats_list.append(stats)
    
    # Create overall reports and visualizations
    if stats_list:
        output_dir = os.path.join(args.results_dir, 'emotional_profiling', 'emotions')
        os.makedirs(output_dir, exist_ok=True)
        
        # Create comprehensive report
        report_path = create_emotion_report(stats_list, output_dir)
        print(f"\n✓ Comprehensive report saved: {report_path}")
        
        # Create summary visualization
        viz_path = create_summary_visualization(stats_list, output_dir)
        print(f"✓ Summary visualization saved: {viz_path}")
        
        # Print overall summary
        print("\n" + "="*70)
        print("📈 EMOTION PROCESSING SUMMARY")
        print("="*70)
        
        total_words = sum(s['total_words'] for s in stats_list)
        total_ignored = sum(s['ignored_words'] for s in stats_list)
        total_processed = sum(s['processed_words'] for s in stats_list)
        
        print(f"Files processed: {len(stats_list)}")
        print(f"Total words: {total_words:,}")
        print(f"Words ignored (N/A): {total_ignored:,} ({total_ignored/total_words*100:.1f}%)")
        print(f"Words processed: {total_processed:,} ({total_processed/total_words*100:.1f}%)")
        
        # Show emotion summary
        emotion_columns = ['anger', 'fear', 'anticipation', 'trust', 'surprise', 
                          'sadness', 'joy', 'disgust', 'negative', 'positive']
        
        print(f"\n📊 EMOTION DISTRIBUTION:")
        print("-" * 50)
        
        total_emotion_counts = {emotion: sum(s['emotion_counts'][emotion] for s in stats_list) 
                               for emotion in emotion_columns}
        
        for emotion, count in total_emotion_counts.items():
            proportion = count / total_processed if total_processed > 0 else 0
            print(f"{emotion.capitalize():>12}: {count:>6,} ({proportion:>6.1%})")
        
        print("="*70)

if __name__ == "__main__":
    main()
