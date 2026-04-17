#!/usr/bin/env python3
"""
Community Integration Script

This script integrates community detection results by creating unified CSV files 
that include all nodes with their respective community IDs.

Features:
- Processes both comments and submissions data
- Adds community_id column to node data
- Handles different numbers of communities for each data type
- Creates comprehensive integrated CSV files
- Provides detailed statistics and validation

Author: Data Analysis Script
Date: 2025
"""

import pandas as pd
import os
import glob
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_community_numbers(base_path):
    """
    Get the list of community numbers from directory structure.
    
    Args:
        base_path (str): Path to the communities directory
        
    Returns:
        list: Sorted list of community numbers
    """
    community_dirs = glob.glob(os.path.join(base_path, "community_*"))
    community_numbers = []
    
    for dir_path in community_dirs:
        dir_name = os.path.basename(dir_path)
        if dir_name.startswith("community_"):
            try:
                num = int(dir_name.split("_")[1])
                community_numbers.append(num)
            except (ValueError, IndexError):
                logger.warning(f"Skipping invalid directory name: {dir_name}")
    
    return sorted(community_numbers)

def integrate_community_nodes(data_type, base_dir):
    """
    Integrate all community nodes for a specific data type (comments or submissions).
    
    Args:
        data_type (str): Either 'comments' or 'submissions'
        base_dir (str): Base directory path
        
    Returns:
        pd.DataFrame: Integrated dataframe with community_id column
    """
    logger.info(f"Starting integration for {data_type}")
    
    # Define paths
    communities_path = os.path.join(base_dir, "results", "communities", data_type)
    
    # Get community numbers
    community_numbers = get_community_numbers(communities_path)
    logger.info(f"Found {len(community_numbers)} communities for {data_type}: {min(community_numbers)}-{max(community_numbers)}")
    
    # Initialize list to store dataframes
    all_dfs = []
    
    # Process each community
    for community_id in community_numbers:
        community_dir = os.path.join(communities_path, f"community_{community_id}")
        nodes_file = os.path.join(community_dir, f"community_{community_id}_nodes.csv")
        
        if os.path.exists(nodes_file):
            try:
                # Read the community nodes file
                df = pd.read_csv(nodes_file)
                
                # Add community_id column
                df['community_id'] = community_id
                
                # Reorder columns to put community_id after Label (if it exists) or at the beginning
                cols = df.columns.tolist()
                if 'community_id' in cols:
                    cols.remove('community_id')
                
                # Find the best position for community_id
                if 'Label' in cols:
                    label_idx = cols.index('Label')
                    cols.insert(label_idx + 1, 'community_id')
                elif 'Id' in cols:
                    id_idx = cols.index('Id')
                    cols.insert(id_idx + 1, 'community_id')
                else:
                    cols.insert(0, 'community_id')
                
                df = df[cols]
                all_dfs.append(df)
                
                logger.info(f"Processed community {community_id}: {len(df)} nodes")
                
            except Exception as e:
                logger.error(f"Error processing community {community_id}: {e}")
        else:
            logger.warning(f"Nodes file not found: {nodes_file}")
    
    if not all_dfs:
        logger.error(f"No valid community data found for {data_type}")
        return None
    
    # Concatenate all dataframes
    integrated_df = pd.concat(all_dfs, ignore_index=True)
    
    logger.info(f"Integration complete for {data_type}: {len(integrated_df)} total nodes across {len(community_numbers)} communities")
    
    return integrated_df

def save_integrated_data(df, output_path, data_type):
    """
    Save integrated dataframe to CSV and provide statistics.
    
    Args:
        df (pd.DataFrame): Integrated dataframe
        output_path (str): Path to save the CSV file
        data_type (str): Type of data (comments/submissions)
    """
    if df is None:
        logger.error(f"No data to save for {data_type}")
        return
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Saved integrated {data_type} data to: {output_path}")
    
    # Print statistics
    print(f"\n=== {data_type.upper()} INTEGRATION STATISTICS ===")
    print(f"Total nodes: {len(df):,}")
    print(f"Total communities: {df['community_id'].nunique()}")
    print(f"Community size distribution:")
    community_sizes = df['community_id'].value_counts().sort_index()
    print(community_sizes.describe())
    
    print(f"\nCommunity breakdown:")
    for comm_id in sorted(df['community_id'].unique()):
        count = len(df[df['community_id'] == comm_id])
        print(f"  Community {comm_id}: {count:,} nodes")
    
    print(f"\nColumn names: {list(df.columns)}")
    print(f"Sample data:")
    print(df.head(3).to_string())

def validate_integration(original_path, integrated_df, data_type):
    """
    Validate that the integration preserved all original data.
    
    Args:
        original_path (str): Path to original nodes file
        integrated_df (pd.DataFrame): Integrated dataframe
        data_type (str): Type of data
    """
    if not os.path.exists(original_path):
        logger.warning(f"Original file not found for validation: {original_path}")
        return
    
    try:
        original_df = pd.read_csv(original_path)
        
        print(f"\n=== {data_type.upper()} VALIDATION ===")
        print(f"Original nodes: {len(original_df):,}")
        print(f"Integrated nodes: {len(integrated_df):,}")
        
        # Check if we have the same or more nodes (communities might have overlapping nodes)
        if len(integrated_df) >= len(original_df):
            print(f"✓ Integration successful - all nodes preserved or expanded")
        else:
            print(f"⚠ Warning: Integrated data has fewer nodes than original")
        
        # Check for unique nodes
        if 'Id' in integrated_df.columns and 'Id' in original_df.columns:
            original_ids = set(original_df['Id'].unique())
            integrated_ids = set(integrated_df['Id'].unique())
            
            missing_ids = original_ids - integrated_ids
            extra_ids = integrated_ids - original_ids
            
            if missing_ids:
                print(f"⚠ Missing {len(missing_ids)} node IDs from integration")
            if extra_ids:
                print(f"ℹ Found {len(extra_ids)} additional node IDs in communities")
                
        print(f"✓ Validation complete")
        
    except Exception as e:
        logger.error(f"Error during validation: {e}")

def main():
    """
    Main function to integrate community data for both comments and submissions.
    """
    print("🔄 Starting Community Integration Process")
    print("=" * 60)
    
    # Define base directory (current script directory)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define output directory
    output_dir = os.path.join(base_dir, "results", "integrated")
    os.makedirs(output_dir, exist_ok=True)
    
    # Process both data types
    data_types = ['comments', 'submissions']
    
    for data_type in data_types:
        print(f"\n{'='*20} PROCESSING {data_type.upper()} {'='*20}")
        
        # Integrate community nodes
        integrated_df = integrate_community_nodes(data_type, base_dir)
        
        if integrated_df is not None:
            # Define output path
            output_path = os.path.join(output_dir, f"{data_type}_nodes_with_communities.csv")
            
            # Save integrated data
            save_integrated_data(integrated_df, output_path, data_type)
            
            # Validate against original data
            original_path = os.path.join(base_dir, "results", "table", f"{data_type}_nodes.csv")
            validate_integration(original_path, integrated_df, data_type)
        
        print(f"{'='*20} COMPLETED {data_type.upper()} {'='*20}")
    
    print(f"\n🎉 Community integration process completed!")
    print(f"📁 Output files saved in: {output_dir}")
    print("\n📊 Summary of output files:")
    
    # List output files
    if os.path.exists(output_dir):
        output_files = os.listdir(output_dir)
        for file in sorted(output_files):
            if file.endswith('.csv'):
                file_path = os.path.join(output_dir, file)
                file_size = os.path.getsize(file_path)
                print(f"  • {file} ({file_size:,} bytes)")

if __name__ == "__main__":
    main()
