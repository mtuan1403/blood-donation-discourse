#!/usr/bin/env python3
"""
Quick verification script for integrated community data
"""

import pandas as pd
import os

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Read integrated files
    comments_file = os.path.join(base_dir, "results", "integrated", "comments_nodes_with_communities.csv")
    submissions_file = os.path.join(base_dir, "results", "integrated", "submissions_nodes_with_communities.csv")
    
    print("🔍 VERIFICATION OF INTEGRATED DATA")
    print("=" * 50)
    
    if os.path.exists(comments_file):
        comments_df = pd.read_csv(comments_file)
        print(f"\n📝 COMMENTS DATA:")
        print(f"   • Total nodes: {len(comments_df):,}")
        print(f"   • Communities: {comments_df['community_id'].min()}-{comments_df['community_id'].max()} ({comments_df['community_id'].nunique()} total)")
        print(f"   • Largest community: {comments_df['community_id'].value_counts().iloc[0]} nodes (Community {comments_df['community_id'].value_counts().index[0]})")
        print(f"   • Smallest community: {comments_df['community_id'].value_counts().iloc[-1]} nodes")
        print(f"   • Average community size: {len(comments_df) / comments_df['community_id'].nunique():.1f} nodes")
        
        # Show some examples
        print(f"\n   📋 Sample nodes from different communities:")
        for i in range(min(3, comments_df['community_id'].nunique())):
            comm_id = sorted(comments_df['community_id'].unique())[i]
            sample_nodes = comments_df[comments_df['community_id'] == comm_id]['Label'].head(3).tolist()
            print(f"      Community {comm_id}: {', '.join(sample_nodes)}")
    
    if os.path.exists(submissions_file):
        submissions_df = pd.read_csv(submissions_file)
        print(f"\n📊 SUBMISSIONS DATA:")
        print(f"   • Total nodes: {len(submissions_df):,}")
        print(f"   • Communities: {submissions_df['community_id'].min()}-{submissions_df['community_id'].max()} ({submissions_df['community_id'].nunique()} total)")
        print(f"   • Largest community: {submissions_df['community_id'].value_counts().iloc[0]} nodes (Community {submissions_df['community_id'].value_counts().index[0]})")
        print(f"   • Smallest community: {submissions_df['community_id'].value_counts().iloc[-1]} nodes")
        print(f"   • Average community size: {len(submissions_df) / submissions_df['community_id'].nunique():.1f} nodes")
        
        # Show some examples
        print(f"\n   📋 Sample nodes from different communities:")
        for i in range(min(3, submissions_df['community_id'].nunique())):
            comm_id = sorted(submissions_df['community_id'].unique())[i]
            sample_nodes = submissions_df[submissions_df['community_id'] == comm_id]['Label'].head(3).tolist()
            print(f"      Community {comm_id}: {', '.join(sample_nodes)}")
    
    print("\n✅ Integration verification completed!")
    print("\n📁 Files created:")
    print(f"   • {comments_file}")
    print(f"   • {submissions_file}")

if __name__ == "__main__":
    main()
