"""
Network Construction Module
Converts co-occurrence matrices to GraphML network files with thresholding.
"""

import pandas as pd
import networkx as nx
from pathlib import Path
import os
from tqdm import tqdm

def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent

def load_cooccurrence_pairs(pairs_file):
    """
    Load co-occurrence pairs from CSV file.
    
    Args:
        pairs_file (str/Path): Path to co-occurrence pairs CSV file
        
    Returns:
        pd.DataFrame: DataFrame with word1, word2, co_occurrence columns
    """
    print(f"Loading co-occurrence pairs from {pairs_file}")
    df = pd.read_csv(pairs_file)
    print(f"Loaded {len(df)} co-occurrence pairs")
    return df

def apply_threshold_filter(df_pairs, min_frequency=3):
    """
    Apply frequency threshold to filter out low-frequency edges.
    
    Args:
        df_pairs (pd.DataFrame): Co-occurrence pairs DataFrame
        min_frequency (int): Minimum co-occurrence frequency to keep
        
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    print(f"Applying threshold filter (min_frequency >= {min_frequency})")
    
    initial_count = len(df_pairs)
    df_filtered = df_pairs[df_pairs['co_occurrence'] >= min_frequency].copy()
    final_count = len(df_filtered)
    
    print(f"Kept {final_count:,} out of {initial_count:,} pairs ({final_count/initial_count*100:.1f}%)")
    
    # Show statistics
    if final_count > 0:
        print(f"Frequency range after filtering: {df_filtered['co_occurrence'].min()} - {df_filtered['co_occurrence'].max()}")
        print(f"Mean frequency: {df_filtered['co_occurrence'].mean():.2f}")
    
    return df_filtered

def construct_network_from_pairs(df_pairs):
    """
    Construct NetworkX graph from co-occurrence pairs.
    
    Args:
        df_pairs (pd.DataFrame): Filtered co-occurrence pairs
        
    Returns:
        nx.Graph: Undirected graph with weighted edges
    """
    print("Constructing NetworkX graph...")
    
    # Create undirected graph
    G = nx.Graph()
    
    # Add edges with weights
    for _, row in tqdm(df_pairs.iterrows(), total=len(df_pairs), desc="Adding edges"):
        word1 = row['word1']
        word2 = row['word2']
        weight = row['co_occurrence']
        
        G.add_edge(word1, word2, weight=weight)
    
    print(f"Graph constructed with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    return G

def calculate_basic_network_stats(G):
    """
    Calculate and print basic network statistics.
    
    Args:
        G (nx.Graph): NetworkX graph
    """
    print("\n" + "=" * 40)
    print("NETWORK STATISTICS")
    print("=" * 40)
    
    print(f"Nodes: {G.number_of_nodes():,}")
    print(f"Edges: {G.number_of_edges():,}")
    
    if G.number_of_nodes() > 0:
        density = nx.density(G)
        print(f"Density: {density:.6f}")
        
        # Degree statistics
        degrees = dict(G.degree())
        degree_values = list(degrees.values())
        print(f"Average degree: {sum(degree_values)/len(degree_values):.2f}")
        print(f"Max degree: {max(degree_values)}")
        print(f"Min degree: {min(degree_values)}")
        
        # Connected components
        num_components = nx.number_connected_components(G)
        print(f"Connected components: {num_components}")
        
        if num_components > 0:
            largest_cc = max(nx.connected_components(G), key=len)
            print(f"Largest component size: {len(largest_cc)} nodes ({len(largest_cc)/G.number_of_nodes()*100:.1f}%)")
        
        # Weight statistics
        weights = [data['weight'] for _, _, data in G.edges(data=True)]
        if weights:
            print(f"Edge weight range: {min(weights)} - {max(weights)}")
            print(f"Average edge weight: {sum(weights)/len(weights):.2f}")

def save_graphml(G, output_file):
    """
    Save NetworkX graph as GraphML file.
    
    Args:
        G (nx.Graph): NetworkX graph
        output_file (str/Path): Output GraphML file path
    """
    print(f"Saving graph to {output_file}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save as GraphML
    nx.write_graphml(G, output_file)
    print(f"GraphML file saved successfully")

def process_comments_network(min_frequency=3):
    """
    Process comments co-occurrence pairs to create network.
    
    Args:
        min_frequency (int): Minimum co-occurrence frequency threshold
    """
    print("=" * 60)
    print("PROCESSING COMMENTS NETWORK")
    print("=" * 60)
    
    # File paths
    project_root = get_project_root()
    pairs_file = project_root / "results" / "matrix" / "comments_cooccurrence_pairs.csv"
    output_file = project_root / "results" / "graph" / "comments_network.graphml"
    
    # Load pairs
    df_pairs = load_cooccurrence_pairs(pairs_file)
    
    # Apply threshold
    df_filtered = apply_threshold_filter(df_pairs, min_frequency)
    
    if len(df_filtered) == 0:
        print("Warning: No pairs left after filtering! Consider lowering the threshold.")
        return None
    
    # Construct network
    G = construct_network_from_pairs(df_filtered)
    
    # Calculate statistics
    calculate_basic_network_stats(G)
    
    # Save GraphML
    save_graphml(G, output_file)
    
    print(f"\nComments network processing complete!")
    print(f"GraphML output: {output_file}")
    
    return G

def process_submissions_network(min_frequency=3):
    """
    Process submissions co-occurrence pairs to create network.
    
    Args:
        min_frequency (int): Minimum co-occurrence frequency threshold
    """
    print("=" * 60)
    print("PROCESSING SUBMISSIONS NETWORK")
    print("=" * 60)
    
    # File paths
    project_root = get_project_root()
    pairs_file = project_root / "results" / "matrix" / "submissions_cooccurrence_pairs.csv"
    output_file = project_root / "results" / "graph" / "submissions_network.graphml"
    
    # Load pairs
    df_pairs = load_cooccurrence_pairs(pairs_file)
    
    # Apply threshold
    df_filtered = apply_threshold_filter(df_pairs, min_frequency)
    
    if len(df_filtered) == 0:
        print("Warning: No pairs left after filtering! Consider lowering the threshold.")
        return None
    
    # Construct network
    G = construct_network_from_pairs(df_filtered)
    
    # Calculate statistics
    calculate_basic_network_stats(G)
    
    # Save GraphML
    save_graphml(G, output_file)
    
    print(f"\nSubmissions network processing complete!")
    print(f"GraphML output: {output_file}")
    
    return G

def find_optimal_threshold(pairs_file, target_edges_range=(1000, 10000)):
    """
    Find optimal threshold to get network in target edge range.
    
    Args:
        pairs_file (str/Path): Path to co-occurrence pairs CSV
        target_edges_range (tuple): (min_edges, max_edges) target range
        
    Returns:
        int: Suggested threshold
    """
    print(f"Finding optimal threshold for {pairs_file}")
    df = pd.read_csv(pairs_file)
    
    print(f"Testing different thresholds:")
    print(f"Target range: {target_edges_range[0]:,} - {target_edges_range[1]:,} edges")
    
    min_target, max_target = target_edges_range
    
    # Test different thresholds
    thresholds = [1, 2, 3, 5, 10, 15, 20, 30, 50]
    
    for threshold in thresholds:
        filtered_count = len(df[df['co_occurrence'] >= threshold])
        print(f"  Threshold >= {threshold}: {filtered_count:,} edges")
        
        if min_target <= filtered_count <= max_target:
            print(f"  → Optimal threshold found: {threshold}")
            return threshold
    
    # If no optimal found, suggest based on closest to middle of range
    target_middle = (min_target + max_target) // 2
    best_threshold = 3  # default
    best_diff = float('inf')
    
    for threshold in thresholds:
        filtered_count = len(df[df['co_occurrence'] >= threshold])
        diff = abs(filtered_count - target_middle)
        if diff < best_diff:
            best_diff = diff
            best_threshold = threshold
    
    print(f"  → Suggested threshold (closest to target): {best_threshold}")
    return best_threshold

def main():
    """Main network construction function"""
    print("Network Construction Pipeline")
    print("=" * 50)
    
    try:
        # Find optimal thresholds
        project_root = get_project_root()
        
        print("Finding optimal thresholds...")
        comments_pairs = project_root / "results" / "matrix" / "comments_cooccurrence_pairs.csv"
        submissions_pairs = project_root / "results" / "matrix" / "submissions_cooccurrence_pairs.csv"
        
        if comments_pairs.exists():
            comments_threshold = find_optimal_threshold(comments_pairs)
        else:
            comments_threshold = 3
            print(f"Comments pairs file not found, using default threshold: {comments_threshold}")
        
        if submissions_pairs.exists():
            submissions_threshold = find_optimal_threshold(submissions_pairs)
        else:
            submissions_threshold = 3
            print(f"Submissions pairs file not found, using default threshold: {submissions_threshold}")
        
        print("\n" + "=" * 70)
        
        # Process comments network
        comments_graph = process_comments_network(min_frequency=comments_threshold)
        
        print("\n" + "=" * 70)
        
        # Process submissions network
        submissions_graph = process_submissions_network(min_frequency=submissions_threshold)
        
        print("\n" + "=" * 70)
        print("NETWORK CONSTRUCTION COMPLETE!")
        print("=" * 70)
        print("Output files created:")
        print("  - comments_network.graphml")
        print("  - submissions_network.graphml")
        
    except Exception as e:
        print(f"Error during network construction: {e}")
        raise

if __name__ == "__main__":
    main()

