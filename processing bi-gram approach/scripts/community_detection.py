"""
Community Detection and Network Analysis Module
Implements Louvain method for community detection and generates files for Cytoscape visualization.
"""

import pandas as pd
import networkx as nx
from pathlib import Path
import os
from collections import defaultdict, Counter
from tqdm import tqdm
import community as community_louvain  # pip install python-louvain
import math

def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent

def load_network(graphml_file):
    """
    Load NetworkX graph from GraphML file.
    
    Args:
        graphml_file (str/Path): Path to GraphML file
        
    Returns:
        nx.Graph: Loaded graph
    """
    print(f"Loading network from {graphml_file}")
    G = nx.read_graphml(graphml_file)
    print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G

def detect_communities_louvain(G, resolution=1.0, random_state=42):
    """
    Detect communities using Louvain method.
    
    Args:
        G (nx.Graph): NetworkX graph
        resolution (float): Resolution parameter for modularity optimization
        random_state (int): Random seed for reproducibility
        
    Returns:
        dict: Node to community mapping
        float: Modularity score
    """
    print(f"Detecting communities using Louvain method (resolution={resolution})")
    
    # Detect communities
    partition = community_louvain.best_partition(G, resolution=resolution, random_state=random_state)
    
    # Calculate modularity
    modularity = community_louvain.modularity(partition, G)
    
    num_communities = len(set(partition.values()))
    print(f"Found {num_communities} communities with modularity score: {modularity:.4f}")
    
    return partition, modularity

def calculate_centrality_measures(G):
    """
    Calculate various centrality measures for all nodes.
    
    Args:
        G (nx.Graph): NetworkX graph
        
    Returns:
        dict: Dictionary with centrality measures for each node
    """
    print("Calculating centrality measures...")
    
    centrality_data = {}
    
    # Degree centrality
    print("  - Degree centrality")
    degree_cent = nx.degree_centrality(G)
    
    # Closeness centrality
    print("  - Closeness centrality")
    closeness_cent = nx.closeness_centrality(G)
    
    # Betweenness centrality
    print("  - Betweenness centrality")
    betweenness_cent = nx.betweenness_centrality(G)
    
    # Eigenvector centrality
    print("  - Eigenvector centrality")
    try:
        eigenvector_cent = nx.eigenvector_centrality(G, max_iter=1000)
    except:
        print("    Warning: Eigenvector centrality failed, using zeros")
        eigenvector_cent = {node: 0.0 for node in G.nodes()}
    
    # Combine all measures
    for node in G.nodes():
        centrality_data[node] = {
            'degree_centrality': degree_cent.get(node, 0),
            'closeness_centrality': closeness_cent.get(node, 0),
            'betweenness_centrality': betweenness_cent.get(node, 0),
            'eigenvector_centrality': eigenvector_cent.get(node, 0)
        }
    
    return centrality_data

def calculate_word_frequencies(texts, vocabulary):
    """
    Calculate word frequencies from the original text corpus.
    
    Args:
        texts (list): List of text strings
        vocabulary (set): Set of vocabulary words
        
    Returns:
        dict: Word frequency counts
    """
    print("Calculating word frequencies from corpus...")
    
    word_freq = Counter()
    
    for text in tqdm(texts, desc="Processing texts"):
        if pd.notna(text) and text:
            tokens = text.split()
            for token in tokens:
                if token in vocabulary:
                    word_freq[token] += 1
    
    return dict(word_freq)

def calculate_pmi_scores(G, word_frequencies, total_words):
    """
    Calculate Pointwise Mutual Information (PMI) scores for edges.
    
    Args:
        G (nx.Graph): NetworkX graph
        word_frequencies (dict): Word frequency counts
        total_words (int): Total word count in corpus
        
    Returns:
        dict: PMI scores for each edge
    """
    print("Calculating PMI scores...")
    
    pmi_scores = {}
    
    for u, v, data in tqdm(G.edges(data=True), desc="Calculating PMI"):
        weight = data['weight']
        
        # Get individual word frequencies
        freq_u = word_frequencies.get(u, 1)
        freq_v = word_frequencies.get(v, 1)
        
        # Calculate probabilities
        p_u = freq_u / total_words
        p_v = freq_v / total_words
        p_uv = weight / total_words
        
        # Calculate PMI
        if p_u > 0 and p_v > 0 and p_uv > 0:
            pmi = math.log2(p_uv / (p_u * p_v))
        else:
            pmi = 0
        
        pmi_scores[(u, v)] = pmi
    
    return pmi_scores

def analyze_communities(G, partition, centrality_data):
    """
    Analyze community structure and properties.
    
    Args:
        G (nx.Graph): NetworkX graph
        partition (dict): Node to community mapping
        centrality_data (dict): Centrality measures for nodes
        
    Returns:
        dict: Community analysis results
    """
    print("Analyzing community structure...")
    
    # Group nodes by community
    communities = defaultdict(list)
    for node, community_id in partition.items():
        communities[community_id].append(node)
    
    community_analysis = {}
    
    for comm_id, nodes in communities.items():
        # Basic stats
        comm_size = len(nodes)
        
        # Create subgraph for this community
        subgraph = G.subgraph(nodes)
        
        # Calculate edge density
        if comm_size > 1:
            possible_edges = comm_size * (comm_size - 1) // 2
            actual_edges = subgraph.number_of_edges()
            edge_density = actual_edges / possible_edges if possible_edges > 0 else 0
        else:
            edge_density = 0
        
        # Get centrality rankings for this community
        comm_centrality = {node: centrality_data[node] for node in nodes}
        
        # Sort by closeness centrality
        sorted_by_closeness = sorted(nodes, 
                                   key=lambda x: centrality_data[x]['closeness_centrality'], 
                                   reverse=True)
        
        # Get top words for different rankings
        top_words = {
            'top_100': sorted_by_closeness[:100],
            'top_50': sorted_by_closeness[:50],
            'top_40': sorted_by_closeness[:40],
            'top_30': sorted_by_closeness[:30],
            'top_20': sorted_by_closeness[:20],
            'top_10': sorted_by_closeness[:10],
            'top_5': sorted_by_closeness[:5]
        }
        
        community_analysis[comm_id] = {
            'nodes': nodes,
            'size': comm_size,
            'edge_density': edge_density,
            'top_words': top_words,
            'centrality_data': comm_centrality
        }
    
    return community_analysis

def save_community_networks(G, community_analysis, output_dir):
    """
    Save individual network files for each community.
    
    Args:
        G (nx.Graph): Original full network
        community_analysis (dict): Community analysis results
        output_dir (Path): Output directory for community files
    """
    print(f"Creating individual community networks...")
    
    for comm_id, analysis in community_analysis.items():
        comm_dir = output_dir / f"community_{comm_id}"
        
        # Create subgraph for this community
        community_nodes = analysis['nodes']
        subgraph = G.subgraph(community_nodes).copy()
        
        # Only save if the subgraph has edges
        if subgraph.number_of_edges() > 0:
            # Save as GraphML
            graphml_file = comm_dir / f"community_{comm_id}_network.graphml"
            nx.write_graphml(subgraph, graphml_file)
            print(f"  Saved community {comm_id} network: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")
            
            # Calculate and print community network statistics
            if subgraph.number_of_nodes() > 1:
                density = nx.density(subgraph)
                print(f"    Community {comm_id} density: {density:.4f}")
            
            # Create community-specific Cytoscape files
            create_community_cytoscape_files(subgraph, analysis, comm_dir, f"community_{comm_id}")
        else:
            print(f"  Skipped community {comm_id}: no internal edges")

def create_community_cytoscape_files(subgraph, analysis, output_dir, prefix):
    """
    Create Cytoscape files for a specific community.
    
    Args:
        subgraph (nx.Graph): Community subgraph
        analysis (dict): Community analysis data
        output_dir (Path): Output directory
        prefix (str): File prefix
    """
    # Create nodes CSV for this community
    nodes_data = []
    for node in subgraph.nodes():
        degree = subgraph.degree(node)
        weighted_degree = subgraph.degree(node, weight='weight')
        centrality = analysis['centrality_data'][node]
        
        nodes_data.append({
            'Id': node,
            'Label': node,
            'Degree': degree,
            'Weighted_Degree': weighted_degree,
            'Closeness_Centrality': centrality['closeness_centrality'],
            'Betweenness_Centrality': centrality['betweenness_centrality'],
            'Eigenvector_Centrality': centrality['eigenvector_centrality']
        })
    
    df_nodes = pd.DataFrame(nodes_data)
    nodes_file = output_dir / f"{prefix}_nodes.csv"
    df_nodes.to_csv(nodes_file, index=False)
    
    # Create edges CSV for this community
    edges_data = []
    if subgraph.number_of_edges() > 0:
        # Calculate weight range for normalization
        weights = [data['weight'] for _, _, data in subgraph.edges(data=True)]
        max_weight = max(weights)
        min_weight = min(weights)
        
        for u, v, data in subgraph.edges(data=True):
            weight = data['weight']
            
            # Normalize weight
            if max_weight > min_weight:
                normalized_weight = (weight - min_weight) / (max_weight - min_weight)
            else:
                normalized_weight = 1.0
            
            edges_data.append({
                'Source': u,
                'Target': v,
                'Weight': weight,
                'Normalized_Weight': normalized_weight
            })
    
    df_edges = pd.DataFrame(edges_data)
    edges_file = output_dir / f"{prefix}_edges.csv"
    df_edges.to_csv(edges_file, index=False)

def save_community_files(G, community_analysis, partition, modularity, output_dir):
    """
    Save community detection results to files.
    
    Args:
        G (nx.Graph): Original full network
        community_analysis (dict): Community analysis results
        partition (dict): Node to community mapping
        modularity (float): Overall modularity score
        output_dir (Path): Output directory for community files
    """
    print(f"Saving community files to {output_dir}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual community files
    for comm_id, analysis in community_analysis.items():
        comm_dir = output_dir / f"community_{comm_id}"
        os.makedirs(comm_dir, exist_ok=True)
        
        # Create community nodes CSV
        nodes_data = []
        for node in analysis['nodes']:
            centrality = analysis['centrality_data'][node]
            nodes_data.append({
                'node': node,
                'closeness_centrality': centrality['closeness_centrality']
            })
        
        df_nodes = pd.DataFrame(nodes_data)
        df_nodes = df_nodes.sort_values('closeness_centrality', ascending=False)
        df_nodes.to_csv(comm_dir / f"community_{comm_id}_nodes.csv", index=False)
    
    # Save individual community networks
    save_community_networks(G, community_analysis, output_dir)
    
    # Save general community report
    report_data = []
    for comm_id, analysis in community_analysis.items():
        top_words = analysis['top_words']
        
        report_data.append({
            'community_id': f"C{comm_id}",
            'community_size': analysis['size'],
            'edge_density': analysis['edge_density'],
            'modularity_score': modularity,
            'top_100_words': ', '.join(top_words['top_100']),
            'top_50_words': ', '.join(top_words['top_50']),
            'top_40_words': ', '.join(top_words['top_40']),
            'top_30_words': ', '.join(top_words['top_30']),
            'top_20_words': ', '.join(top_words['top_20']),
            'top_10_words': ', '.join(top_words['top_10']),
            'top_5_words': ', '.join(top_words['top_5'])
        })
    
    df_report = pd.DataFrame(report_data)
    df_report = df_report.sort_values('community_size', ascending=False)
    df_report.to_csv(output_dir / "community_detection_report.csv", index=False)
    
    print(f"Saved {len(community_analysis)} community files, networks, and general report")

def create_cytoscape_files(G, partition, centrality_data, word_frequencies, pmi_scores, output_dir, prefix):
    """
    Create nodes and edges CSV files for Cytoscape visualization.
    
    Args:
        G (nx.Graph): NetworkX graph
        partition (dict): Node to community mapping
        centrality_data (dict): Centrality measures
        word_frequencies (dict): Word frequency counts
        pmi_scores (dict): PMI scores for edges
        output_dir (Path): Output directory
        prefix (str): File prefix (e.g., 'comments', 'submissions')
    """
    print(f"Creating Cytoscape files with prefix '{prefix}'...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create nodes CSV
    nodes_data = []
    for node in G.nodes():
        degree = G.degree(node)
        weighted_degree = G.degree(node, weight='weight')
        community_id = partition.get(node, 0)
        word_freq = word_frequencies.get(node, 0)
        
        centrality = centrality_data[node]
        
        nodes_data.append({
            'Id': node,
            'Label': node,
            'Cluster_ID': f"C{community_id}",
            'Word_Frequency': word_freq,
            'Degree': degree,
            'Weighted_Degree': weighted_degree,
            'Closeness_Centrality': centrality['closeness_centrality'],
            'Betweenness_Centrality': centrality['betweenness_centrality'],
            'Eigenvector_Centrality': centrality['eigenvector_centrality']
        })
    
    df_nodes = pd.DataFrame(nodes_data)
    nodes_file = output_dir / f"{prefix}_nodes.csv"
    df_nodes.to_csv(nodes_file, index=False)
    
    # Create edges CSV
    edges_data = []
    for u, v, data in G.edges(data=True):
        weight = data['weight']
        pmi = pmi_scores.get((u, v), pmi_scores.get((v, u), 0))
        
        # Normalize weight (simple min-max scaling)
        if not hasattr(create_cytoscape_files, '_weight_range_calculated'):
            weights = [data['weight'] for _, _, data in G.edges(data=True)]
            create_cytoscape_files._max_weight = max(weights)
            create_cytoscape_files._min_weight = min(weights)
            create_cytoscape_files._weight_range_calculated = True
        
        max_weight = create_cytoscape_files._max_weight
        min_weight = create_cytoscape_files._min_weight
        
        if max_weight > min_weight:
            normalized_weight = (weight - min_weight) / (max_weight - min_weight)
        else:
            normalized_weight = 1.0
        
        edges_data.append({
            'Source': u,
            'Target': v,
            'Weight': weight,
            'Normalized_Weight': normalized_weight,
            'PMI_Score': pmi
        })
    
    df_edges = pd.DataFrame(edges_data)
    edges_file = output_dir / f"{prefix}_edges.csv"
    df_edges.to_csv(edges_file, index=False)
    
    print(f"Created {nodes_file} with {len(df_nodes)} nodes")
    print(f"Created {edges_file} with {len(df_edges)} edges")

def load_corpus_for_frequencies(data_file, text_columns):
    """
    Load corpus data to calculate word frequencies.
    
    Args:
        data_file (Path): Path to processed data file
        text_columns (list): List of column names containing text
        
    Returns:
        list: List of text strings
        int: Total word count
    """
    print(f"Loading corpus from {data_file}")
    df = pd.read_csv(data_file)
    
    texts = []
    total_words = 0
    
    for col in text_columns:
        if col in df.columns:
            for text in df[col].dropna():
                if text:
                    texts.append(text)
                    total_words += len(text.split())
    
    print(f"Loaded {len(texts)} texts with {total_words} total words")
    return texts, total_words

def process_dataset_communities(dataset_name, graphml_file, data_file, text_columns):
    """
    Process community detection for a single dataset.
    
    Args:
        dataset_name (str): Name of dataset ('comments' or 'submissions')
        graphml_file (Path): Path to GraphML network file
        data_file (Path): Path to processed data file
        text_columns (list): Column names containing text data
    """
    print("=" * 70)
    print(f"PROCESSING {dataset_name.upper()} COMMUNITY DETECTION")
    print("=" * 70)
    
    project_root = get_project_root()
    
    # Load network
    G = load_network(graphml_file)
    
    if G.number_of_nodes() == 0:
        print(f"Warning: Empty network for {dataset_name}, skipping...")
        return
    
    # Detect communities
    partition, modularity = detect_communities_louvain(G)
    
    # Calculate centrality measures
    centrality_data = calculate_centrality_measures(G)
    
    # Load corpus for word frequencies
    texts, total_words = load_corpus_for_frequencies(data_file, text_columns)
    vocabulary = set(G.nodes())
    word_frequencies = calculate_word_frequencies(texts, vocabulary)
    
    # Calculate PMI scores
    pmi_scores = calculate_pmi_scores(G, word_frequencies, total_words)
    
    # Analyze communities
    community_analysis = analyze_communities(G, partition, centrality_data)
    
    # Save community files
    communities_dir = project_root / "results" / "communities" / dataset_name
    save_community_files(G, community_analysis, partition, modularity, communities_dir)
    
    # Create Cytoscape files
    table_dir = project_root / "results" / "table"
    create_cytoscape_files(G, partition, centrality_data, word_frequencies, 
                          pmi_scores, table_dir, dataset_name)
    
    print(f"\n{dataset_name.capitalize()} community detection complete!")
    print(f"Communities: {len(community_analysis)}")
    print(f"Modularity: {modularity:.4f}")

def main():
    """Main community detection function"""
    print("Community Detection and Network Analysis Pipeline")
    print("=" * 70)
    
    try:
        project_root = get_project_root()
        
        # Process comments
        comments_graphml = project_root / "results" / "graph" / "comments_network.graphml"
        comments_data = project_root / "data" / "cleaned" / "blooddonors_comments_processed.csv"
        
        if comments_graphml.exists() and comments_data.exists():
            process_dataset_communities("comments", comments_graphml, comments_data, ["cleaned_body"])
        else:
            print(f"Comments files not found, skipping...")
        
        print("\n" + "=" * 70)
        
        # Process submissions
        submissions_graphml = project_root / "results" / "graph" / "submissions_network.graphml"
        submissions_data = project_root / "data" / "cleaned" / "blooddonors_submissions_processed.csv"
        
        if submissions_graphml.exists() and submissions_data.exists():
            process_dataset_communities("submissions", submissions_graphml, submissions_data, 
                                      ["cleaned_title", "cleaned_selftext"])
        else:
            print(f"Submissions files not found, skipping...")
        
        print("\n" + "=" * 70)
        print("COMMUNITY DETECTION COMPLETE!")
        print("=" * 70)
        print("Output files created:")
        print("  - Community folders with individual CSV files")
        print("  - Community detection reports")
        print("  - Cytoscape nodes and edges CSV files")
        
    except Exception as e:
        print(f"Error during community detection: {e}")
        raise

if __name__ == "__main__":
    main()
