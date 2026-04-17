"""
Bi-gram Co-occurrence Matrix Construction Module
Constructs co-occurrence matrices from preprocessed text data using a window size of 2 (bigrams).
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
import os
from tqdm import tqdm

def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent

def extract_bigrams(text):
    """
    Extract bigrams (adjacent word pairs) from a text string.
    
    Args:
        text (str): Preprocessed text with space-separated tokens
        
    Returns:
        list: List of (word1, word2) tuples representing bigrams
    """
    if pd.isna(text) or not text:
        return []
    
    tokens = text.split()
    if len(tokens) < 2:
        return []
    
    bigrams = []
    for i in range(len(tokens) - 1):
        word1 = tokens[i]
        word2 = tokens[i + 1]
        bigrams.append((word1, word2))
    
    return bigrams

def build_vocabulary(texts):
    """
    Build vocabulary from all texts.
    
    Args:
        texts (list): List of preprocessed text strings
        
    Returns:
        dict: Mapping from word to index
        list: List of unique words (vocabulary)
    """
    vocab_set = set()
    
    print("Building vocabulary...")
    for text in tqdm(texts, desc="Processing texts"):
        if pd.notna(text) and text:
            tokens = text.split()
            vocab_set.update(tokens)
    
    vocabulary = sorted(list(vocab_set))
    word_to_idx = {word: i for i, word in enumerate(vocabulary)}
    
    print(f"Vocabulary size: {len(vocabulary)} unique words")
    return word_to_idx, vocabulary

def construct_cooccurrence_matrix(texts, word_to_idx):
    """
    Construct co-occurrence matrix from texts using bigram approach.
    
    Args:
        texts (list): List of preprocessed text strings
        word_to_idx (dict): Mapping from word to matrix index
        
    Returns:
        np.ndarray: Symmetric co-occurrence matrix
        dict: Co-occurrence counts as {(word1, word2): count}
    """
    vocab_size = len(word_to_idx)
    cooccurrence_matrix = np.zeros((vocab_size, vocab_size), dtype=int)
    cooccurrence_counts = defaultdict(int)
    
    print("Constructing co-occurrence matrix...")
    total_bigrams = 0
    
    for text in tqdm(texts, desc="Processing texts for bigrams"):
        bigrams = extract_bigrams(text)
        
        for word1, word2 in bigrams:
            if word1 in word_to_idx and word2 in word_to_idx:
                idx1 = word_to_idx[word1]
                idx2 = word_to_idx[word2]
                
                # Increment both directions for undirected graph
                cooccurrence_matrix[idx1, idx2] += 1
                cooccurrence_matrix[idx2, idx1] += 1
                
                # Store counts (ensure consistent ordering)
                pair = tuple(sorted([word1, word2]))
                cooccurrence_counts[pair] += 1
                
                total_bigrams += 1
    
    print(f"Total bigrams processed: {total_bigrams}")
    print(f"Unique co-occurrence pairs: {len(cooccurrence_counts)}")
    
    return cooccurrence_matrix, dict(cooccurrence_counts)

def save_cooccurrence_matrix(matrix, vocabulary, output_file):
    """
    Save co-occurrence matrix as CSV with word labels.
    
    Args:
        matrix (np.ndarray): Co-occurrence matrix
        vocabulary (list): List of words corresponding to matrix indices
        output_file (str/Path): Output CSV file path
    """
    print(f"Saving co-occurrence matrix to {output_file}")
    
    # Create DataFrame with word labels
    df_matrix = pd.DataFrame(matrix, index=vocabulary, columns=vocabulary)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to CSV
    df_matrix.to_csv(output_file)
    print(f"Matrix saved with shape: {matrix.shape}")

def save_cooccurrence_pairs(cooccurrence_counts, output_file):
    """
    Save co-occurrence pairs as CSV.
    
    Args:
        cooccurrence_counts (dict): Dictionary of {(word1, word2): count}
        output_file (str/Path): Output CSV file path
    """
    print(f"Saving co-occurrence pairs to {output_file}")
    
    # Convert to list of dictionaries
    pairs_data = []
    for (word1, word2), count in cooccurrence_counts.items():
        pairs_data.append({
            'word1': word1,
            'word2': word2,
            'co_occurrence': count
        })
    
    # Create DataFrame and sort by co-occurrence count (descending)
    df_pairs = pd.DataFrame(pairs_data)
    df_pairs = df_pairs.sort_values('co_occurrence', ascending=False)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to CSV
    df_pairs.to_csv(output_file, index=False)
    print(f"Saved {len(df_pairs)} co-occurrence pairs")
    
    # Show top pairs
    print("\nTop 10 co-occurrence pairs:")
    print(df_pairs.head(10).to_string(index=False))

def process_comments_matrix():
    """Process comments data to create co-occurrence matrix and pairs."""
    print("=" * 60)
    print("PROCESSING COMMENTS DATA")
    print("=" * 60)
    
    # File paths
    project_root = get_project_root()
    input_file = project_root / "data" / "cleaned" / "blooddonors_comments_processed.csv"
    matrix_output = project_root / "results" / "matrix" / "comments_cooccurrence_matrix.csv"
    pairs_output = project_root / "results" / "matrix" / "comments_cooccurrence_pairs.csv"
    
    # Load data
    print(f"Loading comments data from {input_file}")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} comments")
    
    # Extract texts
    texts = df['cleaned_body'].dropna().tolist()
    print(f"Processing {len(texts)} valid comments")
    
    # Build vocabulary
    word_to_idx, vocabulary = build_vocabulary(texts)
    
    # Construct co-occurrence matrix
    matrix, cooccurrence_counts = construct_cooccurrence_matrix(texts, word_to_idx)
    
    # Save results
    save_cooccurrence_matrix(matrix, vocabulary, matrix_output)
    save_cooccurrence_pairs(cooccurrence_counts, pairs_output)
    
    print(f"\nComments processing complete!")
    print(f"Matrix output: {matrix_output}")
    print(f"Pairs output: {pairs_output}")
    
    return matrix, vocabulary, cooccurrence_counts

def process_submissions_matrix():
    """Process submissions data to create co-occurrence matrix and pairs."""
    print("=" * 60)
    print("PROCESSING SUBMISSIONS DATA")
    print("=" * 60)
    
    # File paths
    project_root = get_project_root()
    input_file = project_root / "data" / "cleaned" / "blooddonors_submissions_processed.csv"
    matrix_output = project_root / "results" / "matrix" / "submissions_cooccurrence_matrix.csv"
    pairs_output = project_root / "results" / "matrix" / "submissions_cooccurrence_pairs.csv"
    
    # Load data
    print(f"Loading submissions data from {input_file}")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} submissions")
    
    # Combine title and selftext
    combined_texts = []
    for _, row in df.iterrows():
        combined_text = ""
        
        # Add title if available
        if pd.notna(row['cleaned_title']) and row['cleaned_title']:
            combined_text += row['cleaned_title']
        
        # Add selftext if available
        if pd.notna(row['cleaned_selftext']) and row['cleaned_selftext']:
            if combined_text:
                combined_text += " " + row['cleaned_selftext']
            else:
                combined_text = row['cleaned_selftext']
        
        if combined_text.strip():
            combined_texts.append(combined_text.strip())
    
    print(f"Processing {len(combined_texts)} valid submissions (combined title + selftext)")
    
    # Build vocabulary
    word_to_idx, vocabulary = build_vocabulary(combined_texts)
    
    # Construct co-occurrence matrix
    matrix, cooccurrence_counts = construct_cooccurrence_matrix(combined_texts, word_to_idx)
    
    # Save results
    save_cooccurrence_matrix(matrix, vocabulary, matrix_output)
    save_cooccurrence_pairs(cooccurrence_counts, pairs_output)
    
    print(f"\nSubmissions processing complete!")
    print(f"Matrix output: {matrix_output}")
    print(f"Pairs output: {pairs_output}")
    
    return matrix, vocabulary, cooccurrence_counts

def get_matrix_statistics(matrix, vocabulary, cooccurrence_counts):
    """Print statistics about the constructed matrix."""
    print("\n" + "=" * 40)
    print("MATRIX STATISTICS")
    print("=" * 40)
    
    vocab_size = len(vocabulary)
    total_possible_pairs = vocab_size * (vocab_size - 1) // 2
    actual_pairs = len(cooccurrence_counts)
    sparsity = 1 - (actual_pairs / total_possible_pairs)
    
    print(f"Vocabulary size: {vocab_size:,}")
    print(f"Matrix shape: {matrix.shape}")
    print(f"Total possible unique pairs: {total_possible_pairs:,}")
    print(f"Actual co-occurring pairs: {actual_pairs:,}")
    print(f"Matrix sparsity: {sparsity:.4f} ({sparsity*100:.2f}% empty)")
    
    # Count statistics
    counts = list(cooccurrence_counts.values())
    print(f"\nCo-occurrence count statistics:")
    print(f"  Min: {min(counts)}")
    print(f"  Max: {max(counts)}")
    print(f"  Mean: {np.mean(counts):.2f}")
    print(f"  Median: {np.median(counts):.2f}")
    print(f"  Std: {np.std(counts):.2f}")
    
    # Frequency distribution
    count_freq = Counter(counts)
    print(f"\nFrequency distribution (top 10):")
    for count, freq in sorted(count_freq.items(), reverse=True)[:10]:
        print(f"  Co-occurrence count {count}: {freq} pairs")

def main():
    """Main matrix construction function"""
    print("Bi-gram Co-occurrence Matrix Construction Pipeline")
    print("=" * 70)
    
    try:
        # Process comments
        comments_matrix, comments_vocab, comments_counts = process_comments_matrix()
        get_matrix_statistics(comments_matrix, comments_vocab, comments_counts)
        
        print("\n" + "=" * 70)
        
        # Process submissions
        submissions_matrix, submissions_vocab, submissions_counts = process_submissions_matrix()
        get_matrix_statistics(submissions_matrix, submissions_vocab, submissions_counts)
        
        print("\n" + "=" * 70)
        print("MATRIX CONSTRUCTION COMPLETE!")
        print("=" * 70)
        print("Output files created:")
        print("  - comments_cooccurrence_matrix.csv")
        print("  - comments_cooccurrence_pairs.csv")
        print("  - submissions_cooccurrence_matrix.csv")
        print("  - submissions_cooccurrence_pairs.csv")
        
    except Exception as e:
        print(f"Error during matrix construction: {e}")
        raise

if __name__ == "__main__":
    main()

