"""
Emotional Profiling Module
Adds NRC Emotion Lexicon and VAD (Valence-Arousal-Dominance) scores to node CSV files.
"""

import pandas as pd
from pathlib import Path
import os
from tqdm import tqdm
from collections import defaultdict

def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent

def load_nrc_emotion_lexicon(lexicon_file):
    """
    Load NRC Emotion Lexicon and create a word-to-emotions mapping.
    
    Args:
        lexicon_file (Path): Path to NRC emotion lexicon file
        
    Returns:
        dict: Mapping from word to emotion scores
    """
    print("Loading NRC Emotion Lexicon...")
    
    # Initialize emotions mapping
    emotion_mapping = defaultdict(lambda: {
        'anger': 0, 'fear': 0, 'anticipation': 0, 'trust': 0, 
        'surprise': 0, 'sadness': 0, 'joy': 0, 'disgust': 0,
        'negative': 0, 'positive': 0
    })
    
    # Read the lexicon file
    with open(lexicon_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing NRC lexicon"):
            parts = line.strip().split('\t')
            if len(parts) == 3:
                word, emotion, score = parts
                word = word.lower()
                score = int(score)
                
                if emotion in emotion_mapping[word]:
                    emotion_mapping[word][emotion] = score
    
    print(f"Loaded NRC emotions for {len(emotion_mapping)} words")
    return dict(emotion_mapping)

def load_vad_lexicon(vad_file):
    """
    Load VAD (Valence-Arousal-Dominance) lexicon.
    
    Args:
        vad_file (Path): Path to VAD lexicon file
        
    Returns:
        dict: Mapping from word to VAD scores
    """
    print("Loading VAD Lexicon...")
    
    vad_mapping = {}
    
    # Read the VAD file
    with open(vad_file, 'r', encoding='utf-8') as f:
        # Skip header
        next(f)
        
        for line in tqdm(f, desc="Processing VAD lexicon"):
            parts = line.strip().split('\t')
            if len(parts) == 4:
                term, valence, arousal, dominance = parts
                term = term.lower()
                
                try:
                    vad_mapping[term] = {
                        'valence': float(valence),
                        'arousal': float(arousal),
                        'dominance': float(dominance)
                    }
                except ValueError:
                    # Skip lines with invalid numeric values
                    continue
    
    print(f"Loaded VAD scores for {len(vad_mapping)} words")
    return vad_mapping

def add_emotional_profiling(df, nrc_emotions, vad_scores):
    """
    Add emotional profiling columns to a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame with node data
        nrc_emotions (dict): NRC emotion mapping
        vad_scores (dict): VAD score mapping
        
    Returns:
        pd.DataFrame: DataFrame with added emotional columns
    """
    print(f"Adding emotional profiling to {len(df)} nodes...")
    
    # Create a copy of the dataframe
    df_emotional = df.copy()
    
    # Initialize emotion columns
    emotion_cols = ['anger', 'fear', 'anticipation', 'trust', 'surprise', 
                   'sadness', 'joy', 'disgust', 'negative', 'positive']
    vad_cols = ['valence', 'arousal', 'dominance']
    
    # Initialize all emotion and VAD columns with N/A
    for col in emotion_cols + vad_cols:
        df_emotional[col] = 'N/A'
    
    # Process each row
    for idx, row in tqdm(df_emotional.iterrows(), total=len(df_emotional), desc="Processing nodes"):
        word = str(row['Label']).lower()
        
        # Add NRC emotion scores
        if word in nrc_emotions:
            emotions = nrc_emotions[word]
            for emotion in emotion_cols:
                df_emotional.at[idx, emotion] = emotions[emotion]
        
        # Add VAD scores
        if word in vad_scores:
            vad = vad_scores[word]
            for vad_col in vad_cols:
                df_emotional.at[idx, vad_col] = vad[vad_col]
    
    return df_emotional

def process_node_file(input_file, output_file, nrc_emotions, vad_scores):
    """
    Process a single node CSV file and add emotional profiling.
    
    Args:
        input_file (Path): Input CSV file path
        output_file (Path): Output CSV file path
        nrc_emotions (dict): NRC emotion mapping
        vad_scores (dict): VAD score mapping
    """
    print(f"\nProcessing: {input_file.name}")
    
    # Load the CSV file
    df = pd.read_csv(input_file)
    
    # Add emotional profiling
    df_emotional = add_emotional_profiling(df, nrc_emotions, vad_scores)
    
    # Ensure output directory exists
    os.makedirs(output_file.parent, exist_ok=True)
    
    # Save the result
    df_emotional.to_csv(output_file, index=False)
    
    print(f"Saved: {output_file}")
    
    # Print summary statistics
    print_emotional_summary(df_emotional)

def print_emotional_summary(df):
    """Print summary statistics for emotional profiling."""
    emotion_cols = ['anger', 'fear', 'anticipation', 'trust', 'surprise', 
                   'sadness', 'joy', 'disgust', 'negative', 'positive']
    vad_cols = ['valence', 'arousal', 'dominance']
    
    print("  Emotional Coverage Summary:")
    
    # NRC emotion coverage
    total_words = len(df)
    nrc_coverage = 0
    vad_coverage = 0
    
    for _, row in df.iterrows():
        # Check if any NRC emotion is not N/A
        if any(row[col] != 'N/A' for col in emotion_cols):
            nrc_coverage += 1
        
        # Check if any VAD score is not N/A  
        if any(row[col] != 'N/A' for col in vad_cols):
            vad_coverage += 1
    
    print(f"    NRC Emotion coverage: {nrc_coverage}/{total_words} ({nrc_coverage/total_words*100:.1f}%)")
    print(f"    VAD score coverage: {vad_coverage}/{total_words} ({vad_coverage/total_words*100:.1f}%)")
    
    # Show top emotions (for words that have NRC data)
    if nrc_coverage > 0:
        emotion_counts = {}
        for col in emotion_cols:
            emotion_counts[col] = (df[col] == 1).sum()
        
        print("    Top emotions:")
        sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
        for emotion, count in sorted_emotions[:5]:
            print(f"      {emotion}: {count} words")

def find_all_node_files():
    """
    Find all node CSV files in the results directory.
    
    Returns:
        dict: Organized paths for different types of node files
    """
    project_root = get_project_root()
    results_dir = project_root / "results"
    
    node_files = {
        'main': [],
        'communities': {
            'comments': [],
            'submissions': []
        }
    }
    
    # Main node files (comments_nodes.csv, submissions_nodes.csv)
    table_dir = results_dir / "table"
    if table_dir.exists():
        for file_path in table_dir.glob("*_nodes.csv"):
            node_files['main'].append(file_path)
    
    # Community node files
    communities_dir = results_dir / "communities"
    if communities_dir.exists():
        # Comments communities
        comments_dir = communities_dir / "comments"
        if comments_dir.exists():
            for community_dir in comments_dir.glob("community_*"):
                if community_dir.is_dir():
                    for file_path in community_dir.glob("*_nodes.csv"):
                        node_files['communities']['comments'].append(file_path)
        
        # Submissions communities
        submissions_dir = communities_dir / "submissions"
        if submissions_dir.exists():
            for community_dir in submissions_dir.glob("community_*"):
                if community_dir.is_dir():
                    for file_path in community_dir.glob("*_nodes.csv"):
                        node_files['communities']['submissions'].append(file_path)
    
    return node_files

def create_output_path(input_path, base_results_dir):
    """
    Create appropriate output path for emotional profiling results.
    
    Args:
        input_path (Path): Original input file path
        base_results_dir (Path): Base results directory
        
    Returns:
        Path: Output file path
    """
    emotional_dir = base_results_dir / "emotional_profiling"
    
    # Determine the structure based on input path
    if "communities" in str(input_path):
        if "comments" in str(input_path):
            # Extract community folder name
            community_name = input_path.parent.name
            output_path = emotional_dir / "communities" / "comments" / community_name / input_path.name
        elif "submissions" in str(input_path):
            # Extract community folder name
            community_name = input_path.parent.name
            output_path = emotional_dir / "communities" / "submissions" / community_name / input_path.name
        else:
            output_path = emotional_dir / "communities" / input_path.name
    else:
        # Main files (comments_nodes.csv, submissions_nodes.csv)
        if "comments" in input_path.name:
            output_path = emotional_dir / "comments" / input_path.name
        elif "submissions" in input_path.name:
            output_path = emotional_dir / "submissions" / input_path.name
        else:
            output_path = emotional_dir / input_path.name
    
    return output_path

def process_all_node_files():
    """
    Process all node CSV files and add emotional profiling.
    """
    print("=" * 70)
    print("EMOTIONAL PROFILING PIPELINE")
    print("=" * 70)
    
    project_root = get_project_root()
    
    # Load lexicons
    nrc_file = project_root / "analyse_sources" / "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
    vad_file = project_root / "analyse_sources" / "NRC-VAD-Lexicon-v2.1.txt"
    
    if not nrc_file.exists():
        raise FileNotFoundError(f"NRC emotion lexicon not found: {nrc_file}")
    if not vad_file.exists():
        raise FileNotFoundError(f"VAD lexicon not found: {vad_file}")
    
    # Load lexicons
    nrc_emotions = load_nrc_emotion_lexicon(nrc_file)
    vad_scores = load_vad_lexicon(vad_file)
    
    # Find all node files
    node_files = find_all_node_files()
    
    # Process main files first (for optimization as suggested)
    print("\n" + "=" * 50)
    print("PROCESSING MAIN NODE FILES")
    print("=" * 50)
    
    for input_file in node_files['main']:
        output_file = create_output_path(input_file, project_root / "results")
        process_node_file(input_file, output_file, nrc_emotions, vad_scores)
    
    # Process community files
    print("\n" + "=" * 50)
    print("PROCESSING COMMUNITY NODE FILES")
    print("=" * 50)
    
    # Comments communities
    if node_files['communities']['comments']:
        print(f"\nProcessing {len(node_files['communities']['comments'])} comments community files...")
        for input_file in node_files['communities']['comments']:
            output_file = create_output_path(input_file, project_root / "results")
            process_node_file(input_file, output_file, nrc_emotions, vad_scores)
    
    # Submissions communities
    if node_files['communities']['submissions']:
        print(f"\nProcessing {len(node_files['communities']['submissions'])} submissions community files...")
        for input_file in node_files['communities']['submissions']:
            output_file = create_output_path(input_file, project_root / "results")
            process_node_file(input_file, output_file, nrc_emotions, vad_scores)

def create_emotional_summary_report():
    """
    Create a summary report of emotional profiling across all files.
    """
    print("\n" + "=" * 50)
    print("CREATING EMOTIONAL SUMMARY REPORT")
    print("=" * 50)
    
    project_root = get_project_root()
    emotional_dir = project_root / "results" / "emotional_profiling"
    
    summary_data = []
    emotion_cols = ['anger', 'fear', 'anticipation', 'trust', 'surprise', 
                   'sadness', 'joy', 'disgust', 'negative', 'positive']
    
    # Process all emotional profiling files
    for csv_file in emotional_dir.rglob("*_nodes.csv"):
        df = pd.read_csv(csv_file)
        
        # Calculate statistics
        total_words = len(df)
        nrc_coverage = sum(1 for _, row in df.iterrows() 
                          if any(row[col] != 'N/A' for col in emotion_cols))
        
        # Count emotions
        emotion_counts = {}
        for col in emotion_cols:
            emotion_counts[col] = (df[col] == 1).sum()
        
        # Relative path for readability
        relative_path = csv_file.relative_to(emotional_dir)
        
        summary_data.append({
            'file': str(relative_path),
            'total_words': total_words,
            'nrc_coverage': nrc_coverage,
            'coverage_percent': nrc_coverage / total_words * 100 if total_words > 0 else 0,
            **emotion_counts
        })
    
    # Create summary DataFrame
    df_summary = pd.DataFrame(summary_data)
    
    # Save summary report
    summary_file = emotional_dir / "emotional_profiling_summary.csv"
    df_summary.to_csv(summary_file, index=False)
    
    print(f"Summary report saved: {summary_file}")
    print(f"Processed {len(df_summary)} files")

def main():
    """Main emotional profiling function"""
    try:
        # Process all node files
        process_all_node_files()
        
        # Create summary report
        create_emotional_summary_report()
        
        print("\n" + "=" * 70)
        print("✅ EMOTIONAL PROFILING COMPLETE!")
        print("=" * 70)
        print("Output files created in:")
        print("  results/emotional_profiling/comments/")
        print("  results/emotional_profiling/submissions/") 
        print("  results/emotional_profiling/communities/")
        print("  results/emotional_profiling/emotional_profiling_summary.csv")
        
    except Exception as e:
        print(f"❌ Error during emotional profiling: {e}")
        raise

if __name__ == "__main__":
    main()
