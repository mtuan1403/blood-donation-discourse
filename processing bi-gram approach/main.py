"""
Main Pipeline for Reddit Bi-gram Co-occurrence Network Analysis

This script orchestrates the entire pipeline:
1. Data preprocessing
2. Co-occurrence matrix construction
3. Network construction
4. Community detection and Cytoscape file generation

Usage:
    python main.py [--skip-preprocessing] [--skip-matrix] [--skip-network] [--skip-community]
"""

import argparse
import sys
import time
from pathlib import Path

# Add scripts directory to path
scripts_dir = Path(__file__).parent / "scripts"
sys.path.insert(0, str(scripts_dir))

def run_step(step_name, module_name, skip=False):
    """
    Run a pipeline step.
    
    Args:
        step_name (str): Human-readable name of the step
        module_name (str): Name of the Python module to run
        skip (bool): Whether to skip this step
        
    Returns:
        bool: True if successful, False otherwise
    """
    if skip:
        print(f"SKIPPING: {step_name}")
        print("=" * 70)
        return True
    
    print(f"STARTING: {step_name}")
    print("=" * 70)
    start_time = time.time()
    
    try:
        # Import and run the module
        module = __import__(module_name)
        module.main()
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ COMPLETED: {step_name} ({elapsed_time:.1f}s)")
        print("=" * 70)
        return True
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n❌ FAILED: {step_name} ({elapsed_time:.1f}s)")
        print(f"Error: {e}")
        print("=" * 70)
        return False

def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")
    
    required_packages = [
        'pandas',
        'numpy',
        'networkx',
        'nltk',
        'contractions',
        'tqdm',
        'python-louvain'
    ]
    
    optional_packages = [
        # No optional packages currently needed
    ]
    
    missing_packages = []
    
    # Check required packages
    for package in required_packages:
        try:
            if package == 'python-louvain':
                import community
            else:
                __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package}")
    
    # Check optional packages
    missing_optional = []
    for package in optional_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package} (optional)")
        except ImportError:
            missing_optional.append(package)
            print(f"  ⚠️  {package} (optional - enhanced features available if installed)")
    
    if missing_packages:
        print(f"\nMissing required packages: {', '.join(missing_packages)}")
        print("Please install them using:")
        for package in missing_packages:
            print(f"  pip install {package}")
        return False
    
    if missing_optional:
        print(f"\nOptional packages not installed: {', '.join(missing_optional)}")
        print("Install for enhanced features:")
        for package in missing_optional:
            print(f"  pip install {package}")
    
    print("All required dependencies are installed! ✅")
    return True

def check_input_files():
    """Check if required input files exist."""
    print("\nChecking input files...")
    
    project_root = Path(__file__).parent
    required_files = [
        project_root / "data" / "cleaned" / "blooddonors_comments_processed.csv",
        project_root / "data" / "cleaned" / "blooddonors_submissions_processed.csv"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if file_path.exists():
            print(f"  ✅ {file_path.name}")
        else:
            missing_files.append(file_path)
            print(f"  ❌ {file_path}")
    
    if missing_files:
        print(f"\nMissing input files:")
        for file_path in missing_files:
            print(f"  {file_path}")
        print("\nPlease ensure the preprocessed data files exist before running the pipeline.")
        return False
    
    print("All input files found! ✅")
    return True

def create_output_directories():
    """Create necessary output directories."""
    print("\nCreating output directories...")
    
    project_root = Path(__file__).parent
    directories = [
        project_root / "results" / "matrix",
        project_root / "results" / "graph", 
        project_root / "results" / "communities",
        project_root / "results" / "table"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {directory}")

def print_pipeline_summary():
    """Print a summary of what the pipeline will do."""
    print("""
🔬 Reddit Bi-gram Co-occurrence Network Analysis Pipeline
=========================================================

This pipeline will process Reddit blood donation data through the following steps:

1️⃣  PREPROCESSING (Optional)
   • Clean and tokenize Reddit comments and submissions
   • Remove spam, bots, and low-quality content
   • Lemmatize and filter text data

2️⃣  MATRIX CONSTRUCTION
   • Extract bi-grams (adjacent word pairs) from cleaned text
   • Build co-occurrence matrices for comments and submissions
   • Generate word pair frequency tables

3️⃣  NETWORK CONSTRUCTION  
   • Convert co-occurrence matrices to NetworkX graphs
   • Apply frequency thresholds to reduce noise
   • Export networks as GraphML files

4️⃣  COMMUNITY DETECTION & ANALYSIS
   • Apply Louvain method for community detection
   • Calculate centrality measures (closeness, betweenness, eigenvector)
   • Extract individual community subnetworks as GraphML files
   • Generate community reports and individual CSV files
   • Create Cytoscape-ready node and edge files for full network and communities

5️⃣  EMOTIONAL PROFILING
   • Map words to NRC Emotion Lexicon (8 emotions + sentiment)
   • Add VAD scores (Valence-Arousal-Dominance) to all nodes
   • Create emotional profiles for full networks and individual communities
   • Generate summary reports of emotional characteristics

📁 OUTPUT FILES:
   • results/matrix/: Co-occurrence matrices and pair frequency tables
   • results/graph/: NetworkX GraphML files for network visualization
   • results/communities/: Community detection results and reports
   • results/table/: Cytoscape-ready CSV files for network visualization
   • results/emotional_profiling/: Node files with emotional and VAD scores

""")

def print_final_summary(successful_steps, failed_steps, total_time):
    """Print final pipeline summary."""
    print("\n" + "=" * 70)
    print("🎉 PIPELINE EXECUTION COMPLETE!")
    print("=" * 70)
    
    print(f"⏱️  Total execution time: {total_time:.1f} seconds")
    print(f"✅ Successful steps: {len(successful_steps)}")
    if successful_steps:
        for step in successful_steps:
            print(f"   • {step}")
    
    if failed_steps:
        print(f"❌ Failed steps: {len(failed_steps)}")
        for step in failed_steps:
            print(f"   • {step}")
    
    print("\n📁 Output files location:")
    project_root = Path(__file__).parent
    print(f"   {project_root / 'results'}")
    
    if not failed_steps:
        print("\n🎯 Next steps:")
        print("   • Import GraphML files into network analysis software")
        print("   • Use Cytoscape CSV files for network visualization")
        print("   • Analyze community detection reports")
        print("   • Explore word co-occurrence patterns")

def main():
    """Main pipeline execution function."""
    parser = argparse.ArgumentParser(description="Reddit Bi-gram Co-occurrence Network Analysis Pipeline")
    parser.add_argument("--skip-preprocessing", action="store_true", 
                       help="Skip data preprocessing step")
    parser.add_argument("--skip-matrix", action="store_true",
                       help="Skip matrix construction step")
    parser.add_argument("--skip-network", action="store_true", 
                       help="Skip network construction step")
    parser.add_argument("--skip-community", action="store_true",
                       help="Skip community detection step")
    parser.add_argument("--skip-emotional", action="store_true",
                       help="Skip emotional profiling step")
    parser.add_argument("--check-only", action="store_true",
                       help="Only check dependencies and files, don't run pipeline")
    
    args = parser.parse_args()
    
    # Print pipeline summary
    print_pipeline_summary()
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install missing packages.")
        sys.exit(1)
    
    # Create output directories
    create_output_directories()
    
    # Check input files (only if not skipping preprocessing)
    if not args.skip_preprocessing:
        print("\nNote: Preprocessing will be run, so raw data files are needed.")
    else:
        if not check_input_files():
            print("\n❌ Input file check failed. Please ensure preprocessed files exist.")
            sys.exit(1)
    
    if args.check_only:
        print("\n✅ All checks passed! Pipeline is ready to run.")
        return
    
    # Pipeline execution
    print("\n" + "=" * 70)
    print("🚀 STARTING PIPELINE EXECUTION")
    print("=" * 70)
    
    start_time = time.time()
    successful_steps = []
    failed_steps = []
    
    # Define pipeline steps
    steps = [
        ("Data Preprocessing", "preprocessing", args.skip_preprocessing),
        ("Matrix Construction", "matrix_construction", args.skip_matrix),
        ("Network Construction", "network_construction", args.skip_network),
        ("Community Detection & Analysis", "community_detection", args.skip_community),
        ("Emotional Profiling", "emotional_profiling", args.skip_emotional)
    ]
    
    # Execute steps
    for step_name, module_name, skip in steps:
        if run_step(step_name, module_name, skip):
            if not skip:
                successful_steps.append(step_name)
        else:
            failed_steps.append(step_name)
            
            # Ask user if they want to continue
            user_input = input(f"\n⚠️  {step_name} failed. Continue with remaining steps? (y/n): ")
            if user_input.lower() not in ['y', 'yes']:
                print("Pipeline execution stopped by user.")
                break
    
    # Print final summary
    total_time = time.time() - start_time
    print_final_summary(successful_steps, failed_steps, total_time)
    
    # Exit with appropriate code
    if failed_steps:
        sys.exit(1)
    else:
        print("\n🎉 All steps completed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()

