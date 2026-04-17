"""
Reddit Text Preprocessing Module
Cleans and preprocesses Reddit comments and submissions for network analysis.
"""

import re
import pandas as pd
from nltk.corpus import stopwords, words
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import contractions  # pip install contractions
import os
from pathlib import Path

def setup_nltk_resources():
    """Download required NLTK resources if not already available"""
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/words')
        nltk.data.find('taggers/averaged_perceptron_tagger')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading required NLTK resources...")
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('words')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')

class RedditTextProcessor:
    def __init__(self):
        setup_nltk_resources()
        
        # Initialize resources
        self.stop_words = set(stopwords.words("english")) - {"no", "not", "never", "none"}
        self.english_vocab = set(w.lower() for w in words.words())
        self.lemmatizer = WordNetLemmatizer()
        
        # Contraction artifacts to remove
        self.contraction_artifacts = {"'s", "'m", "'re", "'ve", "'d", "'ll", "n't", "'t"}
        
        # Meaningful short words to preserve
        self.meaningful_short = {'ok', 'no', 'go', 'up', 'me', 'we', 'he', 'be', 'do', 'to', 'so', 'my', 'by', 'or', 'if'}
    
    def get_wordnet_pos(self, tag):
        """Map POS tag to WordNet POS for better lemmatization"""
        if tag.startswith("J"): return wordnet.ADJ
        elif tag.startswith("V"): return wordnet.VERB
        elif tag.startswith("N"): return wordnet.NOUN
        elif tag.startswith("R"): return wordnet.ADV
        return wordnet.NOUN
    
    def is_corrupted_text(self, text):
        """Detect various forms of corrupted/spam text"""
        if not text or len(text.strip()) < 3:
            return True
            
        # Check for common Reddit deletion markers
        deletion_markers = ["[deleted]", "[removed]", "nan", "", "none", "null"]
        if text.lower().strip() in deletion_markers:
            return True
        
        # Check for bot conversion text
        if "mouseover" in text.lower() and "metric conversion" in text.lower():
            return True
            
        # Check for excessive special characters (like Unicode corruption)
        special_char_count = sum(1 for c in text if ord(c) > 127)
        if len(text) > 0 and special_char_count / len(text) > 0.3:
            return True
        
        # Check for excessive repetition
        words = text.split()
        if len(words) > 3:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            max_count = max(word_counts.values())
            if max_count / len(words) > 0.4:  # Single word makes up >40%
                return True
        
        # Check for concatenated text without spaces
        if len(words) > 0:
            avg_word_length = sum(len(word) for word in words) / len(words)
            if avg_word_length > 12:  # Unusually long average word length
                return True
        
        # Check for specific spam patterns
        spam_patterns = [
            r'(.{15,})\1{2,}',  # Long sequences repeated 3+ times
            r'\b(\w{8,})\1+',   # Long words repeated
            r'[‡Æâ‡Øç‡Æï‡Æ≥‡ØÅ‡Æï‡Øç‡Æï‡ØÅ]',  # Unicode corruption patterns
        ]
        
        for pattern in spam_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def is_english_text(self, tokens, threshold=0.6):
        """Check if enough tokens are English words"""
        if not tokens:
            return False
        
        # Filter out artifacts for this check
        meaningful_tokens = [t for t in tokens if t not in self.contraction_artifacts and len(t) > 1]
        if not meaningful_tokens:
            return False
        
        english_count = sum(1 for t in meaningful_tokens if t.lower() in self.english_vocab)
        ratio = english_count / len(meaningful_tokens)
        return ratio >= threshold
    
    def clean_text(self, text, min_tokens=3):
        """Main text cleaning function"""
        if pd.isna(text):
            return None
        
        text = str(text).strip()
        
        # Early corruption detection
        if self.is_corrupted_text(text):
            return None
        
        # Expand contractions first
        try:
            text = contractions.fix(text)
        except:
            pass
        
        # Remove Reddit-specific elements
        text = re.sub(r'/u/\S+|u/\S+', '', text)  # usernames
        text = re.sub(r'/r/\S+|r/\S+', '', text)  # subreddit mentions
        text = re.sub(r'&gt;.*?(?=\n|$)', '', text, flags=re.MULTILINE)  # quoted text
        text = re.sub(r'edit\s*:', '', text, flags=re.IGNORECASE)  # edit markers
        
        # Remove URLs and links (more comprehensive)
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # markdown links
        text = re.sub(r'&lt;.*?&gt;', '', text)   # HTML-like tags
        text = re.sub(r'<.*?>', '', text)         # HTML tags
        
        # Remove conversion bot text patterns
        text = re.sub(r'\*\*[^*]+\*\*', '', text)  # bold markdown
        text = re.sub(r'^\s*•.*?$', '', text, flags=re.MULTILINE)  # bullet points
        text = re.sub(r'\^+', '', text)           # superscript markers
        
        # Remove numbers (keep short meaningful ones like blood types)
        text = re.sub(r'\b\d{4,}\b', '', text)    # long numbers
        text = re.sub(r'\d+\.\d+', '', text)      # decimals
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but preserve apostrophes temporarily
        text = re.sub(r"[^\w\s']", " ", text)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove contraction artifacts and non-alphabetic tokens
        tokens = [t for t in tokens if t not in self.contraction_artifacts]
        tokens = [t for t in tokens if t.isalpha()]
        tokens = [t for t in tokens if len(t) > 1 or t in ['a', 'i']]
        
        # Check if text is English
        if not self.is_english_text(tokens):
            return None
        
        # Remove stopwords
        tokens = [t for t in tokens if t not in self.stop_words]
        
        # Filter short words (keep meaningful ones)
        tokens = [t for t in tokens if len(t) > 2 or t in self.meaningful_short]
        
        # Lemmatization with POS tagging
        pos_tags = pos_tag(tokens)
        tokens = [self.lemmatizer.lemmatize(t, self.get_wordnet_pos(tag)) for t, tag in pos_tags]
        
        # Final cleanup
        tokens = [t for t in tokens if len(t) > 1 and t.isalpha()]
        
        # Quality checks on final tokens
        if tokens:
            # Check for excessive repetition in cleaned tokens
            token_counts = {}
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
            
            if token_counts:
                max_token_count = max(token_counts.values())
                if max_token_count / len(tokens) > 0.4:
                    return None
            
            # Check average token length
            avg_token_length = sum(len(token) for token in tokens) / len(tokens)
            if avg_token_length > 8:
                return None
        
        # Check minimum token requirement
        if len(tokens) < min_tokens:
            return None
        
        return " ".join(tokens)

def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent

def process_comments(input_file=None, output_file=None):
    """Process comments dataset"""
    if input_file is None:
        input_file = get_project_root() / "data" / "raw" / "Blooddonors_comments.csv"
    if output_file is None:
        output_file = get_project_root() / "data" / "cleaned" / "blooddonors_comments_processed.csv"
    
    print("Processing comments dataset...")
    
    # Load data
    print("Loading comments data...")
    try:
        df = pd.read_csv(input_file, usecols=["body"], on_bad_lines="skip", low_memory=False)
        print(f"Loaded {len(df):,} comments")
    except Exception as e:
        print(f"Error loading {input_file}: {e}")
        return
    
    # Initialize processor
    processor = RedditTextProcessor()
    
    # Clean the body text
    print("Cleaning comment text...")
    df["cleaned_body"] = df["body"].apply(lambda x: processor.clean_text(x, min_tokens=3))
    
    # Remove rows where cleaning failed
    initial_count = len(df)
    df_clean = df.dropna(subset=["cleaned_body"])
    final_count = len(df_clean)
    
    print(f"Kept {final_count:,} out of {initial_count:,} comments ({final_count/initial_count*100:.1f}%)")
    
    # Save only the cleaned column
    df_output = df_clean[["cleaned_body"]]
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_output.to_csv(output_file, index=False)
    
    print(f"Saved processed comments to {output_file}")
    
    # Show sample
    print("\nSample cleaned comments:")
    for i, text in enumerate(df_output["cleaned_body"].head(5)):
        print(f"{i+1}. {text}")
    
    return df_output

def process_submissions(input_file=None, output_file=None):
    """Process submissions dataset"""
    if input_file is None:
        input_file = get_project_root() / "data" / "raw" / "Blooddonors_submissions.csv"
    if output_file is None:
        output_file = get_project_root() / "data" / "cleaned" / "blooddonors_submissions_processed.csv"
    
    print("\nProcessing submissions dataset...")
    
    # Load data
    print("Loading submissions data...")
    try:
        df = pd.read_csv(input_file, usecols=["title", "selftext"], on_bad_lines="skip", low_memory=False)
        print(f"Loaded {len(df):,} submissions")
    except Exception as e:
        print(f"Error loading {input_file}: {e}")
        return
    
    # Initialize processor
    processor = RedditTextProcessor()
    
    # Process titles (allow shorter minimum tokens for titles)
    print("Cleaning submission titles...")
    df["cleaned_title"] = df["title"].apply(lambda x: processor.clean_text(x, min_tokens=2))
    
    # Process selftext
    print("Cleaning submission selftext...")
    df["cleaned_selftext"] = df["selftext"].apply(lambda x: processor.clean_text(x, min_tokens=3))
    
    # Statistics
    initial_count = len(df)
    titles_clean = df["cleaned_title"].notna().sum()
    selftext_clean = df["cleaned_selftext"].notna().sum()
    both_clean = ((df["cleaned_title"].notna()) & (df["cleaned_selftext"].notna())).sum()
    either_clean = ((df["cleaned_title"].notna()) | (df["cleaned_selftext"].notna())).sum()
    
    print(f"\nProcessing results:")
    print(f"Total submissions: {initial_count:,}")
    print(f"Clean titles: {titles_clean:,} ({titles_clean/initial_count*100:.1f}%)")
    print(f"Clean selftext: {selftext_clean:,} ({selftext_clean/initial_count*100:.1f}%)")
    print(f"Both clean: {both_clean:,} ({both_clean/initial_count*100:.1f}%)")
    print(f"At least one clean: {either_clean:,} ({either_clean/initial_count*100:.1f}%)")
    
    # Keep only rows where at least one column has clean content
    df_output = df[(df["cleaned_title"].notna()) | (df["cleaned_selftext"].notna())][["cleaned_title", "cleaned_selftext"]]
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_output.to_csv(output_file, index=False)
    print(f"Saved processed submissions to {output_file}")
    
    # Show samples
    print("\nSample cleaned titles:")
    sample_titles = df_output["cleaned_title"].dropna().head(5)
    for i, text in enumerate(sample_titles):
        print(f"{i+1}. {text}")
    
    print("\nSample cleaned selftext:")
    sample_selftext = df_output["cleaned_selftext"].dropna().head(3)
    for i, text in enumerate(sample_selftext):
        print(f"{i+1}. {text[:200]}{'...' if len(text) > 200 else ''}")
    
    return df_output

def main():
    """Main preprocessing function"""
    print("Blood Donation Reddit Text Preprocessing Pipeline")
    print("=" * 50)
    
    # Process comments
    comments_df = process_comments()
    
    # Process submissions
    submissions_df = process_submissions()
    
    print("\n" + "=" * 50)
    print("Processing complete!")
    
    if comments_df is not None:
        print(f"Comments processed: {len(comments_df):,} rows")
    if submissions_df is not None:
        print(f"Submissions processed: {len(submissions_df):,} rows")

if __name__ == "__main__":
    main()
