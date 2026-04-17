# Thesis Showcase: Reddit Text Analytics & Semantic Network Analysis

**Grade:** High Distinction (86/100)  
**Project Type:** Honours/Capstone Thesis Development  
**Thesis Link:** [520494600_thesis.pdf](https://drive.google.com/file/d/1Bw4ERtmT37SNejsqKsw-YQXtQoIPRxjk/view)

This repository documents my end-to-end thesis workflow for large-scale Reddit text analytics and semantic network analysis using Python.

## Thesis Highlights

- Built a full ETL and NLP pipeline in Python (`pandas`, `nltk`) to ingest, clean, and structure **75K+ Reddit records** (**66,964 comments + 8,665 submissions**) from `r/Blooddonors`, with **95%+ retained data quality** through custom bot filtering, spam detection, and Unicode corruption handling.
- Constructed semantic co-occurrence networks with **960 lexical nodes / 11,582 weighted edges** (submissions) and **1,245 nodes / 16,214 weighted edges** (comments), then applied Louvain community detection to extract **6 dominant thematic clusters**.
- Performed emotional profiling using the **NRC Emotion Lexicon** plus null-model benchmarking, identifying strong anticipation/trust signals in submissions and a trust-shift in comments consistent with peer-support dynamics.
- Identified the gratitude-focused cluster as the strongest positive emotional segment (**Trust 34.2%, Joy 21.1%**), supporting actionable donor-retention strategy insights for blood collection organisations.

## Project Structure

All implementation code is under:

- [`processing bi-gram approach/`](./processing%20bi-gram%20approach/)

Core pipeline scripts:

- [`processing bi-gram approach/main.py`](./processing%20bi-gram%20approach/main.py) - orchestrates the full workflow.
- [`processing bi-gram approach/scripts/preprocessing.py`](./processing%20bi-gram%20approach/scripts/preprocessing.py) - ETL cleaning, filtering, tokenisation, and lemmatisation.
- [`processing bi-gram approach/scripts/matrix_construction.py`](./processing%20bi-gram%20approach/scripts/matrix_construction.py) - bi-gram extraction and co-occurrence matrix construction.
- [`processing bi-gram approach/scripts/network_construction.py`](./processing%20bi-gram%20approach/scripts/network_construction.py) - weighted semantic network construction.
- [`processing bi-gram approach/scripts/community_detection.py`](./processing%20bi-gram%20approach/scripts/community_detection.py) - Louvain clustering, centrality analysis, Cytoscape exports.
- [`processing bi-gram approach/scripts/emotional_profiling.py`](./processing%20bi-gram%20approach/scripts/emotional_profiling.py) - NRC emotion and VAD annotation of semantic nodes.
- [`processing bi-gram approach/integrate_communities.py`](./processing%20bi-gram%20approach/integrate_communities.py) - consolidated community-level node integration.

## How Each Thesis Claim Maps to Code

### 1) ETL + Data Quality Pipeline (75K+ records, cleaning + retention)

- [`scripts/preprocessing.py`](./processing%20bi-gram%20approach/scripts/preprocessing.py): corruption checks, deletion-marker handling, spam/bot filters, token-level quality controls.
- [`main.py`](./processing%20bi-gram%20approach/main.py): end-to-end orchestration and reproducible stepwise execution.

### 2) Semantic Networks + Community Detection

- [`scripts/matrix_construction.py`](./processing%20bi-gram%20approach/scripts/matrix_construction.py): generates co-occurrence matrices and pair-frequency tables.
- [`scripts/network_construction.py`](./processing%20bi-gram%20approach/scripts/network_construction.py): converts pair tables into weighted GraphML networks.
- [`scripts/community_detection.py`](./processing%20bi-gram%20approach/scripts/community_detection.py): applies Louvain clustering and exports community-level analytics.

### 3) Emotional Profiling + Null-Model Benchmarks

- [`scripts/emotional_profiling.py`](./processing%20bi-gram%20approach/scripts/emotional_profiling.py): emotion/VAD lexicon mapping and emotional summary outputs.
- [`scripts/circumplex_model.py`](./processing%20bi-gram%20approach/scripts/circumplex_model.py): circumplex-level emotional modelling support.

### 4) Community-Level Insights for Donor Retention

- [`integrate_communities.py`](./processing%20bi-gram%20approach/integrate_communities.py): merges nodes with community IDs for downstream interpretation/reporting.
- [`scripts/community_detection.py`](./processing%20bi-gram%20approach/scripts/community_detection.py): centrality-ranked lexical themes and per-community exports.

## Quick Start

From the project folder:

```bash
cd "processing bi-gram approach"
python main.py
```

Optional flags:

```bash
python main.py --skip-preprocessing --skip-matrix --skip-network --skip-community --skip-emotional
```

## Portfolio Repository Note

To keep this repository lightweight and suitable for GitHub portfolio review:

- Raw/cleaned Reddit datasets are excluded.
- Generated results and visualization exports are excluded.

## Outputs

The pipeline writes outputs to:

- `processing bi-gram approach/results/matrix/`
- `processing bi-gram approach/results/graph/`
- `processing bi-gram approach/results/communities/`
- `processing bi-gram approach/results/table/`
- `processing bi-gram approach/results/emotional_profiling/`

---

If you are viewing this repository for portfolio purposes, start with:

1. [`main.py`](./processing%20bi-gram%20approach/main.py)
2. [`scripts/preprocessing.py`](./processing%20bi-gram%20approach/scripts/preprocessing.py)
3. [`scripts/community_detection.py`](./processing%20bi-gram%20approach/scripts/community_detection.py)
4. [`scripts/emotional_profiling.py`](./processing%20bi-gram%20approach/scripts/emotional_profiling.py)
