# Agentic Biomarker Discovery

> An agentic RAG framework for biomedical literature retrieval, evaluated on the BioASQ
> Task 8B benchmark. The project investigates retrieval quality, the emergence of tool
> use with model scale, the gap between answer relevance and answer faithfulness ("the
> Grounding Paradox"), and whether a knowledge graph layer can recover what dense
> retrieval misses.

## Results Summary

| Finding | Headline number |
|---|---|
| **Retrieval is solved with the right embedding** | Recall@5: 0.567 (base PubMedBERT) → **0.960** (S-PubMedBert-MS-MARCO) |
| **Tool use is emergent at scale, not a prompting artifact** | Identical ReAct prompts: **3.1% hop rate** on Llama-3.1-8B → **19.5%** on Llama-3.3-70B |
| **The Grounding Paradox** | Answer Relevancy stays > 0.90 while Faithfulness sits at 0.07–0.13 — RAGAS exposes what EM/F1 cannot |
| **A knowledge graph layer recovers what FAISS misses** | Context Precision: 0.105 → **0.173** with cross-encoder reranking + KG fusion |

## What's in here

Five Jupyter notebooks built to be run end-to-end on Google Colab (one H100 or A100):

1. **`01_data_and_faiss_index.ipynb`** — Parse the BioASQ Task 8B corpus, embed ~39K snippets
   with a biomedical sentence encoder, build a FAISS IndexFlatIP, evaluate retrieval quality
   with Recall@k on a held-out test set.
2. **`02_react_agent_ablation.ipynb`** — Four-condition controlled ablation isolating the
   effect of retrieval (zero-shot vs single-pass RAG) and the effect of model scale on
   emergent tool use (Llama-3.1-8B vs Llama-3.3-70B).
3. **`03_ragas_evaluation_and_demo.ipynb`** — RAGAS evaluation (Faithfulness, Answer
   Relevancy, Context Precision) on the ablation conditions. Surfaces the Grounding
   Paradox and a three-type error taxonomy.
4. **`04_knowledge_graph_construction.ipynb`** — Builds a biomedical co-occurrence
   knowledge graph (2,082 nodes, 8,173 edges, clustering coefficient 0.538 — a small-world
   topology) from the same corpus and analyzes structural properties.
5. **`05_hybrid_conversational_agent.ipynb`** — Multi-turn conversational agent with
   hybrid FAISS + KG retrieval, cross-encoder reranking, and subject-aware query
   reformulation. Demonstrated on an eight-question clinical narrative.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    BioASQ Task 8B Corpus                        │
│              (3,243 training Qs, ~39K snippets)                 │
└──────────────────────────────┬──────────────────────────────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                 │
              ▼                                 ▼
┌─────────────────────────────┐   ┌────────────────────────────┐
│   FAISS IndexFlatIP         │   │  Co-occurrence Knowledge   │
│   S-PubMedBert-MS-MARCO     │   │  Graph (NetworkX)          │
│   dim=768                   │   │  2,082 nodes / 8,173 edges │
└──────────────┬──────────────┘   └─────────────┬──────────────┘
               │                                │
               └──────────────┬─────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │   Hybrid ReAct Agent          │
              │   ─ search() (FAISS)          │
              │   ─ graph_search() (KG)       │
              │   ─ cross-encoder reranking   │
              │   ─ subject-aware reformul.   │
              └───────────────┬───────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │   Two-Tier Evaluation         │
              │   ─ EM / F1 / Hops            │
              │   ─ RAGAS (Faithfulness,      │
              │     Answer Relevancy,         │
              │     Context Precision)        │
              └───────────────────────────────┘
```

## Key Findings in Detail

### 1. Embedding choice is the single highest-leverage decision

Three embedding models tested on the same FAISS pipeline:

| Model | Recall@5 | Notes |
|---|---:|---|
| `all-MiniLM-L6-v2` (general) | 0.923 | Strong general-purpose baseline |
| `BiomedBERT-base` (domain) | 0.567 | Anisotropy collapse — vectors cluster in a narrow cone |
| `S-PubMedBert-MS-MARCO` | **0.960** | Biomedical vocabulary + retrieval fine-tuning |

The takeaway is non-obvious: domain vocabulary alone *hurts* if the model isn't fine-tuned on a
retrieval objective. Base biomedical BERT trained on masked-language modeling produces
anisotropic embeddings that collapse cosine similarity into noise. Retrieval-objective
fine-tuning is mandatory.

### 2. Tool use emerges at scale

Identical ReAct prompts produced wildly different behavior across model sizes:

| Model | Tool-use hop rate | Best EM | Best F1 |
|---|---:|---:|---:|
| Llama-3.1-8B (4-bit local) | 3.1% | 0.092 | 0.281 |
| Llama-3.3-70B (Together.ai) | **19.5%** | **0.106** | **0.335** |

Anti-collapse injection (forcing search on the first turn) narrowed but did not close the gap.
Model selection is a first-order concern for production agents — not a hyperparameter tweak.

### 3. The Grounding Paradox

The RAGAS evaluation surfaced a finding that EM/F1 alone could not:

| Condition | Answer Relevancy | Faithfulness | Context Precision |
|---|---:|---:|---:|
| ReAct 8B | 0.931 | 0.070 | 0.105 |
| ReAct 70B | 0.828 | 0.132 | 0.173 |

Models produce fluent, relevant-sounding answers (Answer Relevancy > 0.90), but those answers
are largely **not grounded in the retrieved evidence** (Faithfulness < 0.15). They're being
generated from training memory rather than from the snippets the agent actually pulled. This
is invisible to EM/F1 — both metrics reward a correct answer regardless of where it came
from — and arguably represents the central open problem for biomedical agents.

### 4. A knowledge graph layer helps

The hybrid agent in Notebook 5 fuses FAISS dense retrieval with a co-occurrence KG that
captures structural relationships dense embeddings miss. Combined with cross-encoder
reranking, this lifts Context Precision from 0.105 to 0.173 (+65%). Subject-aware query
reformulation (e.g., "EGFR" → "EGFR AND colorectal cancer" when context shifts) prevents
context poisoning across multi-turn conversations.

## How to Run

The notebooks are designed for Google Colab (Pro recommended for the GPU tiers).

### Prerequisites

- Python 3.10 or 3.11
- An NVIDIA GPU with at least 40 GB VRAM (Colab A100 minimum; H100 ideal for NB1/NB4/NB5)
- API keys (loaded from Colab Secrets or environment variables):
  - **Together.ai** — for Llama-3.3-70B inference (Condition D in NB2, optional in NB5)
  - **OpenAI** — for RAGAS evaluation (NB3 only)

### Setup

```bash
pip install -r requirements.txt
```

The notebooks install their own dependencies in Cell 1, so this file is provided
primarily for graders/reviewers who want to pre-install the environment.

### Data

This project uses [BioASQ Task 8B](http://bioasq.org/participate). The full corpus is not
bundled in this repository — BioASQ requires registration and the data is not licensed for
redistribution.

To run the notebooks end-to-end:

1. Register at [participants-area.bioasq.org](https://participants-area.bioasq.org/datasets/)
2. Download Task 8B (`training8b.json` and the five `8B*_golden.json` test batches)
3. Place the files in a `data/` directory at the repo root
4. Run notebooks in order: NB1 → NB2 → NB3 → NB4 → NB5

NB1 produces the FAISS index that subsequent notebooks depend on; NB4 produces the
knowledge graph that NB5 depends on.

### Run order

| Notebook | Hardware | Approx. runtime | Output |
|---|---|---|---|
| 01 — Data & FAISS Index | H100 / A100 | ~15 min | FAISS index, embeddings cache |
| 02 — ReAct Ablation | A100 | ~60 min | Four-condition results JSON |
| 03 — RAGAS & Demo | A100 | ~20 min | RAGAS scores, error taxonomy |
| 04 — Knowledge Graph | H100 / A100 | ~25 min | NetworkX KG (pickle) |
| 05 — Hybrid Agent | H100 | ~10 min | Eight-turn clinical demo |

## Tech Stack

- **Embeddings & retrieval:** `sentence-transformers`, `faiss-cpu`
- **Local inference:** `transformers`, `bitsandbytes` (4-bit NF4), `accelerate`
- **Remote inference:** Together.ai (Llama-3.3-70B), OpenAI (RAGAS judge)
- **Evaluation:** `ragas==0.2.6`, `datasets`
- **Knowledge graph:** `networkx`
- **Analysis & plots:** `numpy`, `matplotlib`, `tqdm`

## Limitations & Future Work

This is a research prototype, not a clinical tool. Key limitations to be aware of:

- **High Answer Relevancy can sound authoritative even when ungrounded** (Faithfulness < 0.15).
  Outputs should be treated as computational hypotheses requiring wet-lab validation, not as
  clinical findings.
- **BioASQ corpus skew:** oncology and Western populations are overrepresented. The agent's
  knowledge is biased toward what the underlying literature emphasizes.
- **No negation detection:** the agent treats "X is not associated with Y" identically to
  "X is associated with Y" because the retrieval embedding has no sign awareness. This is the
  top-priority follow-up.

Planned extensions:

1. **Negation classifier** to flag "not associated with" snippets before generation
2. **Production graph database** (Neo4j) with directed edges (REGULATES, INHIBITS, EXPRESSES) and UMLS/MeSH normalization
3. **Full KG corpus** on all 38.9K snippets (currently constructed on a subset)
4. **Condition E** — third data point on scale with Llama-3.1-405B

## License

MIT — see `LICENSE` file.

## Acknowledgements

- BioASQ Task 8B (Tsatsaronis et al.) — the benchmark dataset
- ReAct (Yao et al., 2023) — the reasoning-and-acting framework
- RAGAS (Es et al., 2024) — retrieval-aware evaluation
- S-PubMedBert-MS-MARCO (Deka & Kalita) — the embedding model that made retrieval work
- Llama 3 (Meta AI) — the open-weight models powering the agent
- Barabási & Albert (1999) — the network-topology lens used in NB4
