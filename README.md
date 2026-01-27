# NeurIPS 2025 Backend

FastAPI service that powers search and summarization over a NeurIPS 2025 paper dataset using hybrid retrieval (dense, sparse, fuzzy) and LLM refinement.

## Tech Stack

- FastAPI + Uvicorn
- Python (pandas, numpy)
- ML/NLP: Hugging Face Transformers (SPECTER2), Sentence-Transformers CrossEncoder, BM25, RapidFuzz
- OpenAI API for query refinement and summary generation