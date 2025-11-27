import os
import pandas as pd
import numpy as np
import re
import hashlib
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from rapidfuzz import process, fuzz
from transformers import AutoTokenizer, AutoModel
import torch

load_dotenv()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://neurips-2025-map.vercel.app",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app", 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- CONFIGURATION ---
STOP_WORDS = set(["the", "a", "an", "and", "in", "on", "at", "to", "for", "of", "with", "is", "are"])
MIN_RELEVANCE_SCORE = 0.55

def preprocess_text(text: str):
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return [t for t in tokens if t not in STOP_WORDS]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# --- 1. LOAD DATA ---
print("⏳ Loading Data...")
df = pd.read_parquet("neurips_data.parquet")

required_columns = ['paper', 'authors', 'abstract', 'link', 'track', 'award', 
                    'paper_id', 'embedding', 'cluster', 'umap_x', 'umap_y', 
                    'cluster_name', 'problem', 'solution', 'eli5', 'tldr', 'keywords']
missing = [col for col in required_columns if col not in df.columns]
if missing:
    print(f"⚠️ Warning: Missing columns: {missing}")

print(f"✅ Loaded {len(df)} papers")
print(f"   Columns: {list(df.columns)}")

# Pre-compute titles for fuzzy matching
titles_list = df['paper'].tolist()

# --- 2. LOAD MODELS ---
print("⏳ Loading SPECTER2 Model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
specter_tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
specter_model = AutoModel.from_pretrained("allenai/specter2_base").to(device)
specter_model.eval()
print(f"   SPECTER2 loaded on {device}")

# Load pre-computed embeddings
print("⏳ Loading Pre-computed Embeddings...")
embeddings = np.stack(df['embedding'].values).astype('float32')
print(f"   Embedding shape: {embeddings.shape}")

print("⏳ Building Keyword Index...")
corpus_text = (df['paper'] + " " + df['abstract']).tolist()
tokenized_corpus = [preprocess_text(doc) for doc in corpus_text]
bm25 = BM25Okapi(tokenized_corpus)

print("⏳ Loading Re-Ranker...")
reranker = CrossEncoder('mixedbread-ai/mxbai-rerank-xsmall-v1', trust_remote_code=True)

print("✅ Server Ready!")

# Simple in-memory cache
query_cache = {}


def embed_query_specter2(query: str) -> np.ndarray:
    """Embed a query using SPECTER2 to match paper embeddings"""
    inputs = specter_tokenizer(
        [query],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = specter_model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    
    return embedding.astype('float32')


def refine_query_with_llm(query: str) -> tuple[str, bool]:
    """
    Use LLM to:
    1. Extract core research topic
    2. Detect if query is a valid research paper search
    
    Returns: (refined_query, is_valid_research_query)
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "system", 
                "content": """You help users search for machine learning research papers at NeurIPS.

TASK 1: Is this a request to find research papers on a topic?
- YES if: asking about ANY scientific/technical topic, field, method, model, algorithm, problem, application domain
- NO if: asking about logistics, travel, dates, registration, locations, deadlines, career advice, personal questions

Be PERMISSIVE. If someone mentions ANY topic that could have research papers, say YES.

TASK 2: If YES, extract 2-5 search keywords.

RESPOND IN THIS EXACT FORMAT (two lines only):
VALID: yes
KEYWORDS: <keywords>

OR:
VALID: no
KEYWORDS: none

Examples:
"papers on transformers" → VALID: yes | KEYWORDS: transformers
"biology" → VALID: yes | KEYWORDS: biology
"find me papers on biology" → VALID: yes | KEYWORDS: biology
"RLHF" → VALID: yes | KEYWORDS: RLHF reinforcement learning human feedback
"how to get to neurips" → VALID: no | KEYWORDS: none
"when is the deadline" → VALID: no | KEYWORDS: none
"protein folding" → VALID: yes | KEYWORDS: protein folding
"climate" → VALID: yes | KEYWORDS: climate
"what's new in LLMs" → VALID: yes | KEYWORDS: large language models LLM"""
            }, {
                "role": "user", 
                "content": query
            }],
            max_tokens=50,
            temperature=0
        )
        
        result = response.choices[0].message.content.strip()
        print(f"   └─ LLM response: {result}")  # Debug logging
        
        # Parse response - be lenient
        result_lower = result.lower()
        is_valid = "valid: yes" in result_lower or "valid:yes" in result_lower
        
        # Extract keywords
        if "keywords:" in result_lower:
            keywords_part = result.split(":")[-1].strip()
            # Clean up keywords
            keywords = keywords_part.lower().replace("none", "").replace("|", "").strip()
        else:
            keywords = ""
        
        # Use original query if no keywords extracted
        refined = keywords if keywords and keywords != "none" else query
        
        return (refined, is_valid)
        
    except Exception as e:
        print(f"⚠️ LLM refinement failed: {e}")
        return (query, True)  # Assume valid on error


class SearchRequest(BaseModel):
    query: str


@app.post("/search")
async def search(request: SearchRequest):
    original_query = request.query
    
    print("\n" + "="*80)
    print(f"🔍 ORIGINAL QUERY: '{original_query}'")
    print("="*80)
    
    # --- QUERY REFINEMENT + INTENT DETECTION ---
    query, is_valid_research_query = refine_query_with_llm(original_query)
    
    print(f"✨ REFINED QUERY: '{query}'")
    print(f"   └─ Valid research query: {'YES' if is_valid_research_query else 'NO'}")
    
    # Reject non-research queries early
    if not is_valid_research_query:
        print("❌ REJECTED: Not a valid research paper query")
        return {
            "text": "I can only help you find research papers. Try searching for topics like 'diffusion models', 'reinforcement learning', or 'vision transformers'.",
            "relatedIds": [],
            "relatedPapers": []
        }
    
    query_clean = query.lower().strip()
    query_hash = hashlib.md5(query_clean.encode()).hexdigest()
    
    # Check cache
    if query_hash in query_cache:
        print("\n" + "="*80)
        print(f"🎯 CACHE HIT for refined query: '{query_clean}'")
        print("="*80 + "\n")
        return query_cache[query_hash]

    print("\n🔬 PRE-FLIGHT: Semantic Validation...")

    query_vec = embed_query_specter2(query)
    dataset_mean_embedding = embeddings.mean(axis=0, keepdims=True).astype('float32')
    semantic_similarity = cosine_similarity(query_vec, dataset_mean_embedding)[0][0]

    print(f"   📊 Semantic alignment with dataset: {semantic_similarity:.4f}")

    SEMANTIC_THRESHOLD = 0.5

    if semantic_similarity < SEMANTIC_THRESHOLD:
        print(f"   ❌ REJECTED: Query too semantically distant from NeurIPS papers")
        print(f"      (similarity={semantic_similarity:.4f} < threshold={SEMANTIC_THRESHOLD})")
        return {
            "text": "Sorry, your query doesn't seem to match any research topics in the dataset.",
            "relatedIds": [],
            "relatedPapers": []
        }

    print(f"   ✅ PASSED: Query is semantically aligned")
    
    if len(query_clean) < 3:
        print("❌ Query too short, rejecting...")
        return {
            "text": "Sorry, your query doesn't seem to match any research topics in the dataset.",
            "relatedIds": [],
            "relatedPapers": []
        }
 
    # --- STEP 1: FUZZY MATCHING ---
    print("\n📝 STEP 1: Fuzzy Title Matching...")
    fuzzy_matches = process.extract(
        query, 
        titles_list, 
        scorer=fuzz.partial_ratio, 
        limit=5
    )
    fuzzy_indices = [match[2] for match in fuzzy_matches if match[1] > 85]
    print(f"   └─ Found {len(fuzzy_indices)} fuzzy matches (score > 85)")
    if fuzzy_indices:
        for match in fuzzy_matches[:3]:
            if match[1] > 85:
                print(f"      • {match[0][:60]}... (score: {match[1]})")

    # --- STEP 2: HYBRID RETRIEVAL ---
    print("\n🔎 STEP 2: Hybrid Retrieval...")
    
    # A. Dense (using SPECTER2)
    dense_scores = cosine_similarity(query_vec, embeddings)[0]
    dense_top_k = np.argsort(dense_scores)[::-1][:30]
    best_dense_score = dense_scores[dense_top_k[0]]
    print(f"   📊 Dense (SPECTER2 Vector): Top-30 retrieved")
    print(f"      • Best score: {best_dense_score:.4f}")
    print(f"      • 30th score: {dense_scores[dense_top_k[-1]]:.4f}")
    
    # B. Sparse
    tokenized_query = preprocess_text(query)
    if not tokenized_query: 
        tokenized_query = query.lower().split()
    sparse_scores = bm25.get_scores(tokenized_query)
    sparse_top_k = np.argsort(sparse_scores)[::-1][:30]
    print(f"   📊 Sparse (BM25): Top-30 retrieved")
    print(f"      • Tokens used: {tokenized_query}")
    print(f"      • Best score: {sparse_scores[sparse_top_k[0]]:.4f}")
    print(f"      • 30th score: {sparse_scores[sparse_top_k[-1]]:.4f}")
    
    # C. Union
    candidate_indices = list(set(fuzzy_indices) | set(dense_top_k) | set(sparse_top_k))
    print(f"   🔗 Combined: {len(candidate_indices)} unique candidates")
    
    # [OPTIMIZATION 2] Reduce candidates from 15 to 10
    if len(candidate_indices) > 10:
        dense_set = set(dense_top_k[:7])
        sparse_set = set(sparse_top_k[:5]) 
        fuzzy_set = set(fuzzy_indices)
        candidate_indices = list(dense_set | sparse_set | fuzzy_set)[:10]
        print(f"      └─ Capped to 10 for efficiency")
        
    if not candidate_indices:
        print("❌ No candidates found!")
        return {
            "text": "I couldn't find any papers matching that topic.",
            "relatedIds": [],
            "relatedPapers": []
        }

    # --- STEP 3: RE-RANKING ---
    # [OPTIMIZATION 3] Skip reranking when dense score is very high
    SKIP_RERANK_THRESHOLD = 0.85
    
    if best_dense_score > SKIP_RERANK_THRESHOLD:
        print(f"\n⚡ SKIP RERANK: Dense score {best_dense_score:.4f} > {SKIP_RERANK_THRESHOLD}")
        print(f"   Using dense retrieval scores directly...")
        scored_candidates = [(idx, float(dense_scores[idx])) for idx in dense_top_k[:10]]
    else:
        print("\n🎯 STEP 3: Re-ranking with CrossEncoder...")
        candidates = df.iloc[candidate_indices]
        
        # Use problem + solution (more concise and information-dense than abstract)
        cross_input = [
            [query, f"{row['paper']}: {row['problem']} {row['solution']}"] 
            for _, row in candidates.iterrows()
        ]
        
        rerank_logits = reranker.predict(cross_input, batch_size=32)
        print(f"   📊 Raw Logits Stats:")
        print(f"      • Shape: {rerank_logits.shape}")
        print(f"      • Min: {rerank_logits.min():.4f}")
        print(f"      • Max: {rerank_logits.max():.4f}")
        print(f"      • Mean: {rerank_logits.mean():.4f}")
        
        rerank_probs = sigmoid(rerank_logits)
        print(f"   📊 After Sigmoid:")
        print(f"      • Min: {rerank_probs.min():.4f}")
        print(f"      • Max: {rerank_probs.max():.4f}")
        print(f"      • Mean: {rerank_probs.mean():.4f}")
        
        scored_candidates = []
        for idx, prob in zip(candidate_indices, rerank_probs):
            scored_candidates.append((idx, prob))
            
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Show top 10 after reranking
    print(f"\n   🏆 Top 10 After Scoring:")
    for i, (idx, score) in enumerate(scored_candidates[:10], 1):
        paper_title = df.iloc[idx]['paper'][:60]
        print(f"      {i:2d}. {paper_title}... ({score:.4f})")
    
    # --- STEP 4: FILTERING ---
    print("\n🚦 STEP 4: Relevance Filtering...")
    
    best_score = scored_candidates[0][1]
    top_3_avg = np.mean([x[1] for x in scored_candidates[:3]])
    
    print(f"   📊 Quality Metrics:")
    print(f"      • Best score: {best_score:.4f}")
    print(f"      • Top-3 avg: {top_3_avg:.4f}")
    print(f"      • Threshold: {MIN_RELEVANCE_SCORE:.2f}")

    if best_score < MIN_RELEVANCE_SCORE or top_3_avg < 0.55:
        print(f"   ❌ REJECTED: Scores too low (best={best_score:.4f}, top3_avg={top_3_avg:.4f})")
        return {
            "text": "Sorry, your query doesn't seem to match any research topics in the dataset.",
            "relatedIds": [],
            "relatedPapers": []
        }
    
    print(f"   ✅ PASSED: Query is relevant!")
    
    filtered = [x for x in scored_candidates if x[1] > MIN_RELEVANCE_SCORE]
    print(f"   📋 Papers above {MIN_RELEVANCE_SCORE}: {len(filtered)}")
    
    final_indices = []
    final_scores = {}
    
    amazing_matches = [x for x in filtered if x[1] > 0.70]
    
    if len(amazing_matches) > 0:
        print(f"   🌟 Found {len(amazing_matches)} AMAZING matches (>0.70)")
        final_indices = [x[0] for x in amazing_matches[:15]]
        for idx, score in amazing_matches[:15]:
            paper_id = df.iloc[idx]['paper_id']
            final_scores[str(paper_id)] = float(score)
    else:
        print(f"   ⭐ No amazing matches, showing {min(len(filtered), 10)} GOOD matches")
        final_indices = [x[0] for x in filtered[:10]]
        for idx, score in filtered[:10]:
            paper_id = df.iloc[idx]['paper_id']
            final_scores[str(paper_id)] = float(score)
        
    final_results = df.iloc[final_indices].copy()
    print(f"   ✅ Final selection: {len(final_results)} papers")
    
    # --- STEP 5: GENERATION ---
    print("\n💬 STEP 5: Generating Summary with GPT...")
    
    context_text = ""
    for i, row in enumerate(final_results.itertuples(), 1):
        summary = row.tldr if pd.notna(row.tldr) and row.tldr else row.eli5
        context_text += f"[{i}] Title: {row.paper}\nSummary: {summary}\n\n"

    num_papers = len(final_results)
    
    system_prompt = f"""
    You are a senior research analyst for NeurIPS.
    
    GOAL: Help the user discover relevant papers by grouping them into technical themes.

    CRITICAL: You MUST mention exactly {num_papers} paper(s) in your response. Count them carefully.

    FORMATTING RULES:
    1. Start with a single sentence: "I found {num_papers} papers related to [Topic]."
    2. Use **Markdown Bullet Points** to display ALL {num_papers} paper(s) one by one, along with their easy-to-understand summary.
    3. **Bold** the paper's name (e.g., "**Diffusion Sampling:**").
    4. Write a concise 1-sentence description of the paper's contribution.
    5. You MUST list all {num_papers} paper(s) provided - do not skip any.

    Example Output (if 5 papers):
    I found 5 papers on Biology.

    * **HiPoNet: A Multi-View Simplicial Complex Network for High Dimensional Point-Cloud and Single-Cell data:** Introduces a novel neural network for feature extraction of geometric information. 
    * **Rationalized All-Atom Protein Design with Unified Multi-Modal Bayesian Flow:** A new method that makes designing proteins easier. 
    * **[Paper 3 Title]:** [Description] 
    * **[Paper 4 Title]:** [Description] 
    * **[Paper 5 Title]:** [Description] 
    """

    try:
        gpt_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {query}\n\nPapers:\n{context_text}"}
            ]
        )
        answer = gpt_response.choices[0].message.content
        print(f"   ✅ Generated {len(answer)} chars")
    except Exception as e:
        print(f"   ⚠️ GPT generation failed: {e}")
        answer = "I found relevant papers, but couldn't generate a summary."

    papers_data = final_results[['paper_id', 'paper', 'authors', 'cluster_name', 
                                  'problem', 'solution', 'eli5', 'tldr', 'keywords',
                                  'abstract', 'link', 'track', 'award']].copy()
    
    related_papers = []
    for _, row in papers_data.iterrows():
        paper_id_str = str(row['paper_id']) if pd.notna(row['paper_id']) else ''
        
        authors = row['authors']
        if isinstance(authors, str):
            try:
                import ast
                authors = ast.literal_eval(authors)
            except:
                authors = [a.strip() for a in authors.split(',') if a.strip()]
        elif not isinstance(authors, list):
            authors = []

        related_papers.append({
            'paper_id': paper_id_str,
            'title': str(row['paper']) if pd.notna(row['paper']) else '',
            'paper': str(row['paper']) if pd.notna(row['paper']) else '',
            'authors': authors,
            'cluster_name': str(row['cluster_name']) if pd.notna(row['cluster_name']) else '',
            'problem': str(row['problem']) if pd.notna(row['problem']) else '',
            'solution': str(row['solution']) if pd.notna(row['solution']) else '',
            'eli5': str(row['eli5']) if pd.notna(row['eli5']) else '',
            'tldr': str(row['tldr']) if pd.notna(row['tldr']) else '',
            'keywords': str(row['keywords']) if pd.notna(row['keywords']) else '',
            'abstract': str(row['abstract']) if pd.notna(row['abstract']) else '',
            'link': str(row['link']) if pd.notna(row['link']) else '',
            'track': str(row['track']) if pd.notna(row['track']) else '',
            'award': str(row['award']) if pd.notna(row['award']) else '',
            'score': final_scores.get(paper_id_str, 0.0)
        })
    
    print(f"\n✅ COMPLETE: Returning {len(related_papers)} papers")
    print("="*80 + "\n")
    
    result = {
        "text": answer,
        "relatedIds": [str(pid) for pid in final_results['paper_id'].tolist()],
        "relatedPapers": related_papers,
        "bestScore": float(best_score)
    }
    
    if len(query_cache) < 100:
        query_cache[query_hash] = result
        print(f"💾 Cached result for refined query: '{query_clean}' (cache size: {len(query_cache)}/100)")
    else:
        print(f"⚠️ Cache full (100 entries), not caching this result")
    
    return result


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "papers_loaded": len(df),
        "embedding_shape": embeddings.shape,
        "model": "allenai/specter2_base"
    }