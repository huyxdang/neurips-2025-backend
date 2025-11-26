from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModel

print("Pre-downloading models...")
AutoTokenizer.from_pretrained("allenai/specter2_base")
AutoModel.from_pretrained("allenai/specter2_base")
CrossEncoder('mixedbread-ai/mxbai-rerank-xsmall-v1', trust_remote_code=True)
print("Done!")

