from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/LaBSE"
)

sentences = [
    "我喜欢人工智能",
    "I love artificial intelligence",
    "今天天气很好"
]

embeddings = model.encode(sentences)

print(embeddings.shape)