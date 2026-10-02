import os
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Windows 无符号链接权限，huggingface_hub 会退化为复制缓存，静默该提示
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/LaBSE"
)

sentences = [
    "我喜欢人工智能",
    "I love artificial intelligence",
    "今天天气很好"
]

embeddings = model.encode(
    sentences,
    normalize_embeddings=False
)
print(embeddings.shape)

# 最后统一 FP32 normalize
embeddings = embeddings.astype(np.float32)
embeddings /= np.linalg.norm(
    embeddings,
    axis=1,
    keepdims=True
)

similarity = embeddings @ embeddings.T
print(similarity)

# 相似度矩阵热力图（encode 默认输出归一化向量，点积即余弦相似度）
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 中文显示
plt.rcParams["axes.unicode_minus"] = False

ax = sns.heatmap(
    similarity,
    annot=True, fmt=".3f",
    cmap="Blues", vmin=0, vmax=1,  # 论文常用单色顺序色阶，灰度打印也清晰
    xticklabels=sentences, yticklabels=sentences,
    square=True,
    linewidths=0.5, linecolor="white",  # 白色分隔线，贴近期刊排版
    cbar_kws={"shrink": 0.8, "label": "Cosine Similarity"},
    annot_kws={"size": 11},
)
# 深色格子用白字、浅色格子用黑字，保证可读性
for text in ax.texts:
    text.set_color("white" if float(text.get_text()) > 0.6 else "black")

plt.title("句子余弦相似度矩阵")
plt.tight_layout()

out_dir = Path(__file__).resolve().parents[1] / "outputs"
out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(out_dir / "similarity_matrix.png", dpi=300)
plt.show()