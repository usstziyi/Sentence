import os
from pathlib import Path

# Hugging Face 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Windows 无符号链接权限提示
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from sentence_transformers import SentenceTransformer


# ============================================================
# 1. 加载 GTE Multilingual Base
# ============================================================

model = SentenceTransformer(
    "Alibaba-NLP/gte-multilingual-base",
    trust_remote_code=True
)


# ============================================================
# 1.1 重建非持久化 buffer
# ============================================================
#
# 模型自带的 modeling.py 里有几个 persistent=False 的 buffer：
#   - NewEmbeddings.position_ids
#   - RoPE 的 inv_freq / cos_cached / sin_cached
#
# 它们不写入 checkpoint。transformers 5.x 默认先在 meta device 上
# 建模型再 to_empty() 实体化，这些 buffer 因此没有被任何真实数值
# 覆盖，读到的是未初始化内存（表现为随机大整数），最终导致
#   IndexError: index xxx is out of bounds for dimension 0 with size 7
#
# 加载完成后按原公式重建它们即可。

_config = model[0].auto_model.config
_max_pos = getattr(_config, "max_position_embeddings", 8192)

for _module in model.modules():
    if "position_ids" in getattr(_module, "_buffers", {}):
        _module.position_ids = torch.arange(
            _max_pos,
            dtype=torch.long,
            device=_module.position_ids.device
        )

    # NTKScalingRotaryEmbedding 会在这里重算 inv_freq 与 cos/sin 缓存
    if hasattr(_module, "_set_cos_sin_cache"):
        _module._set_cos_sin_cache(
            seq_len=_module.max_seq_len_cached,
            device=_module.inv_freq.device,
            dtype=torch.get_default_dtype()
        )


# ============================================================
# 1.2 补回 transformers 5.x 移除的 API
# ============================================================
#
# 远程代码 modeling.py 仍在调用 transformers 4.x 的
# PreTrainedModel.get_extended_attention_mask()，该 API 在
# transformers 5 中已被删除。这里按 4.x 的语义补回来。

_model_cls = type(model[0].auto_model)

if not hasattr(_model_cls, "get_extended_attention_mask"):

    def _get_extended_attention_mask(
        self,
        attention_mask,
        input_shape,
        device=None,
        dtype=None,
        is_decoder=False,
        **kwargs
    ):
        dtype = dtype if dtype is not None else self.dtype

        if attention_mask.dim() == 3:
            extended = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for attention_mask {attention_mask.shape}"
            )

        extended = extended.to(device=extended.device, dtype=dtype)

        # 可见位置填 0，被屏蔽位置填该 dtype 的最小值
        extended = torch.where(
            extended.bool(),
            torch.tensor(0.0, dtype=dtype, device=extended.device),
            torch.tensor(
                torch.finfo(dtype).min,
                dtype=dtype,
                device=extended.device
            )
        )

        return extended

    _model_cls.get_extended_attention_mask = _get_extended_attention_mask


# ============================================================
# 2. 输入句子
# ============================================================

sentences = [
    "我喜欢人工智能",
    "I love artificial intelligence",
    "今天天气很好"
]


# ============================================================
# 3. 生成句子 embedding
# ============================================================

embeddings = model.encode(
    sentences
)

# 转为 FP32
embeddings = embeddings.astype(np.float32)

# 手动进行 L2 normalization
embeddings /= np.linalg.norm(
    embeddings,
    axis=1,
    keepdims=True
)

print("Embedding shape:")
print(embeddings.shape)

print("\nEmbedding dtype:")
print(embeddings.dtype)

print("\nEmbedding norms:")
print(np.linalg.norm(embeddings, axis=1))


# ============================================================
# 4. 计算余弦相似度矩阵
# ============================================================

# embedding 已经过 L2 normalization
#
# cosine_similarity(a, b) = a @ b
#
similarity = embeddings @ embeddings.T

print("\nSimilarity matrix:")
print(similarity)


# ============================================================
# 5. 绘制相似度热力图
# ============================================================

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ax = sns.heatmap(
    similarity,
    annot=True,
    fmt=".3f",
    cmap="Blues",
    vmin=0,
    vmax=1,
    xticklabels=sentences,
    yticklabels=sentences,
    square=True,
    linewidths=0.5,
    linecolor="white",
    cbar_kws={
        "shrink": 0.8,
        "label": "Cosine Similarity"
    },
    annot_kws={
        "size": 11
    }
)

# 深色格子使用白字，浅色格子使用黑字
for text in ax.texts:
    value = float(text.get_text())
    text.set_color(
        "white" if value > 0.6 else "black"
    )

plt.title("GTE-Multilingual-Base 句子余弦相似度矩阵")

plt.tight_layout()


# ============================================================
# 6. 保存
# ============================================================

out_dir = (
    Path(__file__).resolve().parents[1]
    / "outputs"
)

out_dir.mkdir(
    parents=True,
    exist_ok=True
)

output_path = (
    out_dir
    / "gte_multilingual_base_similarity_matrix.png"
)

plt.savefig(
    output_path,
    dpi=300
)

print(f"\nSaved to: {output_path}")

plt.show()