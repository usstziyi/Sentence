import math

import torch
from torch import nn
import torch.nn.functional as F


# ============================================================
# Sinusoidal Positional Encoding
# ============================================================


def build_sinusoidal_position_encoding(
    length: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    创建正弦位置编码。

    Parameters
    ----------
    length:
        时间序列长度 T'。

    dim:
        embedding dimension。

    Returns
    -------
    torch.Tensor
        shape:
        (1, length, dim)
    """

    position = torch.arange(
        length,
        device=device,
        dtype=torch.float32,
    ).unsqueeze(1)

    div_term = torch.exp(
        torch.arange(
            0,
            dim,
            2,
            device=device,
            dtype=torch.float32,
        )
        * (-math.log(10000.0) / dim)
    )

    pe = torch.zeros(
        length,
        dim,
        device=device,
        dtype=torch.float32,
    )

    # 偶数位置
    pe[:, 0::2] = torch.sin(
        position * div_term
    )

    # 奇数位置
    if dim > 1:
        pe[:, 1::2] = torch.cos(
            position * div_term[: pe[:, 1::2].shape[1]]
        )

    pe = pe.unsqueeze(0)

    return pe.to(dtype=dtype)


# ============================================================
# Cross-Attention Block
# ============================================================


class CrossAttentionBlock(nn.Module):
    """
    一个 Cross-Attention Resampler Block。

    Learnable queries:
        (B, N, D)

    EEG context:
        (B, T', D)

    Cross Attention:
        Q = learnable queries
        K = EEG features
        V = EEG features

    输出:
        (B, N, D)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        if dim <= 0:
            raise ValueError(
                "dim must be positive"
            )

        if num_heads <= 0:
            raise ValueError(
                "num_heads must be positive"
            )

        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible "
                f"by num_heads ({num_heads})"
            )

        if ffn_dim <= 0:
            raise ValueError(
                "ffn_dim must be positive"
            )

        if not 0 <= dropout < 1:
            raise ValueError(
                "dropout must satisfy 0 <= dropout < 1"
            )

        # ----------------------------------------------------
        # Pre-Norm for Query
        # ----------------------------------------------------

        self.query_norm = nn.LayerNorm(
            dim
        )

        # ----------------------------------------------------
        # Pre-Norm for EEG Context
        # ----------------------------------------------------

        self.context_norm = nn.LayerNorm(
            dim
        )

        # ----------------------------------------------------
        # Cross Attention
        #
        # Q:
        # (B, N, D)
        #
        # K / V:
        # (B, T', D)
        #
        # Output:
        # (B, N, D)
        # ----------------------------------------------------

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.attention_dropout = nn.Dropout(
            dropout
        )

        # ----------------------------------------------------
        # Feed Forward Network
        # ----------------------------------------------------

        self.ffn_norm = nn.LayerNorm(
            dim
        )

        self.ffn = nn.Sequential(
            nn.Linear(
                dim,
                ffn_dim,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                ffn_dim,
                dim,
            ),
            nn.Dropout(
                dropout
            ),
        )

    def forward(
        self,
        queries: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        queries:
            learnable latent queries

            shape:
            (B, N, D)

        context:
            EEG temporal features

            shape:
            (B, T', D)

        Returns
        -------
        torch.Tensor

            shape:
            (B, N, D)
        """

        # ====================================================
        # 1. Cross Attention
        # ====================================================

        q = self.query_norm(
            queries
        )

        kv = self.context_norm(
            context
        )

        attention_output, _ = self.cross_attention(
            query=q,
            key=kv,
            value=kv,
            need_weights=False,
        )

        # Residual connection
        queries = (
            queries
            + self.attention_dropout(
                attention_output
            )
        )

        # ====================================================
        # 2. Feed Forward Network
        # ====================================================

        ffn_output = self.ffn(
            self.ffn_norm(
                queries
            )
        )

        # Residual connection
        queries = (
            queries
            + ffn_output
        )

        return queries


# ============================================================
# Attention Resampler
# ============================================================


class AttentionResampler(nn.Module):
    """
    将长度可变的 EEG 时间特征:

        (B, T', input_dim)

    重采样成固定数量的 latent tokens:

        (B, num_queries, attention_dim)

    核心思想:

        num_queries 个可学习 Query

                    ↓

        Cross Attention

        Q = Learnable Queries
        K = EEG sequence
        V = EEG sequence

                    ↓

        固定数量 latent tokens

    因此:

        T' 可以变化

    但是:

        num_queries 始终固定。

    不需要 padding，
    不需要 padding mask。
    """

    def __init__(
        self,
        input_dim: int = 40,
        attention_dim: int = 128,
        num_queries: int = 20,
        num_heads: int = 8,
        num_layers: int = 2,
        ffn_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        # ====================================================
        # 参数检查
        # ====================================================

        if input_dim <= 0:
            raise ValueError(
                "input_dim must be positive"
            )

        if attention_dim <= 0:
            raise ValueError(
                "attention_dim must be positive"
            )

        if num_queries <= 0:
            raise ValueError(
                "num_queries must be positive"
            )

        if num_heads <= 0:
            raise ValueError(
                "num_heads must be positive"
            )

        if attention_dim % num_heads != 0:
            raise ValueError(
                "attention_dim must be divisible "
                "by num_heads"
            )

        if num_layers <= 0:
            raise ValueError(
                "num_layers must be positive"
            )

        # ====================================================
        # 保存参数
        # ====================================================

        self.input_dim = input_dim
        self.attention_dim = attention_dim
        self.num_queries = num_queries

        # ====================================================
        # EEG Feature Projection
        #
        # TSConv:
        #
        # (B, T', 40)
        #
        # →
        #
        # (B, T', attention_dim)
        #
        # 默认:
        #
        # 40 → 128
        # ====================================================

        self.input_projection = nn.Linear(
            input_dim,
            attention_dim,
        )

        self.input_norm = nn.LayerNorm(
            attention_dim
        )

        # ====================================================
        # Learnable Queries
        #
        # shape:
        #
        # (1, N, D)
        #
        # 默认:
        #
        # (1, 20, 128)
        #
        # forward 时扩展为:
        #
        # (B, 20, 128)
        # ====================================================

        self.query_tokens = nn.Parameter(
            torch.empty(
                1,
                num_queries,
                attention_dim,
            )
        )

        nn.init.normal_(
            self.query_tokens,
            mean=0.0,
            std=0.02,
        )

        # ====================================================
        # Cross Attention Layers
        # ====================================================

        self.layers = nn.ModuleList(
            [
                CrossAttentionBlock(
                    dim=attention_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # ====================================================
        # Final Norm
        # ====================================================

        self.output_norm = nn.LayerNorm(
            attention_dim
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        features:

            TSConv 时间特征

            shape:

            (B, T', input_dim)

        Returns
        -------
        torch.Tensor

            shape:

            (B, num_queries, attention_dim)
        """

        if features.ndim != 3:
            raise ValueError(
                "AttentionResampler input must have "
                "shape (B, T, D), "
                f"but got {tuple(features.shape)}"
            )

        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected feature dimension "
                f"{self.input_dim}, "
                f"but got {features.shape[-1]}"
            )

        batch_size = features.shape[0]
        sequence_length = features.shape[1]

        # ====================================================
        # 1. Project EEG Features
        #
        # (B, T', 40)
        #
        # →
        #
        # (B, T', 128)
        # ====================================================

        context = self.input_projection(
            features
        )

        # ====================================================
        # 2. Add Temporal Positional Encoding
        #
        # Attention 自己并不知道:
        #
        # t1、t2、t3...
        #
        # 的时间先后关系。
        #
        # 所以给 TSConv 时间特征加入 positional encoding。
        # ====================================================

        positional_encoding = (
            build_sinusoidal_position_encoding(
                length=sequence_length,
                dim=self.attention_dim,
                device=context.device,
                dtype=context.dtype,
            )
        )

        context = (
            context
            + positional_encoding
        )

        context = self.input_norm(
            context
        )

        # ====================================================
        # 3. Expand Learnable Queries
        #
        # (1, N, D)
        #
        # →
        #
        # (B, N, D)
        # ====================================================

        queries = self.query_tokens.expand(
            batch_size,
            -1,
            -1,
        )

        # ====================================================
        # 4. Cross Attention
        #
        # 每一个 Query 都可以观察整个 EEG 时间轴。
        #
        # EEG 长度 T' 可以变化。
        #
        # Query 数量 N 始终固定。
        # ====================================================

        for layer in self.layers:
            queries = layer(
                queries=queries,
                context=context,
            )

        # ====================================================
        # 5. Final Normalization
        # ====================================================

        queries = self.output_norm(
            queries
        )

        return queries


# ============================================================
# TSConv + Attention Resampler EEG Encoder
# ============================================================


class TSConvAttentionEEGEncoder(nn.Module):
    """
    NICE-inspired TSConv EEG Encoder
    + Attention Resampler。

    支持不同 batch 使用不同 EEG 时间长度。

    训练方式
    --------

    - 按 EEG 长度建立 Length Buckets

    - 同一个 batch 内 EEG 长度相同

    - 不同长度 batch 随机交替训练

    - 不使用 padding

    - 不使用 padding mask


    输入
    ----

        eeg:

        (B, C, T)


    默认:

        C = 125


    T 可以在不同 batch 之间变化。


    网络结构
    --------

        EEG

        (B, C, T)

            ↓

        TSConv

        (B, k, 1, T')

            ↓

        squeeze + transpose

        (B, T', k)

            ↓

        Linear Projection

        (B, T', attention_dim)

            ↓

        Temporal Positional Encoding

            ↓

        Attention Resampler

        num_queries learnable queries

            ↓

        (B, num_queries, attention_dim)

            ↓

        Flatten

            ↓

        (B, num_queries × attention_dim)

            ↓

        MLP Projector

            ↓

        embedding_dim

            ↓

        L2 Normalize

            ↓

        (B, embedding_dim)


    默认参数
    --------

        k = 40

        attention_dim = 128

        num_queries = 20

    所以:

        20 × 128 = 2560

    与原 AdaptiveAvgPool:

        40 × 64 = 2560

    的 Flatten 维度一致。
    """

    def __init__(
        self,
        n_chans: int = 125,

        # ----------------------------------------------------
        # TSConv
        # ----------------------------------------------------
        k: int = 40,
        m1: int = 25,
        m2: int = 51,
        s: int = 5,
        drop_prob: float = 0.5,

        # ----------------------------------------------------
        # Attention Resampler
        # ----------------------------------------------------
        attention_dim: int = 128,
        num_queries: int = 20,
        num_heads: int = 8,
        num_attention_layers: int = 2,
        attention_ffn_dim: int = 256,
        attention_dropout: float = 0.1,

        # ----------------------------------------------------
        # Projection
        # ----------------------------------------------------
        projection_hidden_dim: int = 512,
        embedding_dim: int = 1024,
    ):
        super().__init__()

        # ====================================================
        # 1. 参数检查
        # ====================================================

        if n_chans <= 0:
            raise ValueError(
                "n_chans must be positive"
            )

        if k <= 0:
            raise ValueError(
                "k must be positive"
            )

        if m1 <= 0:
            raise ValueError(
                "m1 must be positive"
            )

        if m2 <= 0:
            raise ValueError(
                "m2 must be positive"
            )

        if s <= 0:
            raise ValueError(
                "s must be positive"
            )

        if not 0 <= drop_prob < 1:
            raise ValueError(
                "drop_prob must satisfy "
                "0 <= drop_prob < 1"
            )

        if attention_dim <= 0:
            raise ValueError(
                "attention_dim must be positive"
            )

        if num_queries <= 0:
            raise ValueError(
                "num_queries must be positive"
            )

        if num_heads <= 0:
            raise ValueError(
                "num_heads must be positive"
            )

        if attention_dim % num_heads != 0:
            raise ValueError(
                f"attention_dim ({attention_dim}) "
                f"must be divisible by "
                f"num_heads ({num_heads})"
            )

        if num_attention_layers <= 0:
            raise ValueError(
                "num_attention_layers "
                "must be positive"
            )

        if attention_ffn_dim <= 0:
            raise ValueError(
                "attention_ffn_dim "
                "must be positive"
            )

        if not 0 <= attention_dropout < 1:
            raise ValueError(
                "attention_dropout must satisfy "
                "0 <= attention_dropout < 1"
            )

        if projection_hidden_dim <= 0:
            raise ValueError(
                "projection_hidden_dim "
                "must be positive"
            )

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be positive"
            )

        # ====================================================
        # 2. 保存参数
        # ====================================================

        self.n_chans = n_chans

        self.k = k

        self.m1 = m1
        self.m2 = m2
        self.s = s

        self.attention_dim = attention_dim

        self.num_queries = num_queries

        self.embedding_dim = embedding_dim

        # ----------------------------------------------------
        # Attention Resampler 输出:
        #
        # (B, num_queries, attention_dim)
        #
        # Flatten:
        #
        # (B, num_queries × attention_dim)
        #
        # 默认:
        #
        # 20 × 128 = 2560
        # ----------------------------------------------------

        self.flatten_dim = (
            num_queries
            * attention_dim
        )

        # ====================================================
        # 3. TSConv Backbone
        # ====================================================

        self.tsconv = nn.Sequential(

            # ------------------------------------------------
            # Temporal Convolution
            #
            # Input:
            #
            # (B, 1, C, T)
            #
            # Output:
            #
            # (B, k, C, T1)
            #
            # T1 = T - m1 + 1
            # ------------------------------------------------

            nn.Conv2d(
                in_channels=1,
                out_channels=k,
                kernel_size=(1, m1),
                stride=(1, 1),
            ),

            # ------------------------------------------------
            # Temporal Average Pooling
            #
            # Input:
            #
            # (B, k, C, T1)
            #
            # Output:
            #
            # (B, k, C, T2)
            #
            # T2 =
            #
            # floor((T1 - m2) / s) + 1
            # ------------------------------------------------

            nn.AvgPool2d(
                kernel_size=(1, m2),
                stride=(1, s),
            ),

            nn.BatchNorm2d(
                k
            ),

            nn.ELU(),

            # ------------------------------------------------
            # Spatial Convolution
            #
            # 卷积核一次跨越所有 EEG 电极。
            #
            # (B, k, C, T2)
            #
            # →
            #
            # (B, k, 1, T2)
            # ------------------------------------------------

            nn.Conv2d(
                in_channels=k,
                out_channels=k,
                kernel_size=(n_chans, 1),
                stride=(1, 1),
            ),

            nn.BatchNorm2d(
                k
            ),

            nn.ELU(),

            nn.Dropout(
                p=drop_prob
            ),
        )

        # ====================================================
        # 4. Attention Resampler
        # ====================================================

        self.attention_resampler = AttentionResampler(
            input_dim=k,
            attention_dim=attention_dim,
            num_queries=num_queries,
            num_heads=num_heads,
            num_layers=num_attention_layers,
            ffn_dim=attention_ffn_dim,
            dropout=attention_dropout,
        )

        # ====================================================
        # 5. EEG → Text Shared Space Projector
        #
        # 默认:
        #
        # Attention Resampler:
        #
        # (B, 20, 128)
        #
        # Flatten:
        #
        # (B, 2560)
        #
        # MLP:
        #
        # 2560 → 512 → 1024
        # ====================================================

        self.eeg_projection = nn.Sequential(

            nn.Linear(
                self.flatten_dim,
                projection_hidden_dim,
            ),

            nn.GELU(),

            nn.Linear(
                projection_hidden_dim,
                embedding_dim,
            ),
        )

    def forward(
        self,
        eeg: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        eeg:

            EEG input

            shape:

            (B, C, T)

        不同 batch 的 T 可以不同。

        Returns
        -------
        torch.Tensor

            L2 normalized EEG embedding

            shape:

            (B, embedding_dim)
        """

        # ====================================================
        # 1. 输入检查
        # ====================================================

        if eeg.ndim != 3:
            raise ValueError(
                "EEG input must have shape "
                "(B, C, T), "
                f"but got {tuple(eeg.shape)}"
            )

        if eeg.shape[1] != self.n_chans:
            raise ValueError(
                f"Expected "
                f"{self.n_chans} EEG channels, "
                f"but got {eeg.shape[1]}"
            )

        # ----------------------------------------------------
        # Temporal Conv 后至少需要 m2 个时间点
        #
        # T - m1 + 1 >= m2
        #
        # 因此:
        #
        # T >= m1 + m2 - 1
        #
        # 默认:
        #
        # 25 + 51 - 1 = 75
        # ----------------------------------------------------

        min_samples = (
            self.m1
            + self.m2
            - 1
        )

        if eeg.shape[-1] < min_samples:
            raise ValueError(
                f"EEG sequence is too short: "
                f"got {eeg.shape[-1]} samples, "
                f"but at least "
                f"{min_samples} are required."
            )

        # ====================================================
        # 2. Add Conv2d Channel Dimension
        #
        # (B, C, T)
        #
        # →
        #
        # (B, 1, C, T)
        # ====================================================

        eeg = eeg.unsqueeze(
            1
        )

        # ====================================================
        # 3. TSConv
        #
        # (B, 1, C, T)
        #
        # →
        #
        # (B, k, 1, T')
        #
        # T' 会随 EEG 输入长度变化。
        # ====================================================

        features = self.tsconv(
            eeg
        )

        # ====================================================
        # 4. 转换为时间序列格式
        #
        # 当前:
        #
        # (B, k, 1, T')
        #
        # squeeze spatial dimension:
        #
        # (B, k, T')
        #
        # transpose:
        #
        # (B, T', k)
        #
        # MultiheadAttention 需要这种格式。
        # ====================================================

        features = features.squeeze(
            2
        )

        features = features.transpose(
            1,
            2,
        )

        # ====================================================
        # 5. Attention Resampler
        #
        # 输入:
        #
        # (B, T', k)
        #
        # 默认:
        #
        # (B, T', 40)
        #
        # →
        #
        # (B, 20, 128)
        #
        # 无论 T' 是多少，
        # Query 数始终为 20。
        # ====================================================

        features = self.attention_resampler(
            features
        )

        # ====================================================
        # 6. Flatten
        #
        # (B, 20, 128)
        #
        # →
        #
        # (B, 2560)
        # ====================================================

        features = features.flatten(
            start_dim=1
        )

        # ====================================================
        # 7. Projection Head
        #
        # (B, 2560)
        #
        # →
        #
        # (B, 512)
        #
        # →
        #
        # (B, 1024)
        # ====================================================

        embeddings = self.eeg_projection(
            features
        )

        # ====================================================
        # 8. L2 Normalize
        # ====================================================

        embeddings = F.normalize(
            embeddings,
            p=2,
            dim=-1,
        )

        return embeddings


# ============================================================
# Demo
# ============================================================


if __name__ == "__main__":

    # ========================================================
    # 创建模型
    # ========================================================

    model = TSConvAttentionEEGEncoder(

        # EEG
        n_chans=125,

        # NICE TSConv
        k=40,
        m1=25,
        m2=51,
        s=5,
        drop_prob=0.5,

        # Attention Resampler
        attention_dim=128,
        num_queries=20,
        num_heads=8,
        num_attention_layers=2,
        attention_ffn_dim=256,
        attention_dropout=0.1,

        # Projection
        projection_hidden_dim=512,
        embedding_dim=1024,
    )

    # Demo 时关闭 Dropout
    model.eval()

    # ========================================================
    # 模拟不同长度 bucket
    # ========================================================

    batch_size = 8

    eeg_short = torch.randn(
        batch_size,
        125,
        2000,
    )

    eeg_medium = torch.randn(
        batch_size,
        125,
        2800,
    )

    eeg_long = torch.randn(
        batch_size,
        125,
        3600,
    )

    # ========================================================
    # 同一个模型处理不同 EEG 长度
    # ========================================================

    with torch.no_grad():

        embedding_short = model(
            eeg_short
        )

        embedding_medium = model(
            eeg_medium
        )

        embedding_long = model(
            eeg_long
        )

    # ========================================================
    # 输出 Shape
    # ========================================================

    print(
        "========================================"
    )

    print("Short EEG:")
    print(
        "Input: ",
        eeg_short.shape,
    )
    print(
        "Output:",
        embedding_short.shape,
    )

    print(
        "========================================"
    )

    print("Medium EEG:")
    print(
        "Input: ",
        eeg_medium.shape,
    )
    print(
        "Output:",
        embedding_medium.shape,
    )

    print(
        "========================================"
    )

    print("Long EEG:")
    print(
        "Input: ",
        eeg_long.shape,
    )
    print(
        "Output:",
        embedding_long.shape,
    )

    print(
        "========================================"
    )

    # ========================================================
    # 检查 L2 Norm
    # ========================================================

    print(
        "\nShort embedding norms:"
    )

    print(
        torch.linalg.vector_norm(
            embedding_short,
            dim=-1,
        )
    )

    print(
        "\nMedium embedding norms:"
    )

    print(
        torch.linalg.vector_norm(
            embedding_medium,
            dim=-1,
        )
    )

    print(
        "\nLong embedding norms:"
    )

    print(
        torch.linalg.vector_norm(
            embedding_long,
            dim=-1,
        )
    )

    # ========================================================
    # 参数量
    # ========================================================

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "\n========================================"
    )

    print(
        f"Total parameters: "
        f"{total_params:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_params:,}"
    )

    print(
        "========================================"
    )