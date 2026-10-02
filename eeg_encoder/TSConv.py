import torch
from torch import nn
import torch.nn.functional as F


class EEGEncoder(nn.Module):
    """
    NICE TSConv EEG Encoder（不包含 SA / GA）

    输入:
        eeg: (B, C, T)

    输出:
        eeg_embedding: (B, embedding_dim)

    默认:
        C = 128
        k = 40
        embedding_dim = 1024

    整体结构:

        EEG
         ↓
        NICE TSConv Backbone
         ↓
        EEG feature
        (B, 40)
         ↓
        MLP Projection
        40 → 512 → 1024
         ↓
        L2 Normalize
         ↓
        (B, 1024)
    """

    def __init__(
        self,
        n_chans: int = 128,
        k: int = 40,
        m1: int = 25,
        m2: int = 51,
        s: int = 5,
        projection_hidden_dim: int = 512,
        embedding_dim: int = 1024,
        drop_prob: float = 0.5,
    ):
        super().__init__()

        self.n_chans = n_chans
        self.feature_dim = k
        self.embedding_dim = embedding_dim

        # ====================================================
        # 1. NICE TSConv Backbone
        # ====================================================

        self.tsconv = nn.Sequential(

            # ------------------------------------------------
            # Temporal Convolution
            #
            # 输入:
            # (B, 1, C, T)
            #
            # 输出:
            # (B, k, C, T')
            # ------------------------------------------------
            nn.Conv2d(
                in_channels=1,
                out_channels=k,
                kernel_size=(1, m1),
                stride=(1, 1),
                bias=False,
            ),

            nn.BatchNorm2d(k),

            # ------------------------------------------------
            # Temporal Average Pooling
            #
            # 主要作用:
            # 1. 时间降采样
            # 2. 时间平滑
            #
            # (B, k, C, T')
            # →
            # (B, k, C, T'')
            # ------------------------------------------------
            nn.AvgPool2d(
                kernel_size=(1, m2),
                stride=(1, s),
            ),

            # ------------------------------------------------
            # Spatial Convolution
            #
            # 卷积核高度 = n_chans
            # 一次融合所有 EEG 电极
            #
            # (B, k, C, T'')
            # →
            # (B, k, 1, T'')
            # ------------------------------------------------
            nn.Conv2d(
                in_channels=k,
                out_channels=k,
                kernel_size=(n_chans, 1),
                stride=(1, 1),
                bias=False,
            ),

            nn.BatchNorm2d(k),

            nn.ELU(),

            nn.Dropout(drop_prob),
        )

        # ====================================================
        # 2. MLP Projection Head
        #
        # NICE TSConv feature:
        #     (B, 40)
        #
        # →
        #
        # Shared embedding:
        #     (B, 1024)
        # ====================================================

        self.eeg_projection = nn.Sequential(
            nn.Linear(
                k,
                projection_hidden_dim,
            ),

            nn.GELU(),

            nn.Linear(
                projection_hidden_dim,
                embedding_dim,
            ),
        )


    def forward(self, eeg):
        """
        eeg:
            (B, C, T)

        return:
            (B, embedding_dim)
        """

        # ====================================================
        # 1. 增加 Conv2d 所需要的 channel 维度
        #
        # (B, C, T)
        # →
        # (B, 1, C, T)
        # ====================================================

        eeg = eeg.unsqueeze(1)

        # ====================================================
        # 2. NICE TSConv
        #
        # (B, 1, C, T)
        # →
        # (B, k, 1, T')
        # ====================================================

        features = self.tsconv(eeg)

        # ====================================================
        # 3. 去掉空间维
        #
        # (B, k, 1, T')
        # →
        # (B, k, T')
        # ====================================================

        features = features.squeeze(2)

        # ====================================================
        # 4. 时间维全局平均
        #
        # (B, k, T')
        # →
        # (B, k)
        #
        # 默认:
        # (B, 40)
        # ====================================================

        features = features.mean(dim=-1)

        # ====================================================
        # 5. MLP Projection
        #
        # (B, 40)
        # →
        # (B, 512)
        # →
        # (B, 1024)
        # ====================================================

        embeddings = self.eeg_projection(
            features
        )

        # ====================================================
        # 6. L2 Normalize
        #
        # 每个 embedding 满足:
        #
        # ||z||_2 ≈ 1
        #
        # 后面可直接使用:
        #
        # similarity = eeg @ text.T
        #
        # 计算 cosine similarity
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

    # --------------------------------------------------------
    # 模拟一个 batch 的 EEG
    #
    # B = 8
    # C = 128
    # T = 1500
    # --------------------------------------------------------

    eeg = torch.randn(
        8,
        128,
        1500,
    )

    print("Input EEG:")
    print(eeg.shape)

    # --------------------------------------------------------
    # 创建模型
    # --------------------------------------------------------

    model = EEGEncoder(
        n_chans=128,

        # NICE TSConv
        k=40,
        m1=25,
        m2=51,
        s=5,

        # Projection
        projection_hidden_dim=512,
        embedding_dim=1024,

        drop_prob=0.5,
    )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    eeg_embeddings = model(eeg)

    print("\nEEG embedding:")
    print(eeg_embeddings.shape)

    # --------------------------------------------------------
    # 检查 L2 norm
    # --------------------------------------------------------

    norms = torch.linalg.vector_norm(
        eeg_embeddings,
        dim=-1,
    )

    print("\nEmbedding norms:")
    print(norms)