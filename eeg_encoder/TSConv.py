import torch
from torch import nn
import torch.nn.functional as F


class EEGEncoder(nn.Module):
    """
    NICE TSConv EEG Encoder（定长 EEG 版本，不使用 padding mask）

    输入:
        eeg: (B, C, T)

    默认:
        C = 125
        k = 40

    输出:
        eeg_embedding: (B, embedding_dim)

    整体结构:

        EEG
        (B, 125, T)
            ↓
        NICE TSConv
            ↓
        (B, 40, 1, T')
            ↓
        1×1 Conv
            ↓
        Flatten
            ↓
        (B, 40 * T')
            ↓
        MLP Projection
        flatten_dim → 512 → 1024
            ↓
        L2 Normalize
            ↓
        (B, 1024)
    """

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        k: int = 40,
        m1: int = 25,
        m2: int = 51,
        s: int = 5,
        projection_hidden_dim: int = 512,
        embedding_dim: int = 1024,
        drop_prob: float = 0.5,
    ):
        super().__init__()

        # ====================================================
        # 1. 参数检查
        # ====================================================

        if n_chans <= 0:
            raise ValueError("n_chans must be positive")

        if n_times <= 0:
            raise ValueError("n_times must be positive")

        if k <= 0:
            raise ValueError("k must be positive")

        if m1 <= 0 or m2 <= 0 or s <= 0:
            raise ValueError("m1, m2 and s must be positive")

        self.n_chans = n_chans
        self.n_times = n_times
        self.k = k
        self.embedding_dim = embedding_dim

        # ====================================================
        # 2. 计算 TSConv 输出时间长度
        # ====================================================

        # Temporal Conv:
        #
        # T1 = T - m1 + 1
        #
        t_after_conv = n_times - m1 + 1

        if t_after_conv <= 0:
            raise ValueError(
                f"n_times={n_times} is too short for m1={m1}"
            )

        # AvgPool:
        #
        # T2 = floor((T1 - m2) / s) + 1
        #
        t_after_pool = (
            (t_after_conv - m2) // s
            + 1
        )

        if t_after_pool <= 0:
            raise ValueError(
                "EEG sequence is too short after temporal convolution "
                f"for AvgPool: n_times={n_times}, m1={m1}, "
                f"m2={m2}, s={s}"
            )

        self.temporal_feature_length = t_after_pool

        # TSConv 最终:
        #
        # (B, k, 1, T2)
        #
        # Flatten:
        #
        # (B, k * T2)
        #
        self.feature_dim = (
            k * t_after_pool
        )

        # ====================================================
        # 3. NICE TSConv Backbone
        # ====================================================

        self.tsconv = nn.Sequential(

            # ------------------------------------------------
            # Temporal Convolution
            #
            # (B, 1, C, T)
            #
            # →
            #
            # (B, k, C, T - m1 + 1)
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
            # NICE 原版:
            #
            # kernel_size = (1, 51)
            # stride      = (1, 5)
            # ------------------------------------------------

            nn.AvgPool2d(
                kernel_size=(1, m2),
                stride=(1, s),
            ),

            nn.BatchNorm2d(k),

            nn.ELU(),

            # ------------------------------------------------
            # Spatial Convolution
            #
            # 卷积核跨越所有 EEG 电极:
            #
            # (B, k, 125, T')
            #
            # →
            #
            # (B, k, 1, T')
            # ------------------------------------------------

            nn.Conv2d(
                in_channels=k,
                out_channels=k,
                kernel_size=(n_chans, 1),
                stride=(1, 1),
            ),

            nn.BatchNorm2d(k),

            nn.ELU(),

            nn.Dropout(drop_prob),
        )

        # ====================================================
        # 4. NICE 中 PatchEmbedding 的 1×1 Conv
        # ====================================================

        self.patch_projection = nn.Conv2d(
            in_channels=k,
            out_channels=k,
            kernel_size=(1, 1),
            stride=(1, 1),
        )

        # ====================================================
        # 5. EEG → Text Shared Space Projection
        #
        # flatten_dim → 512 → 1024
        #
        # 例如:
        #
        # T = 250:
        # flatten_dim = 1440
        #
        # 如果 T 更长:
        # flatten_dim 自动重新计算
        # ====================================================

        self.eeg_projection = nn.Sequential(
            nn.Linear(
                self.feature_dim,
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
        Parameters
        ----------
        eeg : torch.Tensor
            shape = (B, C, T)

        Returns
        -------
        torch.Tensor
            shape = (B, embedding_dim)
        """

        # ====================================================
        # 1. 输入检查
        # ====================================================

        if eeg.ndim != 3:
            raise ValueError(
                "EEG input must have shape (B, C, T), "
                f"but got {tuple(eeg.shape)}"
            )

        if eeg.shape[1] != self.n_chans:
            raise ValueError(
                f"Expected {self.n_chans} EEG channels, "
                f"but got {eeg.shape[1]}"
            )

        if eeg.shape[2] != self.n_times:
            raise ValueError(
                f"Expected fixed EEG length {self.n_times}, "
                f"but got {eeg.shape[2]}"
            )

        # ====================================================
        # 2. 增加 Conv2d 输入维
        #
        # (B, C, T)
        #
        # →
        #
        # (B, 1, C, T)
        # ====================================================

        eeg = eeg.unsqueeze(1)

        # ====================================================
        # 3. NICE TSConv
        #
        # (B, 1, C, T)
        #
        # →
        #
        # (B, k, 1, T')
        # ====================================================

        features = self.tsconv(eeg)

        # ====================================================
        # 4. 1×1 projection
        #
        # shape 不变:
        #
        # (B, k, 1, T')
        # ====================================================

        features = self.patch_projection(
            features
        )

        # ====================================================
        # 5. Flatten
        #
        # (B, k, 1, T')
        #
        # →
        #
        # (B, k * T')
        # ====================================================

        features = features.flatten(
            start_dim=1
        )

        # ====================================================
        # 6. MLP Projection
        #
        # (B, feature_dim)
        #
        # →
        #
        # (B, 512)
        #
        # →
        #
        # (B, embedding_dim)
        # ====================================================

        embeddings = self.eeg_projection(
            features
        )

        # ====================================================
        # 7. L2 Normalize
        #
        # ||z||_2 = 1
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
    # 假设你的预处理最后把所有 EEG 固定成 1500 个采样点
    #
    # 这里只是 demo。
    # 最终改成你实际确定的固定长度即可。
    # ========================================================

    batch_size = 8
    n_chans = 125
    n_times = 1500

    eeg = torch.randn(
        batch_size,
        n_chans,
        n_times,
    )

    model = EEGEncoder(
        n_chans=n_chans,
        n_times=n_times,

        # NICE
        k=40,
        m1=25,
        m2=51,
        s=5,
        drop_prob=0.5,

        # EEG → text embedding
        projection_hidden_dim=512,
        embedding_dim=1024,
    )

    print("Input:")
    print(eeg.shape)

    print("\nTSConv temporal length:")
    print(model.temporal_feature_length)

    print("\nFlatten feature dimension:")
    print(model.feature_dim)

    eeg_embeddings = model(eeg)

    print("\nOutput:")
    print(eeg_embeddings.shape)

    print("\nEmbedding norms:")
    print(
        torch.linalg.vector_norm(
            eeg_embeddings,
            dim=-1,
        )
    )