import torch
from torch import nn
import torch.nn.functional as F


class EEGEncoder_TSConv_without_1x1Conv(nn.Module):
    """
    NICE-inspired TSConv EEG Encoder
    定长 EEG 版本，不使用 padding mask。

    输入:
        eeg: (B, C, T)

    其中:
        B = batch size
        C = EEG 通道数，即 n_chans
        T = 固定 EEG 采样点数，即 n_times

    默认:
        k = 40
        m1 = 25
        m2 = 51
        s = 5
        embedding_dim = 1024

    输出:
        eeg_embedding: (B, embedding_dim)

    整体结构:

        EEG
        (B, C, T)
            ↓
        TSConv Backbone
            ↓
        (B, k, 1, T')
            ↓
        Flatten
            ↓
        (B, k × T')
            ↓
        MLP Projection Head
        (k × T') → 512 → embedding_dim
            ↓
        L2 Normalize
            ↓
        (B, embedding_dim)

    说明:
        这里采用固定长度 EEG，因此不需要 padding mask。

        TSConv 输出中的时间维不做全局平均，而是直接保留后 Flatten，
        从而保留时间位置上的特征信息。

        与 NICE 原始 PatchEmbedding 相比，本实现删除了其中的 1×1 Conv
        projection，直接使用后续 MLP 将 TSConv 特征映射到文本 embedding
        空间，因此属于 NICE-inspired TSConv，而不是严格复现完整的
        NICE PatchEmbedding。
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
        # 1. 参数初始化
        # ====================================================

        self.n_chans = n_chans
        self.n_times = n_times

        # TSConv 输出特征通道数
        self.k = k

        # 最终 EEG / Text 共享 embedding 维度
        self.embedding_dim = embedding_dim

        # ====================================================
        # 2. 计算 TSConv 输出时间长度
        # ====================================================

        # ----------------------------------------------------
        # Temporal Conv
        #
        # kernel_size = m1
        # stride = 1
        # padding = 0
        #
        # T1 = T - m1 + 1
        # ----------------------------------------------------

        t_after_conv = (
            n_times - m1 + 1
        )

        if t_after_conv <= 0:
            raise ValueError(
                f"n_times={n_times} is too short "
                f"for temporal kernel m1={m1}"
            )

        # ----------------------------------------------------
        # Temporal AvgPool
        #
        # kernel_size = m2
        # stride = s
        #
        # T2 = floor((T1 - m2) / s) + 1
        # ----------------------------------------------------

        t_after_pool = (
            (t_after_conv - m2) // s
            + 1
        )

        if t_after_pool <= 0:
            raise ValueError(
                "EEG sequence is too short after temporal "
                "convolution for AvgPool: "
                f"n_times={n_times}, "
                f"m1={m1}, "
                f"m2={m2}, "
                f"s={s}"
            )

        # TSConv 最终时间长度 T'
        self.temporal_feature_length = t_after_pool

        # ----------------------------------------------------
        # TSConv 输出:
        #
        # (B, k, 1, T')
        #
        # Flatten:
        #
        # (B, k × T')
        #
        # 因此:
        #
        # flatten_dim = k × T'
        # ----------------------------------------------------

        self.flatten_dim = k * self.temporal_feature_length

        # ====================================================
        # 3. NICE-inspired TSConv Backbone
        # ====================================================

        self.tsconv = nn.Sequential(

            # ------------------------------------------------
            # Temporal Convolution
            #
            # 输入:
            # (B, 1, C, T)
            #
            # 输出:
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
            # 输入:
            # (B, k, C, T1)
            #
            # 输出:
            # (B, k, C, T')
            #
            # NICE 默认:
            # m2 = 51
            # s  = 5
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
            # kernel height = n_chans
            #
            # 一次跨越所有 EEG 电极，
            # 将空间维 C 压缩为 1。
            #
            # 输入:
            # (B, k, C, T')
            #
            # 输出:
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

            nn.Dropout(
                p=drop_prob
            ),
        )

        # ====================================================
        # 4. EEG → Text Shared Space Projection
        # ====================================================
        #
        # TSConv:
        #
        # (B, k, 1, T')
        #
        # Flatten:
        #
        # (B, flatten_dim)
        #
        # MLP:
        #
        # flatten_dim
        #      ↓
        # projection_hidden_dim
        #      ↓
        # embedding_dim
        #
        #
        # 例如在 NICE 参数下:
        #
        # k  = 40
        # m1 = 25
        # m2 = 51
        # s  = 5
        # T  = 250
        #
        # T1 = 250 - 25 + 1
        #    = 226
        #
        # T' = floor((226 - 51) / 5) + 1
        #    = 36
        #
        # flatten_dim
        # = 40 × 36
        # = 1440
        #
        # 如果 T 改变，
        # flatten_dim 会自动重新计算。
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
            EEG 输入:

            (B, C, T)

        Returns
        -------
        torch.Tensor:
            L2-normalized EEG embedding:

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
                f"Expected {self.n_chans} EEG channels, "
                f"but got {eeg.shape[1]}"
            )

        if eeg.shape[2] != self.n_times:
            raise ValueError(
                f"Expected fixed EEG length "
                f"{self.n_times}, "
                f"but got {eeg.shape[2]}"
            )

        # ====================================================
        # 2. 增加 Conv2d 输入通道维
        # ====================================================
        #
        # (B, C, T)
        #
        # →
        #
        # (B, 1, C, T)
        # ====================================================

        eeg = eeg.unsqueeze(1)

        # ====================================================
        # 3. TSConv Backbone
        # ====================================================
        #
        # (B, 1, C, T)
        #
        # →
        #
        # (B, k, 1, T')
        # ====================================================

        features = self.tsconv(
            eeg
        )

        # ====================================================
        # 4. Flatten
        # ====================================================
        #
        # (B, k, 1, T')
        #
        # →
        #
        # (B, k × T')
        #
        # 即:
        #
        # (B, flatten_dim)
        # ====================================================

        features = features.flatten(
            start_dim=1
        )

        # ====================================================
        # 5. MLP Projection Head
        # ====================================================
        #
        # (B, flatten_dim)
        #
        # →
        #
        # (B, projection_hidden_dim)
        #
        # →
        #
        # (B, embedding_dim)
        # ====================================================

        embeddings = self.eeg_projection(
            features
        )

        # ====================================================
        # 6. L2 Normalize
        # ====================================================
        #
        # z_hat = z / ||z||_2
        #
        # 归一化后:
        #
        # ||z_hat||_2 ≈ 1
        #
        # 如果 Text embedding 同样经过 L2 normalization，
        # 则:
        #
        # eeg_embedding @ text_embedding.T
        #
        # 就等价于 cosine similarity。
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
    # 示例:
    #
    # batch_size = 8
    # EEG channels = 125
    # fixed EEG samples = 1500
    #
    # n_times=1500 这里只用于演示。
    # 正式训练时改成最终确定的统一 EEG 长度。
    # ========================================================

    batch_size = 8
    n_chans = 125
    n_times = 1500

    eeg = torch.randn(
        batch_size,
        n_chans,
        n_times,
    )

    # ========================================================
    # 创建 EEG Encoder
    # ========================================================

    model = EEGEncoder_TSConv_without_1x1Conv(

        # EEG 输入
        n_chans=n_chans,
        n_times=n_times,

        # NICE-inspired TSConv
        k=40,
        m1=25,
        m2=51,
        s=5,
        drop_prob=0.5,

        # MLP Projection Head
        projection_hidden_dim=512,

        # Qwen3 / E5 / BGE 等 1024 维文本空间
        embedding_dim=1024,
    )

    # ========================================================
    # 查看模型关键维度
    # ========================================================

    print("Input EEG shape:")
    print(eeg.shape)

    print("\nTSConv temporal feature length:")
    print(
        model.temporal_feature_length
    )

    print("\nFlatten dimension:")
    print(
        model.flatten_dim
    )

    # ========================================================
    # Forward
    # ========================================================

    eeg_embeddings = model(
        eeg
    )

    print("\nEEG embedding shape:")
    print(
        eeg_embeddings.shape
    )

    # ========================================================
    # 检查 L2 normalization
    # ========================================================

    norms = torch.linalg.vector_norm(
        eeg_embeddings,
        dim=-1,
    )

    print("\nEmbedding norms:")
    print(norms)