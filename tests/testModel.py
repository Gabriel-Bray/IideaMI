import torch
import TUI.IideaTUI as iidea


class LinearBlock(torch.nn.Module):
    """Every flavour of affine / bilinear map."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear1 = torch.nn.Linear(10, 10)
        self.linear2 = torch.nn.Linear(10, 20, bias=False)
        self.linear3 = torch.nn.Linear(20, 10)
        self.bilinear = torch.nn.Bilinear(10, 20, 10)
        self.lazy_linear = torch.nn.LazyLinear(10)
        self.identity = torch.nn.Identity()
        self.flatten = torch.nn.Flatten()
        self.unflatten = torch.nn.Unflatten(1, (2, 5))

    def forward(self):
        pass


class ConvBlock(torch.nn.Module):
    """1d / 2d / 3d convolutions, transposes, and the folding ops."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1d = torch.nn.Conv1d(3, 8, kernel_size=3, padding=1)
        self.conv2d = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv3d = torch.nn.Conv3d(3, 8, kernel_size=3, padding=1)
        self.conv2d_grouped = torch.nn.Conv2d(16, 16, kernel_size=3, groups=4, padding=1)
        self.conv2d_dilated = torch.nn.Conv2d(16, 16, kernel_size=3, dilation=2, padding=2)
        self.conv_transpose1d = torch.nn.ConvTranspose1d(8, 3, kernel_size=3, padding=1)
        self.conv_transpose2d = torch.nn.ConvTranspose2d(16, 3, kernel_size=3, padding=1)
        self.conv_transpose3d = torch.nn.ConvTranspose3d(8, 3, kernel_size=3, padding=1)
        self.lazy_conv2d = torch.nn.LazyConv2d(8, kernel_size=1)
        self.unfold = torch.nn.Unfold(kernel_size=3)
        self.fold = torch.nn.Fold(output_size=(8, 8), kernel_size=3)

    def forward(self):
        pass


class PoolingBlock(torch.nn.Module):
    """Max / average / adaptive / fractional / power pooling."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.maxpool1d = torch.nn.MaxPool1d(2)
        self.maxpool2d = torch.nn.MaxPool2d(2, stride=2)
        self.maxpool3d = torch.nn.MaxPool3d(2)
        self.maxunpool2d = torch.nn.MaxUnpool2d(2)
        self.avgpool1d = torch.nn.AvgPool1d(2)
        self.avgpool2d = torch.nn.AvgPool2d(2)
        self.avgpool3d = torch.nn.AvgPool3d(2)
        self.adaptive_maxpool2d = torch.nn.AdaptiveMaxPool2d((4, 4))
        self.adaptive_avgpool1d = torch.nn.AdaptiveAvgPool1d(4)
        self.adaptive_avgpool2d = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.adaptive_avgpool3d = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fractional_maxpool2d = torch.nn.FractionalMaxPool2d(2, output_ratio=0.5)
        self.lppool2d = torch.nn.LPPool2d(2, kernel_size=2)

    def forward(self):
        pass


class NormBlock(torch.nn.Module):
    """Batch / layer / group / instance / local-response / RMS normalization."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batchnorm1d = torch.nn.BatchNorm1d(10)
        self.batchnorm2d = torch.nn.BatchNorm2d(16)
        self.batchnorm3d = torch.nn.BatchNorm3d(8)
        self.lazy_batchnorm2d = torch.nn.LazyBatchNorm2d()
        self.syncbatchnorm = torch.nn.SyncBatchNorm(16)
        self.layernorm = torch.nn.LayerNorm(10)
        self.layernorm_multi = torch.nn.LayerNorm([16, 8, 8])
        self.groupnorm = torch.nn.GroupNorm(4, 16)
        self.instancenorm1d = torch.nn.InstanceNorm1d(10)
        self.instancenorm2d = torch.nn.InstanceNorm2d(16, affine=True)
        self.instancenorm3d = torch.nn.InstanceNorm3d(8)
        self.local_response_norm = torch.nn.LocalResponseNorm(2)
        self.rmsnorm = torch.nn.RMSNorm(10)

    def forward(self):
        pass


class ActivationBlock(torch.nn.Module):
    """Parameter-free and parameterized nonlinearities."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.relu = torch.nn.ReLU()
        self.relu6 = torch.nn.ReLU6()
        self.leaky_relu = torch.nn.LeakyReLU(0.01)
        self.prelu = torch.nn.PReLU(num_parameters=10)
        self.rrelu = torch.nn.RReLU()
        self.elu = torch.nn.ELU()
        self.celu = torch.nn.CELU()
        self.selu = torch.nn.SELU()
        self.gelu = torch.nn.GELU()
        self.silu = torch.nn.SiLU()
        self.mish = torch.nn.Mish()
        self.glu = torch.nn.GLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.hardsigmoid = torch.nn.Hardsigmoid()
        self.tanh = torch.nn.Tanh()
        self.hardtanh = torch.nn.Hardtanh()
        self.tanhshrink = torch.nn.Tanhshrink()
        self.softplus = torch.nn.Softplus()
        self.softsign = torch.nn.Softsign()
        self.softshrink = torch.nn.Softshrink()
        self.hardshrink = torch.nn.Hardshrink()
        self.hardswish = torch.nn.Hardswish()
        self.threshold = torch.nn.Threshold(0.1, 0.0)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.softmax2d = torch.nn.Softmax2d()
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.softmin = torch.nn.Softmin(dim=-1)
        self.logsigmoid = torch.nn.LogSigmoid()

    def forward(self):
        pass


class AttentionBlock(torch.nn.Module):
    """Multi-head attention and the full transformer stack built on it."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.self_attention = torch.nn.MultiheadAttention(embed_dim=32, num_heads=4)
        self.cross_attention = torch.nn.MultiheadAttention(
            embed_dim=32, num_heads=8, kdim=16, vdim=16, batch_first=True
        )
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=32, nhead=4)
        self.decoder_layer = torch.nn.TransformerDecoderLayer(d_model=32, nhead=4)
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=32, nhead=4), num_layers=3
        )
        self.decoder = torch.nn.TransformerDecoder(
            torch.nn.TransformerDecoderLayer(d_model=32, nhead=4), num_layers=3
        )
        self.transformer = torch.nn.Transformer(
            d_model=32,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=64,
        )

    def forward(self):
        pass


class RecurrentBlock(torch.nn.Module):
    """Full sequence modules and their single-step cell counterparts."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rnn = torch.nn.RNN(10, 20, num_layers=2, batch_first=True)
        self.lstm = torch.nn.LSTM(10, 20, num_layers=2, bidirectional=True)
        self.gru = torch.nn.GRU(10, 20, num_layers=2, dropout=0.1)
        self.rnn_cell = torch.nn.RNNCell(10, 20)
        self.lstm_cell = torch.nn.LSTMCell(10, 20)
        self.gru_cell = torch.nn.GRUCell(10, 20)

    def forward(self):
        pass


class EmbeddingBlock(torch.nn.Module):
    """Sparse lookup tables."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = torch.nn.Embedding(1000, 32)
        self.embedding_padded = torch.nn.Embedding(1000, 32, padding_idx=0)
        self.embedding_bag = torch.nn.EmbeddingBag(1000, 32, mode="mean")

    def forward(self):
        pass


class RegularizationBlock(torch.nn.Module):
    """Dropout family."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout = torch.nn.Dropout(0.5)
        self.dropout1d = torch.nn.Dropout1d(0.2)
        self.dropout2d = torch.nn.Dropout2d(0.2)
        self.dropout3d = torch.nn.Dropout3d(0.2)
        self.alpha_dropout = torch.nn.AlphaDropout(0.2)
        self.feature_alpha_dropout = torch.nn.FeatureAlphaDropout(0.2)

    def forward(self):
        pass


class PaddingBlock(torch.nn.Module):
    """Every padding mode across every dimensionality."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zero_pad2d = torch.nn.ZeroPad2d(1)
        self.constant_pad1d = torch.nn.ConstantPad1d(1, 0.0)
        self.constant_pad2d = torch.nn.ConstantPad2d(1, 0.0)
        self.constant_pad3d = torch.nn.ConstantPad3d(1, 0.0)
        self.reflection_pad1d = torch.nn.ReflectionPad1d(1)
        self.reflection_pad2d = torch.nn.ReflectionPad2d(1)
        self.reflection_pad3d = torch.nn.ReflectionPad3d(1)
        self.replication_pad1d = torch.nn.ReplicationPad1d(1)
        self.replication_pad2d = torch.nn.ReplicationPad2d(1)
        self.replication_pad3d = torch.nn.ReplicationPad3d(1)
        self.circular_pad2d = torch.nn.CircularPad2d(1)

    def forward(self):
        pass


class VisionBlock(torch.nn.Module):
    """Resampling / shuffling ops."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pixel_shuffle = torch.nn.PixelShuffle(2)
        self.pixel_unshuffle = torch.nn.PixelUnshuffle(2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="nearest")
        self.upsample_bilinear = torch.nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsample_nearest = torch.nn.UpsamplingNearest2d(scale_factor=2)
        self.channel_shuffle = torch.nn.ChannelShuffle(2)

    def forward(self):
        pass


class DistanceBlock(torch.nn.Module):
    """Similarity / distance modules."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1)
        self.pairwise_distance = torch.nn.PairwiseDistance(p=2)

    def forward(self):
        pass


class LossBlock(torch.nn.Module):
    """Losses are nn.Modules too, and the analyzer should see them as children."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l1 = torch.nn.L1Loss()
        self.mse = torch.nn.MSELoss()
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.nll = torch.nn.NLLLoss()
        self.bce = torch.nn.BCELoss()
        self.bce_with_logits = torch.nn.BCEWithLogitsLoss()
        self.kl_div = torch.nn.KLDivLoss()
        self.smooth_l1 = torch.nn.SmoothL1Loss()
        self.huber = torch.nn.HuberLoss()
        self.hinge_embedding = torch.nn.HingeEmbeddingLoss()
        self.margin_ranking = torch.nn.MarginRankingLoss()
        self.triplet_margin = torch.nn.TripletMarginLoss()
        self.cosine_embedding = torch.nn.CosineEmbeddingLoss()
        self.ctc = torch.nn.CTCLoss()
        self.poisson_nll = torch.nn.PoissonNLLLoss()
        self.gaussian_nll = torch.nn.GaussianNLLLoss()
        self.soft_margin = torch.nn.SoftMarginLoss()
        self.multi_margin = torch.nn.MultiMarginLoss()
        self.multilabel_margin = torch.nn.MultiLabelMarginLoss()
        self.multilabel_soft_margin = torch.nn.MultiLabelSoftMarginLoss()

    def forward(self):
        pass


class ContainerBlock(torch.nn.Module):
    """Sequential / ModuleList / ModuleDict / ParameterList / ParameterDict.

    These are the interesting cases for a tree walker: their children are named
    by index or by key rather than by attribute.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(20, 10),
        )
        self.nested_sequential = torch.nn.Sequential(
            torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, 3, padding=1),
                torch.nn.BatchNorm2d(16),
                torch.nn.ReLU(),
            ),
            torch.nn.Sequential(
                torch.nn.Conv2d(16, 32, 3, padding=1),
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(),
            ),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.module_list = torch.nn.ModuleList(
            [torch.nn.Linear(10, 10) for _ in range(4)]
        )
        self.module_dict = torch.nn.ModuleDict(
            {
                "conv": torch.nn.Conv2d(3, 8, 3),
                "pool": torch.nn.MaxPool2d(2),
                "norm": torch.nn.LayerNorm(8),
                "act": torch.nn.GELU(),
            }
        )
        self.parameter_list = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(10, 10)) for _ in range(3)]
        )
        self.parameter_dict = torch.nn.ParameterDict(
            {
                "weight": torch.nn.Parameter(torch.randn(10, 10)),
                "bias": torch.nn.Parameter(torch.zeros(10)),
            }
        )

    def forward(self):
        pass


class DeepBranch(torch.nn.Module):
    """A deliberately deep chain, to test how far the traversal can descend."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.level1 = torch.nn.Sequential(
            torch.nn.Sequential(
                torch.nn.Sequential(
                    torch.nn.Sequential(
                        torch.nn.Sequential(torch.nn.Linear(10, 10)),
                        torch.nn.ReLU(),
                    ),
                    torch.nn.LayerNorm(10),
                ),
                torch.nn.Dropout(0.1),
            ),
            torch.nn.Linear(10, 10),
        )

    def forward(self):
        pass


class LeafOnly(torch.nn.Module):
    """No children at all -- the traversal should handle an empty listing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = torch.nn.Parameter(torch.randn(10, 10))
        self.register_buffer("running_stat", torch.zeros(10))

    def forward(self):
        pass


class model(torch.nn.Module):
    """The top-level fixture: one child per layer family, plus loose leaves.

    Constructed with no arguments, exactly as IideaTUI expects:

        m = model()
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # flat leaves, so the root listing is not purely containers
        self.linear1 = torch.nn.Linear(10, 10)
        self.linear2 = torch.nn.Linear(10, 10)
        self.linear3 = torch.nn.Linear(10, 10)
        self.relu = torch.nn.ReLU()

        # one sub-block per family
        self.linear_block = LinearBlock()
        self.conv_block = ConvBlock()
        self.pooling_block = PoolingBlock()
        self.norm_block = NormBlock()
        self.activation_block = ActivationBlock()
        self.attention_block = AttentionBlock()
        self.recurrent_block = RecurrentBlock()
        self.embedding_block = EmbeddingBlock()
        self.regularization_block = RegularizationBlock()
        self.padding_block = PaddingBlock()
        self.vision_block = VisionBlock()
        self.distance_block = DistanceBlock()
        self.loss_block = LossBlock()
        self.container_block = ContainerBlock()
        self.deep_branch = DeepBranch()
        self.leaf_only = LeafOnly()

        # bare parameters and buffers hanging off the root
        self.scale = torch.nn.Parameter(torch.ones(1))
        self.register_buffer("step", torch.zeros(1, dtype=torch.long))

    def forward(self):
        pass


if __name__ == "__main__":

    m = model()
    i = iidea.IideaAnalyzer(m)
    i.main()
