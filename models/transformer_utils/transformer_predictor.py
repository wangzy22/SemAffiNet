import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME

from models.transformer_utils.transformer import TransformerEncoder, TransformerDecoder
from timm.models.layers.weight_init import trunc_normal_


class TransformerPredictor(ME.MinkowskiNetwork):
    def __init__(self, config, in_channels_3d, in_channels_2d, num_classes, D=3):
        super(TransformerPredictor, self).__init__(D)

        self.num_classes = num_classes
        self.num_view = config.viewNum
        self.npb = config.num_point_batch
        self.hidden_dim = config.hidden_dim

        self.pos_embed_2d = nn.Parameter(torch.zeros(1, config.num_tokens_2d, self.hidden_dim))
        self.pe_layer_3d = MLP(3, in_channels_3d, self.hidden_dim, num_layers=3)
        trunc_normal_(self.pos_embed_2d, std=.02)

        self.enc_share = TransformerEncoder(
            embed_dim=config.hidden_dim,
            depth=config.enc_layers,
            num_heads=config.nheads,
            drop_rate=config.drop_rate,
            attn_drop_rate=config.attn_drop_rate,
            drop_path=config.drop_path
        )

        self.dec_3d = TransformerDecoder(
            embed_dim=config.hidden_dim,
            depth=config.dec_layers,
            num_heads=config.nheads,
            drop_rate=config.drop_rate,
            attn_drop_rate=config.attn_drop_rate,
            drop_path=config.drop_path
        )

        self.dec_2d = TransformerDecoder(
            embed_dim=config.hidden_dim,
            depth=config.dec_layers,
            num_heads=config.nheads,
            drop_rate=config.drop_rate,
            attn_drop_rate=config.attn_drop_rate,
            drop_path=config.drop_path
        )

        self.query_embed_2d = nn.Embedding(num_classes, self.hidden_dim)
        self.query_embed_3d = nn.Embedding(num_classes, self.hidden_dim)

        self.input_proj_3d = nn.Conv1d(in_channels_3d, self.hidden_dim, kernel_size=1)
        c2_xavier_fill(self.input_proj_3d)

        self.input_proj_2d = nn.Conv1d(in_channels_2d, self.hidden_dim, kernel_size=1)
        c2_xavier_fill(self.input_proj_2d)

        self.mask_embed_2d = MLP(self.hidden_dim, self.hidden_dim, config.mask_dim, 3)
        self.mask_embed_3d = MLP(self.hidden_dim, self.hidden_dim, config.mask_dim, 3)

    def forward(self, x_2d, x_3d):
        ## 3d pre-processing ##
        srcs_3d = []
        pos_batch = []
        for i in range(len(x_3d.C[:, 0].unique())):
            mask_i = (x_3d.C[:, 0] == i)
            src = x_3d.F[mask_i]
            pos = self.pe_layer_3d(x_3d.C[mask_i][:, 1:].type(torch.float))
            if len(src) > self.npb:
                r = torch.randint(0, len(src), (self.npb,))
                src = src[r]
                pos = pos[r]
            elif len(src) < self.npb:
                r = torch.randint(0, len(src), (self.npb - len(src),))
                src_repeat = src[r]
                pos_repeat = pos[r]
                src = torch.cat([src, src_repeat], dim=0)
                pos = torch.cat([pos, pos_repeat], dim=0)
            srcs_3d.append(src)
            pos_batch.append(pos)
        srcs_3d = torch.stack(srcs_3d, dim=0).transpose(1, 2).contiguous()
        trans_input_3d = self.input_proj_3d(srcs_3d).transpose(1, 2).contiguous()
        pe_3d = torch.stack(pos_batch, dim=0)

        ## 2d pre-processing ##
        VB, C, H, W = x_2d.size()
        B = int(VB // self.num_view)
        N_2d = self.num_view*H*W
        assert(trans_input_3d.shape[0] == B)
        srcs_2d = x_2d.view(VB, C, H*W)
        trans_input_2d = self.input_proj_2d(srcs_2d).transpose(1, 2).contiguous()
        trans_input_2d = trans_input_2d.view(self.num_view, B, H*W, self.hidden_dim).transpose(0, 1).contiguous().view(B, N_2d, self.hidden_dim)

        trans_input = torch.cat([trans_input_2d, trans_input_3d], dim=1)
        pos_input = torch.cat([self.pos_embed_2d.expand(VB, -1, -1).view(self.num_view, B, H*W, self.hidden_dim).transpose(0, 1).contiguous().view(B, N_2d, self.hidden_dim),
                               pe_3d], dim=1)
        trans_feat = self.enc_share(trans_input, pos_input)
        trans_feat_2d = trans_feat[:, :N_2d, :].view(B, self.num_view, H*W, self.hidden_dim).transpose(0, 1).contiguous().view(VB, H*W, self.hidden_dim)
        trans_feat_3d = trans_feat[:, N_2d:, :]

        hs_adain_2d = self.dec_2d(trans_feat_2d, self.query_embed_2d.weight.unsqueeze(dim=0).expand(VB, -1, -1))
        hs_adain_3d = self.dec_3d(trans_feat_3d, self.query_embed_3d.weight.unsqueeze(dim=0).expand(B, -1, -1))

        mask_embed_2d = self.mask_embed_2d(hs_adain_2d[-1])
        mask_embed_3d = self.mask_embed_3d(hs_adain_3d[-1])
        
        return mask_embed_2d, mask_embed_3d, hs_adain_2d, hs_adain_3d


class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:
        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function
        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                # https://github.com/pytorch/pytorch/issues/12013
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"

        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


def c2_xavier_fill(module: nn.Module) -> None:
    """
    Initialize `module.weight` using the "XavierFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.
    Args:
        module (torch.nn.Module): module to initialize.
    """
    # Caffe2 implementation of XavierFill in fact
    # corresponds to kaiming_uniform_ in PyTorch
    nn.init.kaiming_uniform_(module.weight, a=1)
    if module.bias is not None:
        # pyre-fixme[6]: Expected `Tensor` for 1st param but got `Union[nn.Module,
        #  torch.Tensor]`.
        nn.init.constant_(module.bias, 0)


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x