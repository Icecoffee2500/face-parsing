import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from utils.checkpoint_utils import load_checkpoint


class PositionEmbs(nn.Module):
    def __init__(self, num_patches, emb_dim, dropout_rate=0.1):
        super(PositionEmbs, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        out = x + self.pos_embedding

        if self.dropout:
            out = self.dropout(out)

        return out

class GlobalPosEmbed(nn.Module):
    def __init__(self, embed_dim, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.embed_dim = embed_dim // 2
        self.normalize = normalize
        self.temperature = temperature
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.embed_layer = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding= 1, groups = embed_dim)
        
    def forward(self, x):
        b, n, c = x.shape
        patch_n = int((n-1) ** 0.5)
        not_mask = torch.ones((b, patch_n, patch_n), device = x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.embed_dim, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.embed_dim)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2) # B, C, H, W
        pos = self.embed_layer(pos).reshape(b, c, -1).transpose(1, 2)
        pos_cls = torch.zeros((b, 1, c), device = x.device)
        pos =  torch.cat((pos_cls, pos),dim=1)
        return pos + x


class MlpBlock(nn.Module):
    """ Transformer Feed-Forward Block """
    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.1):
        super(MlpBlock, self).__init__()

        # init layers
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.act = nn.GELU()
        if dropout_rate > 0.0:
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)
        else:
            self.dropout1 = None
            self.dropout2 = None

    def forward(self, x):

        out = self.fc1(x)
        out = self.act(out)
        if self.dropout1:
            out = self.dropout1(out)

        out = self.fc2(out)
        out = self.dropout2(out)
        return out


class LinearGeneral(nn.Module):
    def __init__(self, in_dim=(768,), feat_dim=(12, 64)):
        super(LinearGeneral, self).__init__()

        self.weight = nn.Parameter(torch.randn(*in_dim, *feat_dim))
        self.bias = nn.Parameter(torch.zeros(*feat_dim))

    def forward(self, x, dims):
        a = torch.tensordot(x, self.weight, dims=dims) + self.bias
        return a


class SelfAttention(nn.Module):
    def __init__(self, in_dim, heads=8, dropout_rate=0.1):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.head_dim = in_dim // heads
        self.scale = self.head_dim ** 0.5

        self.query = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.key = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.value = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.out = LinearGeneral((self.heads, self.head_dim), (in_dim,))

        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        b, n, _ = x.shape

        q = self.query(x, dims=([2], [0]))
        k = self.key(x, dims=([2], [0]))
        v = self.value(x, dims=([2], [0]))

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn_weights, v)
        out = out.permute(0, 2, 1, 3)

        out = self.out(out, dims=([2, 3], [0, 1]))

        return out


class EncoderBlock(nn.Module):
    def __init__(self, in_dim, mlp_dim, num_heads, dropout_rate=0.1, attn_dropout_rate=0.1):
        super(EncoderBlock, self).__init__()

        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = SelfAttention(in_dim, heads=num_heads, dropout_rate=attn_dropout_rate)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        self.norm2 = nn.LayerNorm(in_dim)
        self.mlp = MlpBlock(in_dim, mlp_dim, in_dim, dropout_rate)

    def forward(self, x):
        residual = x
        out = self.norm1(x)
        out = self.attn(out)
        if self.dropout:
            out = self.dropout(out)
        out += residual
        residual = out

        out = self.norm2(out)
        out = self.mlp(out)
        out += residual
        return out


class Encoder(nn.Module):
    def __init__(self, num_patches, emb_dim, mlp_dim, num_layers=12, num_heads=12, dropout_rate=0.1, attn_dropout_rate=0.0):
        super(Encoder, self).__init__()

        # positional embedding
        # self.pos_embedding = PositionEmbs(num_patches, emb_dim, dropout_rate)

        self.pos_embed = GlobalPosEmbed(emb_dim)
        self.pos_drop = nn.Dropout(p=dropout_rate)

        # encoder blocks
        in_dim = emb_dim
        self.encoder_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = EncoderBlock(in_dim, mlp_dim, num_heads, dropout_rate, attn_dropout_rate)
            self.encoder_layers.append(layer)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x):

        # out = self.pos_embedding(x)
        out = self.pos_drop(self.pos_embed(x))

        for layer in self.encoder_layers:
            out = layer(out)

        out = self.norm(out)
        return out


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self,
                 image_size=(256, 256), # GPE + LPE 에서는 상관없음.
                 patch_size=(16, 16),
                 emb_dim=768,
                 mlp_dim=3072,
                 num_heads=12, #
                 num_layers=12,
                 num_classes=1000,
                 attn_dropout_rate=0.0,
                 dropout_rate=0.1,
                 feat_dim=None): #
        super(VisionTransformer, self).__init__()
        h, w = image_size

        # embedding layer
        fh, fw = patch_size
        gh, gw = h // fh, w // fw
        num_patches = gh * gw
        self.embedding = nn.Conv2d(3, emb_dim, kernel_size=(fh, fw), stride=(fh, fw))
        # class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))

        # transformer
        self.transformer = Encoder(
            num_patches=num_patches,
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate)

        # classfier
        # self.classifier = nn.Linear(emb_dim, num_classes)

        # 채널 맞추기용 1x1 conv
        self.proj16 = nn.Conv2d(emb_dim, 256, kernel_size=1, bias=False)
        self.proj8  = nn.Conv2d(emb_dim, 128, kernel_size=1, bias=False)
        self.proj32 = nn.Conv2d(emb_dim, 512, kernel_size=1, bias=False)

        # 다운샘플용 (학습형 conv 추천)
        self.down32 = nn.Conv2d(emb_dim, emb_dim, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        # print(f"vit input shape: {x.shape}")
        emb = self.embedding(x)     # (n, c, gh, gw)
        # print(f"vit emb shape: {emb.shape}")
        emb = emb.permute(0, 2, 3, 1)  # (n, gh, hw, c)
        b, h, w, c = emb.shape
        emb = emb.reshape(b, h * w, c)

        # prepend class token
        cls_token = self.cls_token.repeat(b, 1, 1)
        emb = torch.cat([cls_token, emb], dim=1)

        # transformer
        feat = self.transformer(emb)
        # print(f"vit feat shape: {feat.shape}")

        # feature extraction
        feat = feat[:, 1:]
        # print(f"vit feat shape: {feat.shape}")
        feat = feat.reshape(b, -1, h, w)
        # print(f"vit feat shape: {feat.shape}")

        # feat16
        feat16 = self.proj16(feat)   # (B, 256, 32, 32)

        # feat8 (업샘플)
        feat8 = F.interpolate(feat, scale_factor=2, mode="bilinear")  # (B, 384, 64, 64)
        feat8 = self.proj8(feat8)                                    # (B, 128, 64, 64)

        # feat32 (다운샘플)
        feat32 = self.down32(feat)  # (B, 384, 16, 16)
        feat32 = self.proj32(feat32)  # (B, 512, 16, 16)

        # # classifier
        # logits = self.classifier(feat[:, 0])
        # return logits
        return feat8, feat16, feat32


def vit_small(patch_size=(16, 16), pretrained_weights_path="pretrained_weights/mae_pretrain_vit_small.pth"):
    model = VisionTransformer(
        patch_size=patch_size,
        emb_dim=384,
        mlp_dim=1536,
        num_heads=6,
        num_layers=12,
    )
    if pretrained_weights_path is not None:
        # state_dict = torch.load(pretrained_weights_path)['state_dict']
        # state_dict = torch.load(pretrained_weights_path)
        # # model.load_state_dict(state_dict)
        # missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        # print(f"Missing keys (intentionally ignored): {missing_keys}")
        # print(f"Unexpected keys: {unexpected_keys}")
        # print(f"Load pretrained weights from {pretrained_weights_path}")

        load_checkpoint(model, pretrained_weights_path)
        print(f"Load pretrained weights from {pretrained_weights_path}")
    return model

if __name__ == '__main__':
    # model = VisionTransformer(num_layers=2)
    model = vit_small()
    # x = torch.randn((2, 3, 256, 256))
    x = torch.randn((32, 3, 512, 512))
    feat8, feat16, feat32 = model(x)
    print(f"vit feat8 shape: {feat8.shape}")
    print(f"vit feat16 shape: {feat16.shape}")
    print(f"vit feat32 shape: {feat32.shape}")

    # state_dict = model.state_dict()

    # for key, value in state_dict.items():
    #     print("{}: {}".format(key, value.shape))