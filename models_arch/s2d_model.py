import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

try:
    from timm.layers import to_2tuple
except ImportError:
    try:
        from timm.models.layers.helpers import to_2tuple
    except ImportError:
        import collections.abc
        from itertools import repeat
        def to_2tuple(x):
            return tuple(x) if isinstance(x, collections.abc.Iterable) else tuple(repeat(x, 2))

try:
    from timm.layers import _assert
except ImportError:
    try:
        from timm.models.layers.trace_utils import _assert
    except ImportError:
        def _assert(condition: bool, message: str = ''):
            assert condition, message

try:
    from timm.layers import PatchEmbed, trunc_normal_
except ImportError:
    from timm.models.layers import PatchEmbed, trunc_normal_

try:
    from timm.models.vision_transformer import _cfg, Attention, DropPath, Mlp, LayerScale, Block
except ImportError:
    from timm.layers import Attention, DropPath, Mlp, LayerScale
    from timm.models.vision_transformer import Block, _cfg

try:
    from timm.models import register_model
except ImportError:
    from timm.models.registry import register_model

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from einops import rearrange

_logger = logging.getLogger(__name__)

def token2feature(tokens):
    B, L, D = tokens.shape
    H = W = int(L ** 0.5)
    x = tokens.permute(0, 2, 1).view(B, D, W, H).contiguous()
    return x

def feature2token(x):
    B, C, W, H = x.shape
    L = W * H
    tokens = x.view(B, C, L).permute(0, 2, 1).contiguous()
    return tokens

class Fovea(nn.Module):

    def __init__(self, smooth=False):
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)

        self.smooth = smooth
        if smooth:
            self.smooth = nn.Parameter(torch.zeros(1) + 10.0)

    def forward(self, x):
        
        b, c, h, w = x.shape
        x = x.contiguous().view(b, c, h * w)

        if self.smooth:
            mask = self.softmax(x * self.smooth)
        else:
            mask = self.softmax(x)
        output = mask * x
        output = output.contiguous().view(b, c, h, w)

        return output

class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25,
                 act_layer=nn.GELU, skip_connect=True,
                 attention=True,
                 num_heads=8, qkv_bias=False, attn_drop=0., drop=0.):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self.attn = Attention(D_hidden_features, num_heads=num_heads,
                              qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop) if attention else nn.Identity()

        self.apply(self._init_weights)
        nn.init.constant_(self.D_fc2.weight, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.attn(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x

class TMAdapter(nn.Module):
    def __init__(self, D_features, num_frames, ratio=0.25):
        super().__init__()
        self.num_frames = num_frames
        self.T_Adapter = Adapter(
            D_features, mlp_ratio=ratio, skip_connect=False, attention=True)
        self.norm = nn.LayerNorm(D_features)
        self.S_Adapter = Adapter(
            D_features, mlp_ratio=ratio, skip_connect=False, attention=False)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        bt, n, d = x.shape

        xt = rearrange(x, '(b t) n d -> (b n) t d', t=self.num_frames)
        xt = self.T_Adapter(xt)
        x = rearrange(xt, '(b n) t d -> (b t) n d', n=n)

        x = self.S_Adapter(self.norm(x))
        return x

class I3DHead(nn.Module):
    
    def __init__(self,
                 num_classes,
                 in_channels,
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 init_std=0.01):
        super().__init__()

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(in_channels, num_classes)

        if self.spatial_type == 'avg':

            self.avg_pool = 'avg_pool'
        else:
            self.avg_pool = None

    def init_weights(self):
        
        trunc_normal_(self.fc_cls, std=self.init_std)

    def forward(self, x):
        
        if self.avg_pool is not None:
            x = x.mean(2, keepdim=True)

        if self.dropout is not None:
            x = self.dropout(x)

        x = x.view(x.shape[0], -1)

        cls_score = self.fc_cls(x)

        return cls_score, x

class Prompt_block(nn.Module):
    def __init__(self, inplanes=None, hide_channel=None, smooth=False, num_frames=1, ratio=0.25):
        super(Prompt_block, self).__init__()

        self.num_frames = num_frames

        self.conv0_0 = nn.Conv2d(
            in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv0_1 = nn.Conv2d(
            in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv1x1 = nn.Conv2d(
            in_channels=hide_channel, out_channels=inplanes, kernel_size=1, stride=1, padding=0)
        self.fovea = Fovea(smooth=smooth)

        self.TMA = TMAdapter(inplanes, num_frames, ratio=ratio)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        
        B, C, W, H = x.shape
        x0 = x[:, 0:int(C / 2), :, :].contiguous()

        x2 = x0.view(B, C // 2, -1).transpose(1, 2).contiguous()
        x2 = self.TMA(x2)

        x0 = self.conv0_0(x0)
        x1 = x[:, int(C / 2):, :, :].contiguous()
        x1 = self.conv0_1(x1)
        x0 = self.fovea(x0) + x1
        x0 = self.conv1x1(x0)
        return x0, x2

class Prompt_PatchEmbed(nn.Module):
    
    def __init__(
            self,
            img_size=14,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=1, stride=1, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        _assert(
            H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(
            W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class VisionTransformer(nn.Module):
    
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            in_chans_l=128,
            num_frames=16,
            num_classes=8,
            prompt_type='deep',
            global_pool='token',
            hidden_dim=8,
            embed_dim=768,
            depth=12,
            adapter_scale=0.25,
            head_dropout_ratio=0.5,
            num_tadapter=1,
            num_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            init_values=None,
            class_token=True,
            no_embed_class=False,
            pre_norm=False,
            fc_norm=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            weight_init='',
            embed_layer=PatchEmbed,
            norm_layer=None,
            act_layer=None,
            block_fn=Block,
    ):
        
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.num_frames = num_frames
        self.global_pool = global_pool

        self.num_features = self.embed_dim = embed_dim
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.grad_checkpointing = False

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,
        )

        num_patches = self.patch_embed.num_patches

        self.patch_embed_prompt = Prompt_PatchEmbed(
            img_size=14, patch_size=patch_size, in_chans=in_chans_l, embed_dim=embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(
            1, 1, embed_dim)) if class_token else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(
            torch.randn(1, embed_len, embed_dim) * .02, requires_grad=False)
        self.temporal_embedding = nn.Parameter(
            torch.zeros(1, num_frames, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        self.prompt_type = prompt_type

        if self.prompt_type in ['shallow', 'deep']:
            prompt_blocks = []
            block_nums = depth if self.prompt_type == 'deep' else 1
            for i in range(block_nums):
                prompt_blocks.append(Prompt_block(
                    inplanes=embed_dim, hide_channel=hidden_dim, smooth=True, num_frames=num_frames, ratio=adapter_scale))
            self.prompt_blocks = nn.Sequential(*prompt_blocks)
            prompt_norms = []
            for i in range(block_nums):
                prompt_norms.append(norm_layer(embed_dim))
            self.prompt_norms = nn.Sequential(*prompt_norms)

        import inspect
        block_params = inspect.signature(block_fn.__init__).parameters
        drop_arg = {'drop': drop_rate} if 'drop' in block_params else {'proj_drop': drop_rate}
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                init_values=init_values,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                **drop_arg
            )
            for i in range(depth)])
        self.ln_post = norm_layer(
            embed_dim) if not use_fc_norm else nn.Identity()

        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head = I3DHead(
            num_classes, embed_dim,dropout_ratio=head_dropout_ratio) if num_classes > 0 else nn.Identity()

        if weight_init != 'skip':
            self.apply(self._init_weights)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.temporal_embedding, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'temporal_embedding', 'cls_token', 'dist_token'}

    def reset_classifier(self, num_classes: int, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
        self.head = I3DHead(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _pos_embed(self, x):
        if self.no_embed_class:

            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(
                    x.shape[0], -1, -1), x), dim=1)
        else:

            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(
                    x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed
        return self.pos_drop(x)

    def forward_features(self, x, a):
        x = self.patch_embed(x)
        a = self.patch_embed_prompt(a)

        if self.prompt_type in ['shallow', 'deep']:
            x_feat = token2feature(self.prompt_norms[0](x))
            a_feat = token2feature(self.prompt_norms[0](a))
            x_feat = torch.cat([x_feat, a_feat], dim=1)
            x_feat, x1 = self.prompt_blocks[0](x_feat)
            x_feat = feature2token(x_feat)
            a = x_feat
            x = x + x1 + x_feat
        else:
            x += a

        x = self._pos_embed(x)

        n = x.shape[1]
        x = rearrange(x, '(b t) n d -> (b n) t d', t=self.num_frames)
        x = x + self.temporal_embedding
        x = rearrange(x, '(b n) t d -> (b t) n d', n=n)

        x = self.norm_pre(x)
        for i, blk in enumerate(self.blocks):
            
            if i >= 1:
                if self.prompt_type in ['deep']:
                    x_ori = x

                    x = self.prompt_norms[i - 1](x)
                    x_feat = token2feature(x[:, 1:])
                    a_feat = token2feature(self.prompt_norms[0](a))
                    x_feat = torch.cat([x_feat, a_feat], dim=1)
                    x_feat, x1 = self.prompt_blocks[i](x_feat)
                    x_feat = feature2token(x_feat)
                    a = x_feat
                    x = torch.cat(
                        [x_ori[:, 0:1], x_ori[:, 1:] + x1 + x_feat], dim=1)
            x = self.blocks[i](x)

        x = self.ln_post(x)
        
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(
                dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        return x

    def forward(self, x, a):
        B, C, T, H, W = x.shape
        assert T == self.num_frames, f'Input video must have {self.num_frames} frames, but got {T} frames'
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.forward_features(x, a)
        x = self.forward_head(x)
        x = rearrange(x, '(b t) c -> b c t', b=B, t=T)
        x = x.unsqueeze(-1).unsqueeze(-1)
        score, x = self.head(x)
        return score, x

def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformer(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            if "model_state_dict" in checkpoint.keys():
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint["model"]
            for key in list(state_dict.keys()):
                if 'landmark' in key:
                    state_dict.pop(key)
            try:
                state_dict['ln_post.weight'] = state_dict.pop('norm.weight')
                state_dict['ln_post.bias'] = state_dict.pop('norm.bias')
            except:
                pass
            if model.patch_embed_prompt.proj.weight.data.shape[1] != state_dict['patch_embed_prompt.proj.weight'].shape[1]:
                del state_dict['patch_embed_prompt.proj.weight']
            missing_keys, unexpected_keys = model.load_state_dict(
                state_dict, strict=False)
            print('Load pretrained model from: ' + pretrained)
            print(f"missing_keys: {missing_keys}")
            print(f"unexpected_keys: {unexpected_keys}")

    return model

@register_model
def s2d_base_patch16_224(pretrained=False, pretrained_cfg=None, **kwargs):
    
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    model.default_cfg = _cfg(mean=IMAGENET_DEFAULT_MEAN,
                             std=IMAGENET_DEFAULT_STD)
    return model
