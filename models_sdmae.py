# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Block
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from util.pos_embed import get_2d_sincos_pos_embed

l2_loss = nn.MSELoss()
l1_loss = nn.L1Loss()
criterion = nn.CrossEntropyLoss().cuda()


def _build_mlp(num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
    mlp = []
    for l in range(num_layers):
        dim1 = input_dim if l == 0 else mlp_dim
        dim2 = output_dim if l == num_layers - 1 else mlp_dim

        mlp.append(nn.Linear(dim1, dim2, bias=False))

        if l < num_layers - 1:
            mlp.append(nn.BatchNorm1d(dim2))
            mlp.append(nn.ReLU(inplace=True))
        elif last_bn:
            # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
            # for simplicity, we further removed gamma in BN
            mlp.append(nn.BatchNorm1d(dim2, affine=False))

    return nn.Sequential(*mlp)

class ctr_model(nn.Module):
    def __init__(self, embed_dim, mlp_dim, depth=24, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super(ctr_model, self).__init__()
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.head = _build_mlp(3, embed_dim, mlp_dim, embed_dim)
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = x[:, :1, :]
        x = torch.squeeze(x)
        x = self.head(x)
        return x

class Encoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super(Encoder, self).__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # 初始位置代码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding 所有位置代码

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

    def forward(self, x, ids_keep):
        # embed patches
        x = self.patch_embed(x)
        # x = x.detach()
        
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  \

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

class SmallDatasetMaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16, batch_size=64, mlp_dim=4096,
                 decoder_embed_dim=512, decoder_depth=8, ctr_depth=2, decoder_num_heads=16, num_classes=64,
                 mlp_ratio=4., m=0.999, norm_layer=nn.LayerNorm, norm_pix_loss=False, stop_grad_conv1=True):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.m = m
        self.stop_grad_conv1 = stop_grad_conv1
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.num_patches = num_patches
        self.batch_szie = batch_size
        self.encoder = Encoder(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                               embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                               norm_layer=norm_layer)

        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        self.base_ctr = ctr_model(embed_dim, mlp_dim, ctr_depth, num_heads, mlp_ratio, norm_layer)
        self.momentum_ctr = ctr_model(embed_dim, mlp_dim, ctr_depth, num_heads, mlp_ratio, norm_layer)
        self.predictor = _build_mlp(2, embed_dim, mlp_dim, embed_dim)

        for param_q, param_k in zip(self.base_ctr.parameters(), self.momentum_ctr.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        # --------------------------------------------------------------------------
        # --------------------------------------------------------------------------
        self.locality = nn.Sequential(nn.Linear(embed_dim, int(embed_dim//2)),
                                      nn.Linear(int(embed_dim//2), self.num_patches)
                                      )
        # --------------------------------------------------------------------------
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.encoder.pos_embed.shape[-1], int(self.encoder.patch_embed.num_patches**.5), cls_token=True)
        self.encoder.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.encoder.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.encoder.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        if self.stop_grad_conv1:
            self.encoder.patch_embed.proj.weight.requires_grad = False
            self.encoder.patch_embed.proj.bias.requires_grad = False

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.encoder.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.no_grad()
    def _momentum_update_key_ctr(self):
        for param_q, param_k in zip(self.base_ctr.parameters(), self.momentum_ctr.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.encoder.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, N, L, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device='cuda')  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep_q = ids_shuffle[:, :len_keep]
        ids_keep_k = ids_shuffle[:, len_keep:]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device='cuda')
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return ids_keep_q, ids_keep_k, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token

        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        pixel_token = x[:, 1:, :]

        return x, pixel_token

    def con_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    @torch.no_grad()
    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        tensors_gather = [torch.ones_like(tensor)
                          for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output

    def ctr_loss(self, q, k, T=1.0):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        # k = self.concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / T
        N = logits.shape[0]  # batch size per GPU
        # labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        labels = torch.arange(N, dtype=torch.long).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * T)

    def loc_loss(self, pred_loc, real_loc):
        real_loc = F.one_hot(real_loc, num_classes=self.num_patches).float()
        loss = l2_loss(pred_loc, real_loc)
        return loss

    def forward(self, s_imgs, w_imgs, mask_ratio=0.75):

        ids_keep_q, ids_keep_k, mask, ids_restore = self.random_masking(self.batch_szie, self.num_patches, mask_ratio)

        latent_q = self.encoder(s_imgs, ids_keep_q)
        latent_k = self.encoder(w_imgs, ids_keep_q)

        location_q = self.locality(latent_q[:, 1:, :])
        location_k = self.locality(latent_k[:, 1:, :])

        pred_q, pixel_token_q = self.forward_decoder(latent_q, ids_restore)  # [N, L, p*p*3]
        pred_k, pixel_token_k = self.forward_decoder(latent_k, ids_restore)

        q1 = self.predictor(self.base_ctr(pred_q))
        q2 = self.predictor(self.base_ctr(pred_k))

        with torch.no_grad():  # no gradient
            self._momentum_update_key_ctr()  # update the momentum encoder
            # compute momentum features as targets
            k1 = self.momentum_ctr(pred_q)
            k2 = self.momentum_ctr(pred_k)

        construction_loss = self.con_loss(s_imgs, pixel_token_q, mask) + self.con_loss(w_imgs, pixel_token_k, mask)

        loc_loss = self.loc_loss(location_q, ids_keep_q) + self.loc_loss(location_k, ids_keep_q)
        contrastive_loss = 0.1 * (self.ctr_loss(q1, k2) + self.ctr_loss(q2, k1))
        loss = contrastive_loss + construction_loss + loc_loss


        return loss, mask


def sdmae_vit_base_patch16_dec512d8b(**kwargs):
    model = SmallDatasetMaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, batch_size=64,
        decoder_embed_dim=128, decoder_depth=1, ctr_depth=2, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def sdmae_vit_large_patch16_dec512d8b(**kwargs):
    model = SmallDatasetMaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, batch_size=64,
        decoder_embed_dim=512, decoder_depth=8, ctr_depth=4, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def sdmae_vit_huge_patch14_dec512d8b(**kwargs):
    model = SmallDatasetMaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, batch_size=64,
        decoder_embed_dim=512, decoder_depth=8, ctr_depth=4, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
sdmae_vit_base_patch16 = sdmae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
sdmae_vit_large_patch16 = sdmae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
sdmae_vit_huge_patch14 = sdmae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks









