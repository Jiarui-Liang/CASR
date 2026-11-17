import torch
from torch import nn
from utils import trunc_normal_
from timm.models.vision_transformer import Block
from util.pos_embed import get_2d_sincos_pos_embed

class AnchoringHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

class MultiCropWrapper(nn.Module):
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        backbone.head.fc = nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x, apply_head=True):
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx: end_idx]), apply_head) # return cls or patch
            if isinstance(_out, tuple):
                _out = _out[0]
            output = torch.cat((output, _out))
            start_idx = end_idx

        if apply_head:
            return self.head(output)
        else:
            return output


class CASR(nn.Module):
    def __init__(self, student, teacher, embed_dim, decoder_embed_dim=512, decoder_depth=8,
                 decoder_num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm):
        super(CASR, self).__init__()

        self.student = student
        self.teacher = teacher
        self.embed_dim = embed_dim

        # encoder specifics
        self.num_patches_global = (224//self.student.backbone.patch_embed.patch_size)**2
        self.num_patches_local = (96//self.student.backbone.patch_embed.patch_size)**2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed_global = nn.Parameter(torch.zeros(1, self.num_patches_global+1, embed_dim), requires_grad=False)
        self.pos_embed_local = nn.Parameter(torch.zeros(1, self.num_patches_local+1, embed_dim), requires_grad=False)


        # decoder specifics
        self.decoder_embed = nn.Linear(self.embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed_global = nn.Parameter(torch.zeros(1, self.num_patches_global+1, decoder_embed_dim), requires_grad=False)
        self.decoder_pos_embed_local = nn.Parameter(torch.zeros(1, self.num_patches_local+1, decoder_embed_dim), requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.embed_dim, bias=True)  # decoder to high-level patch
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        # encoder pos
        pos_embed_global = get_2d_sincos_pos_embed(self.pos_embed_global.shape[-1], int(self.num_patches_global**.5), cls_token=True)
        self.pos_embed_global.data.copy_(torch.from_numpy(pos_embed_global).float().unsqueeze(0))

        pos_embed_local = get_2d_sincos_pos_embed(self.pos_embed_local.shape[-1], int(self.num_patches_local**.5), cls_token=True)
        self.pos_embed_local.data.copy_(torch.from_numpy(pos_embed_local).float().unsqueeze(0))

        # decoder pos
        decoder_pos_embed_global = get_2d_sincos_pos_embed(self.decoder_pos_embed_global.shape[-1], int(self.num_patches_global**.5), cls_token=True)
        self.decoder_pos_embed_global.data.copy_(torch.from_numpy(decoder_pos_embed_global).float().unsqueeze(0))

        decoder_pos_embed_local= get_2d_sincos_pos_embed(self.decoder_pos_embed_local.shape[-1], int(self.num_patches_local**.5), cls_token=True)
        self.decoder_pos_embed_local.data.copy_(torch.from_numpy(decoder_pos_embed_local).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
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

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, apply_global_pos=True):
        # embed patches
        x = self.student.backbone.patch_embed(x)

        # add pos embed w/o cls token
        if apply_global_pos:
            x = x + self.pos_embed_global[:, 1:, :]
        else:
            x = x + self.pos_embed_local[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        if apply_global_pos:
            cls_token = self.cls_token + self.pos_embed_global[:, :1, :]
        else:
            cls_token = self.cls_token + self.pos_embed_local[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # dropout
        x = self.student.backbone.pos_drop(x)

        # apply Transformer blocks
        for blk in self.student.backbone.blocks:
            x = blk(x)
        x = self.student.backbone.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore, apply_global_pos=True):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        if apply_global_pos:
            x = x + self.decoder_pos_embed_global
        else:
            x = x + self.decoder_pos_embed_local

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, teacher, pred, mask):
        """
        teacher: [N, P, embed]
        pred: [N, P, embed]
        mask: [N, P], 0 is keep, 1 is remove,
        """
        loss = (pred - teacher) ** 2
        loss = loss.mean(dim=-1)  # [N, P], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, x, epoch, cssa_criterion, mask_ratio=0.75):
        # CS-SA branch
        teacher_anchor = self.teacher(x[:2], apply_head=True)
        student_anchor = self.student(x, apply_head=True)
        cssa_loss = cssa_criterion(student_anchor, teacher_anchor, epoch)

        # MS-SR: global branch
        ms_sr_global_loss = 0
        for i in range(0, 2):
            # teacher
            teacher_global = self.teacher(x[i], apply_head=False)
            teacher_global = teacher_global[:, 1:, :].detach() # remove cls token
            # student
            latent, mask, ids_restore = self.forward_encoder(x[i], mask_ratio, apply_global_pos=True)
            student_global = self.forward_decoder(latent, ids_restore, apply_global_pos=True)
            # loss
            ms_sr_global_loss += self.forward_loss(teacher_global, student_global, mask)
        ms_sr_global_loss = ms_sr_global_loss / len(x[:2])


        # MS-SR: local branch
        ms_sr_local_loss = 0
        for i in range(2, 10):
            # teacher
            teacher_local = self.teacher(x[i], apply_head=False)
            teacher_local = teacher_local[:, 1:, :].detach()  # remove cls token
            # student
            latent, mask, ids_restore = self.forward_encoder(x[i], mask_ratio, apply_global_pos=False)
            student_local = self.forward_decoder(latent, ids_restore, apply_global_pos=False)
            # loss
            ms_sr_local_loss += self.forward_loss(teacher_local, student_local, mask)
        ms_sr_local_loss = ms_sr_local_loss / len(x[2:])

        # Total loss
        loss = cssa_loss + 3 * ms_sr_global_loss + 3 * ms_sr_local_loss

        return loss, cssa_loss, ms_sr_global_loss, ms_sr_local_loss


