import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CausalAttention(nn.Module):
    def __init__(self, n_head, dim, max_seqlen=64):
        super(CausalAttention, self).__init__()
        assert dim % n_head == 0, f'{dim=}应该是{n_head=}的整数倍'
        self.n_head = n_head
        self.dim = dim
        self.max_seqlen = max_seqlen

        self.Wq = nn.Linear(self.dim, self.dim)
        self.Wk = nn.Linear(self.dim, self.dim)
        self.Wv = nn.Linear(self.dim, self.dim)
        self.Wo = nn.Linear(self.dim, self.dim)

        self.register_buffer("bias", torch.tril(torch.ones(self.max_seqlen, self.max_seqlen))
                                     .view(1, 1, self.max_seqlen, self.max_seqlen))

    def forward(self, x):
        # [1, 16, 512]
        B, T, C = x.shape
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)
        q = q.reshape(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # -> [B, n_head, T, C // self.n_head]
        k = k.reshape(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.reshape(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        # 进行casual mask
        att = att.masked_fill(self.bias[:, :, T, T]==0, float('-inf'))
        y = F.softmax(att, dim=-1) @ v
        y = y.reshape(B, T, C)
        return self.Wo(y)

        
class Block(nn.Module):
    def __init__(self, n_head, dim, max_seqlen=64):
        super(Block, self).__init__()
        self.n_head = n_head
        self.dim = dim
        self.max_seqlen = max_seqlen

        self.ln_1 = nn.LayerNorm(self.dim)
        self.attn = CausalAttention(n_head=self.n_head, dim=self.dim, max_seqlen=self.max_seqlen)
        self.ln_2 = nn.LayerNorm(self.dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, 4 * self.dim),
            nn.GELU(),
            nn.Linear(4 * self.dim, self.dim)
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, n_layer, out_dim, n_head, dim, max_seqlen=64):
        super(Transformer, self).__init__()
        self.n_head = n_head
        self.dim = dim
        self.max_seqlen = max_seqlen
        self.n_layer = n_layer
        self.out_dim = out_dim
        self.transformer = nn.ModuleDict(dict(
            wpe = nn.Embedding(self.max_seqlen, self.dim),
            h = nn.ModuleList([Block(n_head=self.n_head, dim=self.dim, max_seqlen=self.max_seqlen) for _ in range(self.n_layer)]),
            ln_f = nn.LayerNorm(self.dim)
        ))

        self.lm_head = nn.Linear(self.dim, self.out_dim)

        # 计算参数量，如太大请减少n_layer
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("Transformer 参数量: %.2fM" % (n_params/1e6,))

    def forward(self, x):
        B, T, C = x.shape
        device = x.device
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)

        pos_emb = self.transformer.wpe(pos)
        x = x + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)


class RLTransformer(nn.Module):
    def __init__(self, n_layer, state_dim, out_dim, n_head, dim, max_seqlen=64):
        super(RLTransformer, self).__init__()
        self.expand_layer = nn.Linear(state_dim, dim)
        self.transformer = Transformer(n_layer=n_layer, out_dim=out_dim, n_head=n_head, dim=dim, max_seqlen=max_seqlen)
    
    def forward(self, x):
        x = x.unsqueeze(0)
        # [batch, step, dim]
        x = self.expand_layer(x)
        x = self.transformer(x)
        # 最后一个step的输出有全部前向信息
        x = x.squeeze(0)
        return x[-1, :]



if __name__ == '__main__':
    feature_train_1 = torch.rand(20, 8)
    # feature_train_2 = torch.rand(1, 3, 8)
    # feature_train = torch.stack([feature_train_1, feature_train_2])
    rltransformer = RLTransformer(n_layer=1, out_dim=4, n_head=2, dim=256, state_dim=8)
    print(rltransformer(feature_train_1).shape)





