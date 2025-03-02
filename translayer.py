import torch
from torch import nn
#import ipdb
from einops import rearrange
import torch.nn.functional as F

# we provide attentions:
# 1.Self_Attention which only calculates self
# 2.Self_Global_Attention which calculates self and global,also from URDMU
# 3. Fast_Attention: self_attention with no skip
# 4. Slow_Attention: search the most important clip from per n clips, and use them as all 4
# 5. Dynamic_Slow_Attention: search for each
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def Query_map(n):
    x = torch.tril(torch.ones(n, n))
    #x[:, 0][0] = 1
    for i in range(n):
        for j in range(i+1):
            x[i, j] = j + 1
    for i in range(x.size(0)):
        nonzero_idx = torch.nonzero(x[i, :])
        if len(nonzero_idx) > 0:
            nonzero_values = x[i, nonzero_idx[:, 0]]
            softmax_values = torch.softmax(nonzero_values, dim=0)
            x[i, nonzero_idx[:, 0]] = softmax_values
    return x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class PreNorm_cross(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x_slow,x_fast, **kwargs):
        return self.fn(x_slow,self.norm(x_fast), **kwargs)

class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class FeedForward_cross(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x_slow,x_fast):
        return self.net(x_fast)

class Self_Global_Attention(nn.Module):#yuandaima
    def __init__(self, dim,flag, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 4, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(2*inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b,n,d=x.size()
        qkvt = self.to_qkv(x).chunk(4, dim = -1)   
        q, k, v, t = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkvt)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn1 = self.attend(dots)

        tmp_ones = torch.ones(n).cuda()
        tmp_n = torch.linspace(1, n, n).cuda()
        tg_tmp = torch.abs(tmp_n * tmp_ones - tmp_n.view(-1,1))
        attn2 = torch.exp(-tg_tmp / torch.exp(torch.tensor(1.)))
        attn2 = (attn2 / attn2.sum(-1)).unsqueeze(0).unsqueeze(1).repeat(b,self.heads, 1, 1)

        out = torch.cat([torch.matmul(attn1, v),torch.matmul(attn2, t)],dim=-1)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Fast_Attention(nn.Module):
    def __init__(self, dim, flag, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 4, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(2 * inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, d = x.size()
        qkvt = self.to_qkv(x).chunk(4, dim=-1)
        q, k, v, t = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkvt)
        step = 4
        yushu = step-n%step
        for i in range(yushu):
            q = torch.cat([q,q[:,:,-1:,:]],dim=-2)
        q = q.view(b,self.heads,step,-1,d//self.heads)[:,:,0,:,:].repeat(1,1,step,1)[:,:,:n,:]
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn1 = self.attend(dots)

        tmp_ones = torch.ones(n).cuda()
        tmp_n = torch.linspace(1, n, n).cuda()
        tg_tmp = torch.abs(tmp_n * tmp_ones - tmp_n.view(-1, 1))
        attn2 = torch.exp(-tg_tmp / torch.exp(torch.tensor(1.)))
        attn2 = (attn2 / attn2.sum(-1)).unsqueeze(0).unsqueeze(1).repeat(b, self.heads, 1, 1)

        out = torch.cat([torch.matmul(attn1, v), torch.matmul(attn2, t)], dim=-1)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

#select a key clip for each 4 clips
class Slow_Attention(nn.Module):
    def __init__(self, dim, flag, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

    def forward(self, x):
        b, n, d = x.size()
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return out

class Self_Global_Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, flag, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Self_Global_Attention(dim, flag=flag, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class Slow_Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, flag, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Slow_Attention(dim, flag=flag, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x,flag):
        if flag == 'Train':
            skip = 4
            B, C, D = x.size()
            x = x.view(B, skip, C // skip, D)
            x = x[:, -1, :, :]
            for attn, ff in self.layers:
                x = attn(x) + x
                x = ff(x) + x
            x = x.repeat(1, 4, 1)
        else:
            for attn, ff in self.layers:
                x = attn(x) + x
                x = ff(x) + x
        return x

class Fast_Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, flag, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Fast_Attention(dim, flag=flag, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class Cross_Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, flag, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm_cross(dim, Cross_Attention(dim, flag=flag, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm_cross(dim, FeedForward_cross(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x_slow,x_fast):
        for attn, ff in self.layers:
            x = attn(x_slow,x_fast) + x_fast
            x = ff(None,x) + x
        return x

#——一对多交叉注意力
class Cross_Attention(nn.Module):
    def __init__(self, dim, flag="Train", heads = 8,qkv_bias=False, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.q_bias = nn.Parameter(torch.zeros(inner_dim)) if qkv_bias else None
        self.v_bias = nn.Parameter(torch.zeros(inner_dim)) if qkv_bias else None

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.flag = flag
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        #self.to_out = nn.Dropout(dropout) if project_out else nn.Identity()

    def forward(self, x_slow,x_fast):
        b, n, d = x_fast.size()
        qkv_slow = self.to_qkv(x_slow).chunk(3, dim=-1)
        q_slow, _, _ = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv_slow)
        qkv_fast = self.to_qkv(x_fast).chunk(3, dim=-1)
        _, k_fast, v_fast = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv_fast)

        attn = self.attend(torch.matmul(q_slow, k_fast.transpose(-1, -2)) * self.scale)
        out = torch.matmul(attn, v_fast)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return out

#顺次交叉注意力
class Cross_Attention1(nn.Module):
    def __init__(self, dim, flag="Train", heads = 8,qkv_bias=False, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.q_bias = nn.Parameter(torch.zeros(inner_dim)) if qkv_bias else None
        self.v_bias = nn.Parameter(torch.zeros(inner_dim)) if qkv_bias else None

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.flag = flag
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Dropout(dropout) if project_out else nn.Identity()

    def forward(self, x):
        b,n,d=x.size()
        qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False),
                              self.v_bias)) if self.q_bias is not None else None
        qkv = F.linear(input=x,weight=self.to_qkv.weight,bias=qkv_bias)
        qkv = qkv.reshape(b,n,3,self.heads,-1).permute(2,0,3,1,4)
        q, k, v = qkv[0],qkv[1],qkv[2]
        ###  q:[0 1 2 3 4...] q_cross:[0 0 1 2 3...]
        q_cross = torch.cat([q[:,:,0:1,:],q[:,:,:-1,:]],dim=2)
        attn_map = (q_cross*self.scale @ k.transpose(-1, -2)).softmax(dim=-1)
        out = (attn_map @ v).transpose(2,1).reshape(b,n,-1)
        out = self.to_out(out)
        return out
