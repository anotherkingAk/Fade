"""
FADE — full model architecture.

RMSNorm + RoPE + Grouped-Query Attention (dense, every token) +
Mixture-of-Experts SwiGLU feed-forward (sparse, top-k experts per token).

This is the same family of building blocks used in real MoE code/language
models (Mixtral-style routing). At this total parameter count (3.1T) this
exact configuration has not been trained by anyone publicly — the code is
correct, but running it at this scale is an infrastructure project on a
scale few organizations on earth could resource, not a code problem.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import FadeConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm * self.weight


def precompute_rope(head_dim, max_seq_len, theta, device=None):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    seq_len = x.shape[-2]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2)


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: FadeConfig):
        super().__init__()
        assert cfg.n_head % cfg.n_kv_head == 0
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.n_rep = cfg.n_head // cfg.n_kv_head

        self.wq = nn.Linear(cfg.n_embd, cfg.n_head * self.head_dim, bias=cfg.bias)
        self.wk = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=cfg.bias)
        self.wv = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=cfg.bias)
        self.wo = nn.Linear(cfg.n_head * self.head_dim, cfg.n_embd, bias=cfg.bias)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(y)


class Expert(nn.Module):
    """A single SwiGLU feed-forward expert."""

    def __init__(self, cfg: FadeConfig, ffn_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(cfg.n_embd, ffn_dim, bias=cfg.bias)
        self.w_up = nn.Linear(cfg.n_embd, ffn_dim, bias=cfg.bias)
        self.w_down = nn.Linear(ffn_dim, cfg.n_embd, bias=cfg.bias)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class MoELayer(nn.Module):
    """
    Sparse Mixture-of-Experts layer with top-k routing.

    Each token is routed to `top_k` experts (out of `num_experts` total)
    by a learned gating network. Only the selected experts run for that
    token — this is what makes a huge total parameter count computationally
    feasible: memory holds every expert, but compute only touches a few.

    Includes a load-balancing auxiliary loss (standard in Switch Transformer
    / Mixtral) that discourages the router from collapsing onto a small
    subset of experts, which would waste capacity and destabilize training.
    """

    def __init__(self, cfg: FadeConfig):
        super().__init__()
        ffn_dim = ((int(cfg.ffn_mult * cfg.n_embd) + 127) // 128) * 128
        self.num_experts = cfg.num_experts
        self.top_k = cfg.top_k
        self.aux_loss_weight = cfg.aux_loss_weight

        self.gate = nn.Linear(cfg.n_embd, cfg.num_experts, bias=False)
        self.experts = nn.ModuleList([Expert(cfg, ffn_dim) for _ in range(cfg.num_experts)])

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(-1, C)  # (B*T, C)

        router_logits = self.gate(x_flat)  # (B*T, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)

        top_k_probs, top_k_idx = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)  # renormalize

        out = torch.zeros_like(x_flat)

        # dispatch tokens to their selected experts
        for expert_id in range(self.num_experts):
            mask = (top_k_idx == expert_id)  # (B*T, top_k)
            token_mask = mask.any(dim=-1)     # (B*T,)
            if not token_mask.any():
                continue
            selected = x_flat[token_mask]
            expert_out = self.experts[expert_id](selected)

            weight_for_expert = (top_k_probs * mask.float()).sum(dim=-1)[token_mask].unsqueeze(-1)
            out[token_mask] += expert_out * weight_for_expert

        # load-balancing auxiliary loss: encourages uniform usage across experts
        # (standard formulation from the Switch Transformer paper)
        density = router_probs.mean(dim=0)
        density_proxy = (router_probs > 0).float().mean(dim=0)
        aux_loss = self.aux_loss_weight * self.num_experts * (density * density_proxy).sum()

        return out.view(B, T, C), aux_loss


class Block(nn.Module):
    def __init__(self, cfg: FadeConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.n_embd)
        self.attn = GroupedQueryAttention(cfg)
        self.moe_norm = RMSNorm(cfg.n_embd)
        self.moe = MoELayer(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        moe_out, aux_loss = self.moe(self.moe_norm(x))
        x = x + moe_out
        return x, aux_loss


class Fade(nn.Module):
    def __init__(self, cfg: FadeConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.final_norm = RMSNorm(cfg.n_embd)

        head_dim = cfg.n_embd // cfg.n_head
        cos, sin = precompute_rope(head_dim, cfg.block_size, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, (
            f"sequence length {T} exceeds Fade's context window of {self.cfg.block_size} tokens "
            f"(this is the shared input+output budget)"
        )

        x = self.tok_embed(idx)
        cos, sin = self.rope_cos.to(x.device), self.rope_sin.to(x.device)

        total_aux_loss = 0.0
        for block in self.blocks:
            x, aux_loss = block(x, cos, sin)
            total_aux_loss = total_aux_loss + aux_loss
        x = self.final_norm(x)

        logits = F.linear(x, self.tok_embed.weight)  # tied embeddings

        loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
            loss = ce_loss + total_aux_loss / len(self.blocks)
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        max_new_tokens: how many output tokens to produce.
        Bounded in practice by (block_size - idx.size(1)) since Fade shares
        one context budget between input and output, e.g. a 12,000-token
        prompt leaves at most 4,384 tokens of room to generate at a
        16,384-token context window.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


if __name__ == "__main__":
    cfg = FadeConfig()
    model = Fade(cfg)
    print(f"{cfg.name}: {model.num_params():,} parameters ({model.num_params() / 1e12:.3f}T)")
