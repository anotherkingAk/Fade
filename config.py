"""
FADE — a 3.1T total-parameter / ~44.2B active-parameter Mixture-of-Experts
code model.

Same underlying technique as Mixtral/Switch-Transformer: many experts exist
in memory, but only `top_k` activate per token, so actual per-token compute
is much smaller than the total parameter count.

Decisions made here, since none were specified, along with the reasoning
so nothing is a black box:

- context window (block_size): 16,384 tokens. This is a reasonable,
  achievable target for a code model with RoPE — long enough to hold
  a large file or several files of real code, without pushing into
  context lengths that need extra techniques (e.g. context extension
  tricks) beyond what's implemented here.
- max output tokens per generation call: handled by `generate()` in
  model.py — it's a parameter you pass at inference time, not a fixed
  architectural limit. Practically bounded by block_size minus your
  prompt length.
- input tokens: same pool as output — this is a decoder-only model,
  so there's a single shared context budget, not separate input/output
  budgets like some proprietary APIs expose.
"""

from dataclasses import dataclass


@dataclass
class FadeConfig:
    name: str = "Fade"
    vocab_size: int = 64000
    n_layer: int = 64
    n_embd: int = 6144
    n_head: int = 64
    n_kv_head: int = 8
    block_size: int = 16384        # total context window (input + output share this budget)
    num_experts: int = 160
    top_k: int = 2
    ffn_mult: float = 8 / 3         # -> ffn_dim = 16384 after rounding
    dropout: float = 0.0
    bias: bool = False
    rope_theta: float = 1_000_000.0
    tie_embeddings: bool = True
    aux_loss_weight: float = 0.01
    fim_rate: float = 0.5           # fraction of training data using fill-in-the-middle (code-specific)


def _round_hw(x, mult=128):
    return ((x + mult - 1) // mult) * mult


def count_params(cfg: FadeConfig):
    E, L, V = cfg.n_embd, cfg.n_layer, cfg.vocab_size
    head_dim = E // cfg.n_head

    q = E * E
    kv = 2 * E * (cfg.n_kv_head * head_dim)
    o = E * E
    attn_per_layer = q + kv + o

    ffn_dim = _round_hw(int(cfg.ffn_mult * E))
    per_expert = 3 * E * ffn_dim
    moe_total_per_layer = cfg.num_experts * per_expert
    moe_active_per_layer = cfg.top_k * per_expert
    router_per_layer = E * cfg.num_experts

    per_layer_total = attn_per_layer + moe_total_per_layer + router_per_layer + 2 * E
    per_layer_active = attn_per_layer + moe_active_per_layer + router_per_layer + 2 * E

    embed = V * E + E  # embedding + final norm

    total_params = L * per_layer_total + embed
    active_params = L * per_layer_active + embed

    return total_params, active_params, ffn_dim


if __name__ == "__main__":
    cfg = FadeConfig()
    total, active, ffn_dim = count_params(cfg)
    print(f"Model: {cfg.name}")
    print(f"FFN dim per expert: {ffn_dim}")
    print(f"Context window (block_size): {cfg.block_size:,} tokens")
    print(f"Total parameters:  {total:,}  ({total / 1e12:.3f}T)")
    print(f"Active parameters per token: {active:,}  ({active / 1e9:.2f}B)")
    print(f"Active/Total ratio: {active / total * 100:.3f}%")
