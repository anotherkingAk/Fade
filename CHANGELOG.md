# Changelog

All notable changes to this project are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] — Initial release

### Added
- Fade architecture: RMSNorm, RoPE, Grouped-Query Attention, Mixture-of-Experts
  SwiGLU feed-forward (160 experts, top-2 routing)
- Configuration targeting 3.098T total parameters / 44.55B active per token
- Training script with AdamW, cosine LR schedule, gradient accumulation,
  mixed precision, and DDP support
- Fill-in-the-middle (FIM) data pipeline for code-specific pretraining
- 16,384-token context window

### Status
- Architecture and training pipeline only — **no trained weights yet**
- Requires expert-parallel / model-parallel infrastructure to train at
  full scale (see README)
