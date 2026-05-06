"""
cross_attention_nki.py

Fused cross-attention NKI kernel for VLA action heads.
Replaces QK matmul + softmax + AV matmul with a single on-chip pass.

Usage inside NeuronDenoisingWrapper.forward():
    from cross_attention_nki import cross_attention_kernel, get_tile_size

    # Call once in __init__ — shapes are static
    self.tile_size = get_tile_size(
        context_len=config.num_conditioning_tokens,
        action_seq=config.action_chunk_size,
        head_dim=config.attention_head_dim,
        batch_heads=config.neuron_config.batch_size * config.num_heads,
    )

    # Call inside forward()
    output = torch.empty_like(q)
    cross_attention_kernel[self.tile_size](q, k, v, scale, output)

Shapes:
    q:         [batch_heads, action_seq, head_dim]   BF16
    k, v:      [batch_heads, context_len, head_dim]  BF16
    output:    [batch_heads, action_seq, head_dim]   BF16

KV caching: k and v come from VLM conditioning tokens which are identical
across all N denoising steps. Pre-project K and V before the loop and pass
them directly to avoid N-1 redundant projections.
"""

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl


def get_tile_size(
    context_len: int,
    action_seq: int,
    head_dim: int,
    batch_heads: int,
    dtype_bytes: int = 2,
) -> int:
    """
    Compute a safe TILE_SIZE for the context_len dimension before calling
    the kernel. Called from Python (not inside @nki.jit) so the result
    can be used as a compile-time constant via cross_attention_kernel[tile_size].

    Args:
        context_len:  Number of conditioning tokens (VLM output seq length).
        action_seq:   Number of action tokens (action_chunk_size).
        head_dim:     Attention head dimension.
        batch_heads:  batch_size * num_attention_heads.
        dtype_bytes:  Bytes per element (2 for BF16).

    Returns:
        TILE_SIZE as a power of 2, clamped to context_len.
        If context_len fits entirely in SBUF, returns context_len (no tiling).
    """
    # Trainium SBUF: 24 MB per NeuronCore
    SBUF_BYTES = 24 * 1024 * 1024

    # Q is always fully loaded — reserve space for it
    q_bytes = batch_heads * action_seq * head_dim * dtype_bytes

    # Remaining SBUF split 3 ways: K tile, V tile, scores tile
    remaining = (SBUF_BYTES - q_bytes) // 3
    tile_elements = remaining // (batch_heads * head_dim * dtype_bytes)

    # Round down to power of 2 for alignment
    tile = 1
    while tile * 2 <= tile_elements:
        tile *= 2

    # Clamp to context_len — if it fits in one tile, no loop needed
    return min(tile, context_len)


@nki.jit
def cross_attention_kernel[tile_size](q, k, v, scale, outputdest):
    """
    Fused cross-attention kernel. tile_size is a compile-time constant
    passed via bracket syntax: cross_attention_kernel[tile_size](q, k, v, scale, out)

    Args:
        q:          [batch_heads, action_seq, head_dim]   BF16
        k:          [batch_heads, context_len, head_dim]  BF16
        v:          [batch_heads, context_len, head_dim]  BF16
        scale:      float scalar = 1 / sqrt(head_dim)
        outputdest: [batch_heads, action_seq, head_dim]   BF16  (pre-allocated)
    """
    batch_heads, action_seq, head_dim = q.shape
    _, context_len, _ = k.shape

    i_bh = nl.arange(batch_heads)[:, None, None]
    i_q  = nl.arange(action_seq)[None, :, None]
    i_t  = nl.arange(tile_size)[None, :, None]
    i_d  = nl.arange(head_dim)[None, None, :]

    # Load Q entirely — small, always fits in SBUF
    q_imp = nl.load(q[i_bh, i_q, i_d])

    # Accumulate QK scores across context tiles
    scores = nl.zeros((batch_heads, action_seq, context_len), dtype=nl.float32)
    for chunk in nl.affine_range(0, context_len, tile_size):
        i_kv = chunk + i_t
        k_chunk = nl.load(k[i_bh, i_kv, i_d])
        scores[i_bh, i_q, i_kv] = nl.matmul(q_imp, k_chunk, transpose_y=True) * scale

    # Softmax over full context dimension
    weights = nl.softmax(scores, axis=2)

    # Accumulate weighted V across context tiles
    output = nl.zeros((batch_heads, action_seq, head_dim), dtype=nl.float32)
    for chunk in nl.affine_range(0, context_len, tile_size):
        i_kv = chunk + i_t
        v_chunk = nl.load(v[i_bh, i_kv, i_d])
        output += nl.matmul(weights[i_bh, i_q, i_kv], v_chunk)

    nl.store(outputdest[i_bh, i_q, i_d], output)
