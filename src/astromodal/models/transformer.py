"""
Transformer model for causal language modeling in astromodal.

This module implements a standard causal transformer decoder for autoregressive
sequence modeling of multimodal astronomical data.

Classes
-------
CausalTransformerLM
    Causal transformer language model with token, type, and position embeddings
"""

from typing import Dict, Optional
import torch
import torch.nn as nn


class CausalTransformerLM(nn.Module):
    """
    Causal transformer language model for autoregressive sequence generation.

    This model implements a standard decoder-only transformer architecture with:
    - Token embeddings for vocabulary
    - Type embeddings for different modalities (image, scalar, spectrum, etc.)
    - Learned positional embeddings
    - Causal attention masking for autoregressive generation
    - Optional key padding mask for variable-length sequences

    The architecture uses PyTorch's built-in TransformerEncoder with causal masking
    to implement the decoder-only transformer pattern common in language models.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary (total number of tokens)
    n_types : int
        Number of token types (e.g., BOS, image, scalar, spectra groups)
    d_model : int
        Dimension of model embeddings and hidden states
    n_layers : int
        Number of transformer layers
    n_heads : int
        Number of attention heads per layer
    d_ff : int
        Dimension of feedforward network
    dropout : float
        Dropout probability applied throughout the model
    max_len : int
        Maximum sequence length for positional embeddings

    Attributes
    ----------
    tok_emb : nn.Embedding
        Token embedding layer
    type_emb : nn.Embedding
        Token type embedding layer
    pos_emb : nn.Embedding
        Positional embedding layer
    enc : nn.TransformerEncoder
        Transformer encoder layers (used as decoder with causal mask)
    ln : nn.LayerNorm
        Final layer normalization
    head : nn.Linear
        Output projection to vocabulary
    drop : nn.Dropout
        Input dropout layer

    Examples
    --------
    >>> # Create a small transformer for testing
    >>> model = CausalTransformerLM(
    ...     vocab_size=1000,
    ...     n_types=10,
    ...     d_model=256,
    ...     n_layers=6,
    ...     n_heads=8,
    ...     d_ff=1024,
    ...     dropout=0.1,
    ...     max_len=512
    ... )
    >>> tokens = torch.randint(0, 1000, (4, 128))
    >>> types = torch.randint(0, 10, (4, 128))
    >>> attn_mask = torch.ones(4, 128, dtype=torch.bool)
    >>> logits = model(tokens, types, attn_mask)
    >>> logits.shape
    torch.Size([4, 128, 1000])

    >>> # Use with configuration
    >>> from astromodal.config import load_config
    >>> config = load_config()
    >>> model = CausalTransformerLM(
    ...     vocab_size=config.vocab.image + config.vocab.scalar + config.vocab.spectrum * 5 + 5,
    ...     n_types=15,
    ...     d_model=config.model.d_model,
    ...     n_layers=config.model.n_layers,
    ...     n_heads=config.model.n_heads,
    ...     d_ff=config.model.d_ff,
    ...     dropout=config.model.dropout,
    ...     max_len=config.sequences.max_seq_len
    ... )

    Notes
    -----
    The model uses:
    - GELU activation in feedforward layers
    - Pre-layer normalization (norm_first=False)
    - Causal attention mask to prevent attending to future tokens
    - Key padding mask to handle variable-length sequences
    - Cached causal masks for efficiency across forward passes
    - Tied or separate token and output embeddings (currently separate)

    The causal mask is cached per sequence length to avoid recomputation,
    which significantly improves performance for common sequence lengths.
    """

    def __init__(
        self,
        vocab_size: int,
        n_types: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        max_len: int,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_types = n_types
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.max_len = max_len

        # Embedding layers
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.type_emb = nn.Embedding(n_types, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        # Transformer encoder layers (used as decoder with causal mask)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Output layers
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.drop = nn.Dropout(dropout)

        # Cache for causal masks to avoid recomputation
        self._mask_cache: Dict[int, torch.Tensor] = {}

    def _causal_mask(self, L: int, device: torch.device) -> torch.Tensor:
        """
        Get or create causal attention mask for sequence length L.

        The mask is a boolean upper triangular matrix where True indicates
        positions that should be masked (cannot be attended to).

        Parameters
        ----------
        L : int
            Sequence length
        device : torch.device
            Device to create mask on

        Returns
        -------
        torch.Tensor
            Boolean mask of shape (L, L) with True in upper triangle

        Examples
        --------
        >>> model = CausalTransformerLM(1000, 10, 256, 6, 8, 1024, 0.1, 512)
        >>> mask = model._causal_mask(5, torch.device('cpu'))
        >>> mask
        tensor([[False,  True,  True,  True,  True],
                [False, False,  True,  True,  True],
                [False, False, False,  True,  True],
                [False, False, False, False,  True],
                [False, False, False, False, False]])
        """
        m = self._mask_cache.get(L, None)
        if m is None or m.device != device:
            # Create upper triangular matrix with True in upper triangle
            m = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)
            self._mask_cache[L] = m
        return m

    def forward(
        self,
        tokens: torch.Tensor,
        types: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the causal transformer.

        Parameters
        ----------
        tokens : torch.Tensor
            Token indices of shape (B, L)
        types : torch.Tensor
            Token type indices of shape (B, L)
        attn_mask : torch.Tensor
            Boolean attention mask of shape (B, L), True for valid positions

        Returns
        -------
        torch.Tensor
            Logits over vocabulary of shape (B, L, vocab_size)

        Raises
        ------
        ValueError
            If sequence length exceeds max_len

        Examples
        --------
        >>> model = CausalTransformerLM(1000, 10, 256, 6, 8, 1024, 0.1, 512)
        >>> tokens = torch.randint(0, 1000, (4, 128))
        >>> types = torch.randint(0, 10, (4, 128))
        >>> attn_mask = torch.ones(4, 128, dtype=torch.bool)
        >>> logits = model(tokens, types, attn_mask)
        >>> logits.shape
        torch.Size([4, 128, 1000])

        >>> # With padding (last 20 positions are padding)
        >>> attn_mask = torch.ones(4, 128, dtype=torch.bool)
        >>> attn_mask[:, 108:] = False
        >>> logits = model(tokens, types, attn_mask)
        >>> logits.shape
        torch.Size([4, 128, 1000])

        Notes
        -----
        The forward pass:
        1. Combines token, type, and position embeddings
        2. Applies dropout to embeddings
        3. Creates causal attention mask to prevent attending to future positions
        4. Creates key padding mask from attn_mask (inverted)
        5. Passes through transformer layers with both masks
        6. Applies final layer normalization
        7. Projects to vocabulary size
        """
        B, L = tokens.shape

        # Check sequence length
        if L > self.max_len:
            raise ValueError(f"Sequence length {L} exceeds max_len {self.max_len}")

        # Create position indices
        pos = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, L)

        # Combine embeddings: token + type + position
        x = self.tok_emb(tokens) + self.type_emb(types) + self.pos_emb(pos)
        x = self.drop(x)

        # Create masks
        # Key padding mask: True for positions to ignore (inverted from attn_mask)
        key_padding_mask = ~attn_mask

        # Causal mask: prevents attending to future positions
        causal_mask = self._causal_mask(L, tokens.device)

        # Forward through transformer
        h = self.enc(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)

        # Output projection
        h = self.ln(h)
        logits = self.head(h)

        return logits

    def generate(
        self,
        tokens: torch.Tensor,
        types: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Parameters
        ----------
        tokens : torch.Tensor
            Initial token sequence of shape (B, L)
        types : torch.Tensor
            Token type sequence of shape (B, L)
        max_new_tokens : int, default=100
            Maximum number of new tokens to generate
        temperature : float, default=1.0
            Sampling temperature (higher = more random)
        top_k : Optional[int], default=None
            If set, only sample from top k most likely tokens

        Returns
        -------
        torch.Tensor
            Extended token sequence of shape (B, L + max_new_tokens)

        Examples
        --------
        >>> model = CausalTransformerLM(1000, 10, 256, 6, 8, 1024, 0.1, 512)
        >>> model.eval()
        >>> tokens = torch.randint(0, 1000, (1, 10))
        >>> types = torch.randint(0, 10, (1, 10))
        >>> generated = model.generate(tokens, types, max_new_tokens=20)
        >>> generated.shape
        torch.Size([1, 30])

        Notes
        -----
        This is a simple greedy/sampling generation function. For more sophisticated
        generation strategies (beam search, nucleus sampling, etc.), consider using
        specialized libraries like Hugging Face Transformers.
        """
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get predictions for the current sequence
                B, L = tokens.shape
                if L >= self.max_len:
                    break

                attn_mask = torch.ones_like(tokens, dtype=torch.bool)
                logits = self(tokens, types, attn_mask)

                # Focus on the last position
                logits = logits[:, -1, :] / temperature

                # Optional top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('inf')

                # Sample from the distribution
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to sequence
                tokens = torch.cat([tokens, next_token], dim=1)
                # Use same type as last token (could be improved)
                types = torch.cat([types, types[:, -1:]], dim=1)

        return tokens
