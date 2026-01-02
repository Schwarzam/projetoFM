"""
AION-1 Encoder-Decoder Transformer for Masked Multimodal Modeling

This implements the architecture from the AION paper (Section 5):
- Separate encoder and decoder
- Encoder processes observed tokens WITH token embeddings
- Decoder generates masked tokens WITHOUT token embeddings (only modality + position)
- Training objective: predict masked tokens (NOT next-token prediction)

References:
    AION-1: Omnimodal Foundation Model for Astronomical Sciences
    arXiv:2510.17960v1
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import math


class AIONTransformer(nn.Module):
    """
    AION-1 style encoder-decoder transformer for masked multimodal modeling.

    Unlike causal/autoregressive transformers (GPT-style), this model:
    1. Uses an encoder-decoder architecture
    2. Trains with masked token prediction (like BERT/4M)
    3. Decoder queries do NOT receive token embeddings

    Architecture follows Figure 1 and Section 5.1 of the AION paper.

    Parameters
    ----------
    vocab_size : int
        Total vocabulary size (all modalities combined)
    n_types : int
        Number of token type embeddings (modality markers)
    d_model : int
        Model dimension
    n_enc_layers : int
        Number of encoder layers
    n_dec_layers : int
        Number of decoder layers
    n_heads : int
        Number of attention heads
    d_ff : int
        Feedforward dimension
    dropout : float
        Dropout probability
    max_len : int
        Maximum sequence length for positional embeddings
    """

    def __init__(
        self,
        vocab_size: int,
        n_types: int,
        d_model: int,
        n_enc_layers: int,
        n_dec_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        max_len: int,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_types = n_types
        self.d_model = d_model
        self.n_enc_layers = n_enc_layers
        self.n_dec_layers = n_dec_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.max_len = max_len

        # Embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.type_emb = nn.Embedding(n_types, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        # Encoder: processes observed tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_enc_layers)

        # Decoder: generates masked tokens
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_dec_layers)

        # Output layers
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following standard transformer practice."""
        # Token embeddings
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.type_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

        # Output head
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)

    def _create_encoder_input(
        self,
        tokens: torch.Tensor,
        types: torch.Tensor,
        positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Create encoder input embeddings.

        From AION paper Eq. 4:
            e_t^(enc) = Embed_i(x_t^i) + m_i + p_t

        Parameters
        ----------
        tokens : torch.Tensor
            Token IDs, shape (B, L_obs)
        types : torch.Tensor
            Token type IDs, shape (B, L_obs)
        positions : torch.Tensor
            Position indices, shape (B, L_obs)

        Returns
        -------
        torch.Tensor
            Encoder input embeddings, shape (B, L_obs, d_model)
        """
        # Token + Type + Position embeddings
        x = self.tok_emb(tokens) + self.type_emb(types) + self.pos_emb(positions)
        x = self.drop(x)
        return x

    def _create_decoder_queries(
        self,
        types: torch.Tensor,
        positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Create decoder query embeddings.

        From AION paper Eq. 5:
            e_t^(dec) = m_i + p_t

        Note: NO token embeddings! Decoder only gets modality + position info.

        Parameters
        ----------
        types : torch.Tensor
            Token type IDs for queries, shape (B, L_tgt)
        positions : torch.Tensor
            Position indices for queries, shape (B, L_tgt)

        Returns
        -------
        torch.Tensor
            Decoder query embeddings, shape (B, L_tgt, d_model)
        """
        # Type + Position embeddings ONLY (no token embeddings!)
        q = self.type_emb(types) + self.pos_emb(positions)
        q = self.drop(q)
        return q

    def forward(
        self,
        obs_tokens: torch.Tensor,
        obs_types: torch.Tensor,
        obs_positions: torch.Tensor,
        obs_mask: torch.Tensor,
        tgt_types: torch.Tensor,
        tgt_positions: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for masked multimodal modeling.

        Parameters
        ----------
        obs_tokens : torch.Tensor
            Observed token IDs, shape (B, L_obs)
        obs_types : torch.Tensor
            Observed token types, shape (B, L_obs)
        obs_positions : torch.Tensor
            Observed token positions, shape (B, L_obs)
        obs_mask : torch.Tensor
            Observed token mask (True = valid), shape (B, L_obs)
        tgt_types : torch.Tensor
            Target token types (to predict), shape (B, L_tgt)
        tgt_positions : torch.Tensor
            Target token positions, shape (B, L_tgt)
        tgt_mask : torch.Tensor
            Target token mask (True = valid), shape (B, L_tgt)

        Returns
        -------
        torch.Tensor
            Logits over vocabulary for target positions, shape (B, L_tgt, vocab_size)
        """
        # 1. Encode observed tokens (WITH token embeddings)
        enc_input = self._create_encoder_input(obs_tokens, obs_types, obs_positions)

        # Create key padding mask (True = ignore)
        enc_key_padding_mask = ~obs_mask

        # Encode
        memory = self.encoder(enc_input, src_key_padding_mask=enc_key_padding_mask)

        # 2. Create decoder queries (WITHOUT token embeddings)
        dec_queries = self._create_decoder_queries(tgt_types, tgt_positions)

        # Create masks for decoder
        tgt_key_padding_mask = ~tgt_mask
        memory_key_padding_mask = ~obs_mask

        # Decode
        dec_output = self.decoder(
            tgt=dec_queries,
            memory=memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        # 3. Project to vocabulary
        dec_output = self.ln(dec_output)
        logits = self.head(dec_output)

        return logits

    def encode(
        self,
        obs_tokens: torch.Tensor,
        obs_types: torch.Tensor,
        obs_positions: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode observed tokens (for inference/feature extraction).

        Parameters
        ----------
        obs_tokens : torch.Tensor
            Observed token IDs, shape (B, L_obs)
        obs_types : torch.Tensor
            Observed token types, shape (B, L_obs)
        obs_positions : torch.Tensor
            Observed token positions, shape (B, L_obs)
        obs_mask : torch.Tensor
            Observed token mask, shape (B, L_obs)

        Returns
        -------
        torch.Tensor
            Encoder output (memory), shape (B, L_obs, d_model)
        """
        enc_input = self._create_encoder_input(obs_tokens, obs_types, obs_positions)
        enc_key_padding_mask = ~obs_mask
        memory = self.encoder(enc_input, src_key_padding_mask=enc_key_padding_mask)
        return memory

    def decode_step(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        tgt_types: torch.Tensor,
        tgt_positions: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single decoder step (for iterative generation like ROAR).

        Parameters
        ----------
        memory : torch.Tensor
            Encoder output, shape (B, L_obs, d_model)
        memory_mask : torch.Tensor
            Encoder mask, shape (B, L_obs)
        tgt_types : torch.Tensor
            Target types, shape (B, L_tgt)
        tgt_positions : torch.Tensor
            Target positions, shape (B, L_tgt)
        tgt_mask : torch.Tensor
            Target mask, shape (B, L_tgt)

        Returns
        -------
        torch.Tensor
            Logits, shape (B, L_tgt, vocab_size)
        """
        dec_queries = self._create_decoder_queries(tgt_types, tgt_positions)
        tgt_key_padding_mask = ~tgt_mask
        memory_key_padding_mask = ~memory_mask

        dec_output = self.decoder(
            tgt=dec_queries,
            memory=memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        dec_output = self.ln(dec_output)
        logits = self.head(dec_output)
        return logits


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
