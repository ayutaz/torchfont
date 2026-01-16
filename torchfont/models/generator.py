"""Autoregressive glyph generator model.

This module provides a Transformer-based model for generating glyph outlines
conditioned on content (character) and style (font) embeddings.

Examples:
    Training loop::

        model = GlyphGenerator(num_contents=100, num_styles=30)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        for types, coords, style_labels, content_labels in dataloader:
            loss, type_loss, coord_loss = model.compute_loss(
                content_labels, style_labels, types, coords
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    Generation::

        types, coords = model.generate(content_idx=0, style_idx=5)

"""

import torch
from torch import Tensor, nn

from torchfont.io.outline import COORD_DIM, TYPE_DIM, TYPE_TO_IDX


class GlyphGenerator(nn.Module):
    """Autoregressive Transformer model for glyph outline generation.

    The model takes content (character) and style (font) indices as conditions
    and generates a sequence of drawing commands (types) and coordinates.

    Attributes:
        num_contents: Number of content classes (characters).
        num_styles: Number of style classes (fonts).
        d_model: Dimension of the model.
        num_types: Number of command types (default 6).

    """

    def __init__(  # noqa: PLR0913
        self,
        num_contents: int,
        num_styles: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ) -> None:
        """Initialize the glyph generator.

        Args:
            num_contents: Number of content classes (characters).
            num_styles: Number of style classes (fonts).
            d_model: Dimension of the model embeddings and hidden states.
            nhead: Number of attention heads.
            num_layers: Number of transformer decoder layers.
            dim_feedforward: Dimension of feedforward network.
            dropout: Dropout probability.
            max_seq_len: Maximum sequence length for positional encoding.

        """
        super().__init__()
        self.num_contents = num_contents
        self.num_styles = num_styles
        self.d_model = d_model
        self.num_types = TYPE_DIM
        self.coord_dim = COORD_DIM
        self.max_seq_len = max_seq_len

        # Condition embeddings
        self.content_embed = nn.Embedding(num_contents, d_model)
        self.style_embed = nn.Embedding(num_styles, d_model)

        # Input embeddings for autoregressive decoding
        self.type_embed = nn.Embedding(self.num_types, d_model)
        self.coord_proj = nn.Linear(self.coord_dim, d_model)

        # Positional encoding
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # Output heads
        self.type_head = nn.Linear(d_model, self.num_types)
        self.coord_head = nn.Linear(d_model, self.coord_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        content_idx: Tensor,
        style_idx: Tensor,
        tgt_types: Tensor,
        tgt_coords: Tensor,
        tgt_padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass for training with teacher forcing.

        Args:
            content_idx: Content (character) indices. Shape: (batch,).
            style_idx: Style (font) indices. Shape: (batch,).
            tgt_types: Target command types. Shape: (batch, seq_len).
            tgt_coords: Target coordinates. Shape: (batch, seq_len, 6).
            tgt_padding_mask: Padding mask for target. Shape: (batch, seq_len).
                True indicates padding positions.

        Returns:
            Tuple of (type_logits, coord_pred):
                - type_logits: Shape (batch, seq_len, num_types)
                - coord_pred: Shape (batch, seq_len, 6)

        """
        batch_size, seq_len = tgt_types.shape
        device = tgt_types.device

        # Condition encoding: (batch, 1, d_model)
        cond = self.content_embed(content_idx) + self.style_embed(style_idx)
        memory = cond.unsqueeze(1)

        # Target embedding: type + coord + position
        positions = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        )
        tgt = (
            self.type_embed(tgt_types)
            + self.coord_proj(tgt_coords)
            + self.pos_embed(positions)
        )

        # Causal mask for autoregressive decoding
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len,
            device=device,
        )

        # Decode
        out = self.decoder(
            tgt,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )

        # Output predictions
        type_logits = self.type_head(out)
        coord_pred = self.coord_head(out)

        return type_logits, coord_pred

    def compute_loss(  # noqa: PLR0913
        self,
        content_idx: Tensor,
        style_idx: Tensor,
        types: Tensor,
        coords: Tensor,
        padding_mask: Tensor | None = None,
        coord_loss_weight: float = 1.0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute training loss with teacher forcing.

        The input sequence is shifted: input is [:-1], target is [1:].

        Args:
            content_idx: Content indices. Shape: (batch,).
            style_idx: Style indices. Shape: (batch,).
            types: Full type sequence. Shape: (batch, seq_len).
            coords: Full coord sequence. Shape: (batch, seq_len, 6).
            padding_mask: Padding mask. Shape: (batch, seq_len).
            coord_loss_weight: Weight for coordinate loss.

        Returns:
            Tuple of (total_loss, type_loss, coord_loss).

        """
        # Shift for teacher forcing
        input_types = types[:, :-1]
        input_coords = coords[:, :-1]
        target_types = types[:, 1:]
        target_coords = coords[:, 1:]

        if padding_mask is not None:
            input_padding_mask = padding_mask[:, :-1]
            target_padding_mask = padding_mask[:, 1:]
        else:
            input_padding_mask = None
            target_padding_mask = None

        # Forward pass
        type_logits, coord_pred = self.forward(
            content_idx,
            style_idx,
            input_types,
            input_coords,
            input_padding_mask,
        )

        # Type loss (cross entropy)
        type_loss = nn.functional.cross_entropy(
            type_logits.reshape(-1, self.num_types),
            target_types.reshape(-1),
            ignore_index=0,  # Ignore padding
        )

        # Coord loss (MSE, only for non-padding positions)
        if target_padding_mask is not None:
            mask = ~target_padding_mask.unsqueeze(-1)
            coord_loss = nn.functional.mse_loss(coord_pred * mask, target_coords * mask)
        else:
            coord_loss = nn.functional.mse_loss(coord_pred, target_coords)

        total_loss = type_loss + coord_loss_weight * coord_loss

        return total_loss, type_loss, coord_loss

    @torch.no_grad()
    def generate(
        self,
        content_idx: int | Tensor,
        style_idx: int | Tensor,
        max_len: int = 256,
        temperature: float = 1.0,
        device: torch.device | str | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Generate a glyph outline autoregressively.

        Args:
            content_idx: Content (character) index.
            style_idx: Style (font) index.
            max_len: Maximum sequence length to generate.
            temperature: Sampling temperature. Lower = more deterministic.
            device: Device to generate on.

        Returns:
            Tuple of (types, coords):
                - types: Generated command types. Shape: (seq_len,)
                - coords: Generated coordinates. Shape: (seq_len, 6)

        """
        self.eval()

        if device is None:
            device = next(self.parameters()).device

        # Convert to tensors if needed
        if isinstance(content_idx, int):
            content_idx = torch.tensor([content_idx], device=device)
        else:
            content_idx = content_idx.to(device)
            if content_idx.dim() == 0:
                content_idx = content_idx.unsqueeze(0)

        if isinstance(style_idx, int):
            style_idx = torch.tensor([style_idx], device=device)
        else:
            style_idx = style_idx.to(device)
            if style_idx.dim() == 0:
                style_idx = style_idx.unsqueeze(0)

        # Start with moveTo token
        types = torch.tensor([[TYPE_TO_IDX["moveTo"]]], device=device)
        coords = torch.zeros(1, 1, self.coord_dim, device=device)

        for _ in range(max_len - 1):
            # Get predictions for next token
            type_logits, coord_pred = self.forward(
                content_idx,
                style_idx,
                types,
                coords,
            )

            # Sample next type
            next_type_logits = type_logits[:, -1, :] / temperature
            next_type_probs = torch.softmax(next_type_logits, dim=-1)
            next_type = torch.multinomial(next_type_probs, 1)

            # Get next coords
            next_coords = coord_pred[:, -1:, :]

            # Append to sequence
            types = torch.cat([types, next_type], dim=1)
            coords = torch.cat([coords, next_coords], dim=1)

            # Stop if eos
            if next_type.item() == TYPE_TO_IDX["eos"]:
                break

        return types.squeeze(0), coords.squeeze(0)
