"""Tests for the GlyphGenerator model."""

import pytest
import torch

from torchfont.models import GlyphGenerator


class TestGlyphGeneratorInit:
    """Tests for GlyphGenerator initialization."""

    def test_init_default_params(self) -> None:
        """Test initialization with default parameters."""
        model = GlyphGenerator(num_contents=100, num_styles=30)

        assert model.num_contents == 100
        assert model.num_styles == 30
        assert model.d_model == 256
        assert model.num_types == 6
        assert model.coord_dim == 6

    def test_init_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        model = GlyphGenerator(
            num_contents=50,
            num_styles=10,
            d_model=128,
            nhead=4,
            num_layers=3,
            dim_feedforward=512,
            dropout=0.2,
            max_seq_len=256,
        )

        assert model.num_contents == 50
        assert model.num_styles == 10
        assert model.d_model == 128
        assert model.max_seq_len == 256


class TestGlyphGeneratorForward:
    """Tests for GlyphGenerator forward pass."""

    @pytest.fixture
    def model(self) -> GlyphGenerator:
        """Create a small model for testing."""
        return GlyphGenerator(
            num_contents=100,
            num_styles=30,
            d_model=64,
            nhead=4,
            num_layers=2,
        )

    def test_forward_shape(self, model: GlyphGenerator) -> None:
        """Test that forward pass produces correct output shapes."""
        batch_size = 4
        seq_len = 16

        content_idx = torch.randint(0, 100, (batch_size,))
        style_idx = torch.randint(0, 30, (batch_size,))
        tgt_types = torch.randint(0, 6, (batch_size, seq_len))
        tgt_coords = torch.randn(batch_size, seq_len, 6)

        type_logits, coord_pred = model(content_idx, style_idx, tgt_types, tgt_coords)

        assert type_logits.shape == (batch_size, seq_len, 6)
        assert coord_pred.shape == (batch_size, seq_len, 6)

    def test_forward_with_padding_mask(self, model: GlyphGenerator) -> None:
        """Test forward pass with padding mask."""
        batch_size = 4
        seq_len = 16

        content_idx = torch.randint(0, 100, (batch_size,))
        style_idx = torch.randint(0, 30, (batch_size,))
        tgt_types = torch.randint(0, 6, (batch_size, seq_len))
        tgt_coords = torch.randn(batch_size, seq_len, 6)
        padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        padding_mask[:, -4:] = True  # Last 4 positions are padding

        type_logits, coord_pred = model(
            content_idx,
            style_idx,
            tgt_types,
            tgt_coords,
            padding_mask,
        )

        assert type_logits.shape == (batch_size, seq_len, 6)
        assert coord_pred.shape == (batch_size, seq_len, 6)


class TestGlyphGeneratorComputeLoss:
    """Tests for GlyphGenerator loss computation."""

    @pytest.fixture
    def model(self) -> GlyphGenerator:
        """Create a small model for testing."""
        return GlyphGenerator(
            num_contents=100,
            num_styles=30,
            d_model=64,
            nhead=4,
            num_layers=2,
        )

    def test_compute_loss_returns_losses(self, model: GlyphGenerator) -> None:
        """Test that compute_loss returns valid loss values."""
        batch_size = 4
        seq_len = 16

        content_idx = torch.randint(0, 100, (batch_size,))
        style_idx = torch.randint(0, 30, (batch_size,))
        types = torch.randint(1, 6, (batch_size, seq_len))  # Avoid 0 (padding)
        coords = torch.randn(batch_size, seq_len, 6)

        total_loss, type_loss, coord_loss = model.compute_loss(
            content_idx,
            style_idx,
            types,
            coords,
        )

        assert total_loss.ndim == 0  # Scalar
        assert type_loss.ndim == 0
        assert coord_loss.ndim == 0
        assert total_loss.item() >= 0
        assert type_loss.item() >= 0
        assert coord_loss.item() >= 0

    def test_compute_loss_with_padding_mask(self, model: GlyphGenerator) -> None:
        """Test compute_loss with padding mask."""
        batch_size = 4
        seq_len = 16

        content_idx = torch.randint(0, 100, (batch_size,))
        style_idx = torch.randint(0, 30, (batch_size,))
        types = torch.randint(1, 6, (batch_size, seq_len))
        coords = torch.randn(batch_size, seq_len, 6)
        padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        padding_mask[:, -4:] = True

        total_loss, _type_loss, _coord_loss = model.compute_loss(
            content_idx,
            style_idx,
            types,
            coords,
            padding_mask,
        )

        assert total_loss.ndim == 0
        assert not torch.isnan(total_loss)

    def test_compute_loss_backward(self, model: GlyphGenerator) -> None:
        """Test that loss can be backpropagated."""
        batch_size = 4
        seq_len = 16

        content_idx = torch.randint(0, 100, (batch_size,))
        style_idx = torch.randint(0, 30, (batch_size,))
        types = torch.randint(1, 6, (batch_size, seq_len))
        coords = torch.randn(batch_size, seq_len, 6)

        total_loss, _, _ = model.compute_loss(content_idx, style_idx, types, coords)
        total_loss.backward()

        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestGlyphGeneratorGenerate:
    """Tests for GlyphGenerator generation."""

    @pytest.fixture
    def model(self) -> GlyphGenerator:
        """Create a small model for testing."""
        return GlyphGenerator(
            num_contents=100,
            num_styles=30,
            d_model=64,
            nhead=4,
            num_layers=2,
        )

    def test_generate_shape(self, model: GlyphGenerator) -> None:
        """Test that generate produces valid output shapes."""
        types, coords = model.generate(content_idx=0, style_idx=5, max_len=32)

        assert types.ndim == 1
        assert coords.ndim == 2
        assert types.shape[0] == coords.shape[0]
        assert coords.shape[1] == 6
        assert types.shape[0] <= 32

    def test_generate_with_tensor_input(self, model: GlyphGenerator) -> None:
        """Test generate with tensor inputs."""
        content_idx = torch.tensor(10)
        style_idx = torch.tensor(3)

        types, coords = model.generate(
            content_idx=content_idx,
            style_idx=style_idx,
            max_len=16,
        )

        assert types.ndim == 1
        assert coords.ndim == 2

    def test_generate_starts_with_moveto(self, model: GlyphGenerator) -> None:
        """Test that generated sequence starts with moveTo command."""
        types, _ = model.generate(content_idx=0, style_idx=0, max_len=16)

        assert types[0].item() == 1  # moveTo = 1

    def test_generate_deterministic_with_low_temperature(
        self,
        model: GlyphGenerator,
    ) -> None:
        """Test that low temperature produces more deterministic outputs."""
        torch.manual_seed(42)
        types1, _ = model.generate(content_idx=0, style_idx=0, temperature=0.01)

        torch.manual_seed(42)
        types2, _ = model.generate(content_idx=0, style_idx=0, temperature=0.01)

        assert torch.equal(types1, types2)
