"""Training script for GlyphGenerator model.

This script trains a GlyphGenerator model on font data.
It supports both local fonts (FontFolder) and Google Fonts.

Usage:
    # Train on local fonts
    uv run python examples/train_generator.py --data-dir tests/fonts

    # Train on Google Fonts (will download if not present)
    uv run python examples/train_generator.py --google-fonts --download

    # Resume training from checkpoint
    uv run python examples/train_generator.py \\
        --data-dir tests/fonts --resume checkpoints/latest.pt

"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchfont.datasets import FontFolder, GoogleFonts
from torchfont.models import GlyphGenerator
from torchfont.transforms import Compose, LimitSequenceLength


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train GlyphGenerator model")

    # Data arguments
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--data-dir",
        type=Path,
        help="Path to local font directory",
    )
    data_group.add_argument(
        "--google-fonts",
        action="store_true",
        help="Use Google Fonts dataset",
    )

    parser.add_argument(
        "--google-fonts-root",
        type=Path,
        default=Path("data/google/fonts"),
        help="Root directory for Google Fonts (default: data/google/fonts)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download Google Fonts if not present",
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=256,
        help="Maximum sequence length (default: 256)",
    )

    # Model arguments
    parser.add_argument(
        "--d-model",
        type=int,
        default=256,
        help="Model dimension (default: 256)",
    )
    parser.add_argument(
        "--nhead",
        type=int,
        default=8,
        help="Number of attention heads (default: 8)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=6,
        help="Number of transformer layers (default: 6)",
    )

    # Other arguments
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loader workers (default: 4)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory for saving checkpoints (default: checkpoints)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Save checkpoint every N epochs (default: 5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (default: cuda if available)",
    )

    return parser.parse_args()


def collate_fn(
    batch: Sequence[tuple[Tensor, Tensor, int, int]],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Collate function for DataLoader.

    Returns:
        Tuple of (types, coords, style_labels, content_labels, padding_mask).

    """
    types_list = [types for types, _, _, _ in batch]
    coords_list = [coords for _, coords, _, _ in batch]
    style_label_list = [style for _, _, style, _ in batch]
    content_label_list = [content for _, _, _, content in batch]

    types_tensor = pad_sequence(types_list, batch_first=True, padding_value=0)
    coords_tensor = pad_sequence(coords_list, batch_first=True, padding_value=0.0)

    # Create padding mask (True for padding positions)
    max_len = types_tensor.size(1)
    padding_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, types in enumerate(types_list):
        padding_mask[i, len(types) :] = True

    style_label_tensor = torch.as_tensor(style_label_list, dtype=torch.long)
    content_label_tensor = torch.as_tensor(content_label_list, dtype=torch.long)

    return (
        types_tensor,
        coords_tensor,
        style_label_tensor,
        content_label_tensor,
        padding_mask,
    )


def main() -> None:  # noqa: PLR0915
    """Main training function."""
    args = parse_args()

    # Create checkpoint directory
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Setup device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create transform
    transform = Compose([LimitSequenceLength(max_len=args.max_seq_len)])

    # Create dataset
    if args.google_fonts:
        print("Loading Google Fonts dataset...")
        dataset = GoogleFonts(
            root=args.google_fonts_root,
            ref="main",
            transform=transform,
            download=args.download,
        )
    else:
        print(f"Loading fonts from {args.data_dir}...")
        dataset = FontFolder(
            root=args.data_dir,
            transform=transform,
        )

    print(f"Dataset size: {len(dataset)}")
    print(f"Content classes: {len(dataset.content_classes)}")
    print(f"Style classes: {len(dataset.style_classes)}")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    # Create model
    model = GlyphGenerator(
        num_contents=len(dataset.content_classes),
        num_styles=len(dataset.style_classes),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
    ).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume is not None:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        total_type_loss = 0.0
        total_coord_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            leave=True,
        )

        for batch in progress_bar:
            types, coords, style_labels, content_labels, padding_mask = batch

            # Move to device
            types = types.to(device)
            coords = coords.to(device)
            style_labels = style_labels.to(device)
            content_labels = content_labels.to(device)
            padding_mask = padding_mask.to(device)

            # Forward pass
            loss, type_loss, coord_loss = model.compute_loss(
                content_labels,
                style_labels,
                types,
                coords,
                padding_mask,
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update statistics
            total_loss += loss.item()
            total_type_loss += type_loss.item()
            total_coord_loss += coord_loss.item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "type": f"{type_loss.item():.4f}",
                    "coord": f"{coord_loss.item():.4f}",
                },
            )

        # Print epoch summary
        avg_loss = total_loss / num_batches
        avg_type_loss = total_type_loss / num_batches
        avg_coord_loss = total_coord_loss / num_batches
        print(
            f"Epoch {epoch + 1}: "
            f"loss={avg_loss:.4f}, "
            f"type_loss={avg_type_loss:.4f}, "
            f"coord_loss={avg_coord_loss:.4f}",
        )

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = args.checkpoint_dir / f"epoch_{epoch + 1:04d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "args": vars(args),
                    "num_contents": len(dataset.content_classes),
                    "num_styles": len(dataset.style_classes),
                    "content_classes": dataset.content_classes,
                    "style_classes": dataset.style_classes,
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint: {checkpoint_path}")

            # Also save as latest
            latest_path = args.checkpoint_dir / "latest.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "args": vars(args),
                    "num_contents": len(dataset.content_classes),
                    "num_styles": len(dataset.style_classes),
                    "content_classes": dataset.content_classes,
                    "style_classes": dataset.style_classes,
                },
                latest_path,
            )

    print("Training complete!")


if __name__ == "__main__":
    main()
