"""Generate glyphs using a trained GlyphGenerator model.

This script generates glyph outlines and optionally saves them as SVG files.

Usage:
    # Generate a single glyph
    uv run python examples/generate_glyph.py \\
        --checkpoint checkpoints/latest.pt --char A --style 0

    # Generate multiple characters
    uv run python examples/generate_glyph.py \\
        --checkpoint checkpoints/latest.pt --char ABC --style 0

    # List available styles
    uv run python examples/generate_glyph.py \\
        --checkpoint checkpoints/latest.pt --list-styles

"""

import argparse
from pathlib import Path

import torch

from torchfont.io.outline import TYPE_TO_IDX
from torchfont.models import GlyphGenerator

# Maximum number of characters to display when listing
MAX_CHARS_TO_DISPLAY = 100


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate glyphs with GlyphGenerator")

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--char",
        type=str,
        help="Character(s) to generate",
    )
    parser.add_argument(
        "--style",
        type=int,
        default=0,
        help="Style index (default: 0)",
    )
    parser.add_argument(
        "--style-name",
        type=str,
        help="Style name (alternative to --style)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output directory for SVG files (default: output)",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=256,
        help="Maximum sequence length (default: 256)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0, lower = more deterministic)",
    )
    parser.add_argument(
        "--list-styles",
        action="store_true",
        help="List available styles and exit",
    )
    parser.add_argument(
        "--list-chars",
        action="store_true",
        help="List available characters (first 100) and exit",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available)",
    )

    return parser.parse_args()


def outline_to_svg(
    types: torch.Tensor,
    coords: torch.Tensor,
    width: int = 1000,
    height: int = 1000,
    scale: float = 1.0,
) -> str:
    """Convert glyph outline to SVG string.

    Args:
        types: Command types tensor. Shape: (seq_len,).
        coords: Coordinates tensor. Shape: (seq_len, 6).
        width: SVG width.
        height: SVG height.
        scale: Scale factor for coordinates.

    Returns:
        SVG string.

    """
    # Create reverse mapping
    idx_to_type = {v: k for k, v in TYPE_TO_IDX.items()}

    path_data = []
    for i in range(len(types)):
        cmd_type = idx_to_type.get(types[i].item(), "pad")
        c = coords[i] * scale

        if cmd_type == "moveTo":
            path_data.append(f"M {c[0]:.2f} {c[1]:.2f}")
        elif cmd_type == "lineTo":
            path_data.append(f"L {c[0]:.2f} {c[1]:.2f}")
        elif cmd_type == "curveTo":
            # Cubic bezier: control point 1, control point 2, end point
            path_data.append(
                f"C {c[0]:.2f} {c[1]:.2f}, "
                f"{c[2]:.2f} {c[3]:.2f}, {c[4]:.2f} {c[5]:.2f}",
            )
        elif cmd_type == "closePath":
            path_data.append("Z")
        elif cmd_type == "eos":
            break

    path_str = " ".join(path_data)

    # Build SVG string (noqa for E501 - SVG template readability)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <g transform="translate(100, 800) scale(1, -1)">
    <path d="{path_str}" fill="black" stroke="none"/>
  </g>
</svg>"""  # noqa: E501


def main() -> None:  # noqa: C901
    """Main generation function."""
    args = parse_args()

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    # Get metadata from checkpoint
    content_classes = checkpoint.get("content_classes", [])
    style_classes = checkpoint.get("style_classes", [])
    num_contents = checkpoint.get("num_contents", len(content_classes))
    num_styles = checkpoint.get("num_styles", len(style_classes))

    # Handle list commands
    if args.list_styles:
        print("Available styles:")
        for i, style in enumerate(style_classes):
            print(f"  {i}: {style}")
        return

    if args.list_chars:
        print(f"Available characters (first {MAX_CHARS_TO_DISPLAY}):")
        for i, char in enumerate(content_classes[:MAX_CHARS_TO_DISPLAY]):
            print(f"  {i}: {char!r}")
        if len(content_classes) > MAX_CHARS_TO_DISPLAY:
            remaining = len(content_classes) - MAX_CHARS_TO_DISPLAY
            print(f"  ... and {remaining} more")
        return

    if args.char is None:
        print("Error: --char is required (unless using --list-styles or --list-chars)")
        return

    # Setup device
    device = torch.device(args.device)

    # Get model args from checkpoint
    saved_args = checkpoint.get("args", {})

    # Create model with same architecture
    model = GlyphGenerator(
        num_contents=num_contents,
        num_styles=num_styles,
        d_model=saved_args.get("d_model", 256),
        nhead=saved_args.get("nhead", 8),
        num_layers=saved_args.get("num_layers", 6),
        max_seq_len=saved_args.get("max_seq_len", 256),
    ).to(device)

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Model loaded successfully")

    # Resolve style
    style_idx = args.style
    if args.style_name is not None:
        if args.style_name in style_classes:
            style_idx = style_classes.index(args.style_name)
        else:
            print(f"Error: Style '{args.style_name}' not found")
            print("Available styles:")
            for i, style in enumerate(style_classes[:10]):
                print(f"  {i}: {style}")
            return

    # Create content class to index mapping
    content_to_idx = {char: i for i, char in enumerate(content_classes)}

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate for each character
    for char in args.char:
        if char not in content_to_idx:
            print(f"Warning: Character '{char}' not in training data, skipping")
            continue

        content_idx = content_to_idx[char]

        print(f"Generating '{char}' (content={content_idx}, style={style_idx})...")

        # Generate
        types, coords = model.generate(
            content_idx=content_idx,
            style_idx=style_idx,
            max_len=args.max_len,
            temperature=args.temperature,
            device=device,
        )

        print(f"  Generated sequence length: {len(types)}")

        # Convert to SVG
        svg = outline_to_svg(types, coords, scale=800)

        # Save SVG
        style_name = (
            style_classes[style_idx]
            if style_idx < len(style_classes)
            else f"style_{style_idx}"
        )
        safe_style_name = style_name.replace(" ", "_").replace("/", "_")
        output_path = args.output_dir / f"{char}_{safe_style_name}.svg"
        output_path.write_text(svg, encoding="utf-8")
        print(f"  Saved: {output_path}")


if __name__ == "__main__":
    main()
