"""Gradio demo for GlyphGenerator model.

This script provides an interactive web interface for generating glyphs.

Usage:
    uv run python examples/demo_generator.py --checkpoint checkpoints/latest.pt

"""

import argparse
from pathlib import Path

import gradio as gr
import torch

from torchfont.io.outline import TYPE_TO_IDX
from torchfont.models import GlyphGenerator


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Gradio demo for GlyphGenerator")

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/latest.pt"),
        help="Path to model checkpoint (default: checkpoints/latest.pt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public link",
    )

    return parser.parse_args()


def outline_to_svg(
    types: torch.Tensor,
    coords: torch.Tensor,
    width: int = 500,
    height: int = 500,
    scale: float = 400.0,
) -> str:
    """Convert glyph outline to SVG string."""
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
            path_data.append(
                f"C {c[0]:.2f} {c[1]:.2f}, "
                f"{c[2]:.2f} {c[3]:.2f}, {c[4]:.2f} {c[5]:.2f}",
            )
        elif cmd_type == "closePath":
            path_data.append("Z")
        elif cmd_type == "eos":
            break

    path_str = " ".join(path_data)

    # SVG with white background and centered glyph
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"
     viewBox="0 0 {width} {height}" style="background-color: white;">
  <g transform="translate(50, 400) scale(1, -1)">
    <path d="{path_str}" fill="black" stroke="none"/>
  </g>
</svg>"""



def create_demo(
    model: GlyphGenerator,
    content_classes: list[str],
    style_classes: list[str],
    device: torch.device,
) -> gr.Blocks:
    """Create Gradio demo interface."""
    # Create content class to index mapping
    content_to_idx = {char: i for i, char in enumerate(content_classes)}

    def generate_glyph(
        char: str,
        style_name: str,
        temperature: float,
        max_len: int,
    ) -> str:
        """Generate a glyph and return SVG."""
        if not char:
            return "<p>Please enter a character</p>"

        # Take only the first character
        char = char[0]

        if char not in content_to_idx:
            available = "".join(
                c for c in content_classes[:50] if c.isprintable() and not c.isspace()
            )
            return (
                f"<p>Character '{char}' not in training data.</p>"
                f"<p>Available: {available}...</p>"
            )

        content_idx = content_to_idx[char]
        style_idx = style_classes.index(style_name)

        # Generate
        types, coords = model.generate(
            content_idx=content_idx,
            style_idx=style_idx,
            max_len=max_len,
            temperature=temperature,
            device=device,
        )

        # Convert to SVG
        return outline_to_svg(types.cpu(), coords.cpu())


    def generate_comparison(
        char: str,
        temperature: float,
        max_len: int,
    ) -> str:
        """Generate the same character in multiple styles."""
        if not char:
            return "<p>Please enter a character</p>"

        char = char[0]

        if char not in content_to_idx:
            return f"<p>Character '{char}' not in training data.</p>"

        content_idx = content_to_idx[char]

        # Generate for first 6 styles
        svgs = []
        for style_idx in range(min(6, len(style_classes))):
            types, coords = model.generate(
                content_idx=content_idx,
                style_idx=style_idx,
                max_len=max_len,
                temperature=temperature,
                device=device,
            )
            svg = outline_to_svg(
                types.cpu(), coords.cpu(), width=200, height=200, scale=150,
            )
            style_name = style_classes[style_idx]
            div_style = "display: inline-block; text-align: center; margin: 10px;"
            svgs.append(
                f'<div style="{div_style}">'
                f"{svg}<br><small>{style_name}</small></div>",
            )

        return "".join(svgs)

    # Build the interface
    with gr.Blocks(title="Glyph Generator Demo") as demo:
        gr.Markdown("# Glyph Generator Demo")
        gr.Markdown("Generate font glyphs using a trained GlyphGenerator model.")

        with gr.Tab("Single Generation"):
            with gr.Row():
                with gr.Column():
                    char_input = gr.Textbox(
                        label="Character",
                        placeholder="Enter a character (e.g., A)",
                        max_lines=1,
                    )
                    style_dropdown = gr.Dropdown(
                        choices=style_classes,
                        value=style_classes[0] if style_classes else None,
                        label="Font Style",
                    )
                    temperature_slider = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Temperature (lower = more deterministic)",
                    )
                    max_len_slider = gr.Slider(
                        minimum=32,
                        maximum=512,
                        value=256,
                        step=32,
                        label="Max Sequence Length",
                    )
                    generate_btn = gr.Button("Generate", variant="primary")

                with gr.Column():
                    output_html = gr.HTML(label="Generated Glyph")

            generate_btn.click(
                fn=generate_glyph,
                inputs=[char_input, style_dropdown, temperature_slider, max_len_slider],
                outputs=output_html,
            )

        with gr.Tab("Style Comparison"):
            gr.Markdown("Generate the same character in multiple styles.")
            with gr.Row():
                with gr.Column():
                    comp_char_input = gr.Textbox(
                        label="Character",
                        placeholder="Enter a character (e.g., A)",
                        max_lines=1,
                    )
                    comp_temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Temperature",
                    )
                    comp_max_len = gr.Slider(
                        minimum=32,
                        maximum=512,
                        value=256,
                        step=32,
                        label="Max Sequence Length",
                    )
                    compare_btn = gr.Button("Compare Styles", variant="primary")

                with gr.Column():
                    comparison_html = gr.HTML(label="Style Comparison")

            compare_btn.click(
                fn=generate_comparison,
                inputs=[comp_char_input, comp_temperature, comp_max_len],
                outputs=comparison_html,
            )

        gr.Markdown("---")
        gr.Markdown(
            f"**Model Info:** {len(content_classes)} characters, "
            f"{len(style_classes)} styles",
        )

    return demo


def main() -> None:
    """Main function."""
    args = parse_args()

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(
        args.checkpoint,
        map_location=args.device,
        weights_only=False,
    )

    # Get metadata
    content_classes = checkpoint.get("content_classes", [])
    style_classes = checkpoint.get("style_classes", [])
    num_contents = checkpoint.get("num_contents", len(content_classes))
    num_styles = checkpoint.get("num_styles", len(style_classes))
    saved_args = checkpoint.get("args", {})

    # Setup device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create model
    model = GlyphGenerator(
        num_contents=num_contents,
        num_styles=num_styles,
        d_model=saved_args.get("d_model", 256),
        nhead=saved_args.get("nhead", 8),
        num_layers=saved_args.get("num_layers", 6),
        max_seq_len=saved_args.get("max_seq_len", 256),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Model loaded successfully")

    # Create and launch demo
    demo = create_demo(model, content_classes, style_classes, device)
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
