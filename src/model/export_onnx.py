"""
export_onnx.py - Export trained MLP head to ONNX for fast inference

Owner: Zihao
Week:  4-5

Exports the MLP classification head to ONNX format.
Note: only the MLP head is exported - CLIP feature extraction
uses PyTorch at inference time (OpenCLIP does not need ONNX).

Usage:
    python src/model/export_onnx.py \
        --checkpoint models/checkpoints/best.pt \
        --output models/onnx/mlp_head.onnx
"""

import argparse
import torch
from pathlib import Path


def export(checkpoint_path: Path, output_path: Path) -> None:
    """
    Load a trained checkpoint and export the MLP head to ONNX.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        output_path:     Path to write .onnx file
    """
    # TODO: implement torch.onnx.export
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output",     type=str, required=True)
    args = parser.parse_args()
    export(Path(args.checkpoint), Path(args.output))
