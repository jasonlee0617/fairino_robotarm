#!/usr/bin/env python3
"""Export a YOLO-OBB .pt model as named ONNX and fixed-shape TensorRT files."""

import argparse
import os
import shutil
from pathlib import Path

from ultralytics import YOLO


# Edit these defaults for repeated exports; command-line options override them.
DEFAULT_MODEL = "$HOME/my-workspace/fairino_robotarm/src/visual_perception/models/yolo-obb-640.pt"
DEFAULT_OUTPUT_DIR = "$HOME/my-workspace/fairino_robotarm/src/visual_perception/models"
DEFAULT_IMGSZ = 640
DEFAULT_NAME = "yolo-obb-640"  # None creates yolo_obb-{imgsz}.onnx and yolo_obb-{imgsz}.engine.
DEFAULT_DEVICE = 0
DEFAULT_WORKSPACE = None


def output_name(name: str | None, suffix: str, imgsz: int) -> str:
    filename = name or f"yolo_obb-{imgsz}"
    return filename if filename.endswith(suffix) else f"{filename}{suffix}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Input .pt model path.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for exported files.")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="Fixed square input size.")
    parser.add_argument("--name", default=DEFAULT_NAME, help="Shared output basename without extension.")
    parser.add_argument("--onnx-name", help="ONNX filename; overrides --name for ONNX.")
    parser.add_argument("--engine-name", help="TensorRT filename; overrides --name for engine.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="CUDA device for TensorRT export.")
    parser.add_argument("--workspace", type=float, default=DEFAULT_WORKSPACE, help="TensorRT workspace in GiB.")
    return parser.parse_args()


def move_export(source: Path, destination: Path) -> None:
    if not source.is_file() or source.stat().st_size == 0:
        raise RuntimeError(f"Export did not create a valid file: {source}")
    if source.resolve() != destination.resolve():
        shutil.move(str(source), str(destination))
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"Missing final export: {destination}")


def main() -> None:
    args = parse_args()
    model_path = Path(os.path.expandvars(args.model)).expanduser().resolve()
    output_dir = Path(os.path.expandvars(args.output_dir)).expanduser().resolve()
    if model_path.suffix != ".pt" or not model_path.is_file():
        raise FileNotFoundError(f"Input model must be an existing .pt file: {model_path}")
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / output_name(args.onnx_name or args.name, ".onnx", args.imgsz)
    engine_path = output_dir / output_name(args.engine_name or args.name, ".engine", args.imgsz)
    if onnx_path == engine_path:
        raise ValueError("ONNX and engine output names must differ")

    model = YOLO(str(model_path))
    engine_options = {"imgsz": args.imgsz, "dynamic": False, "simplify": True, "half": True, "device": args.device}
    if args.workspace is not None:
        engine_options["workspace"] = args.workspace
    engine_source = Path(model.export(format="engine", **engine_options))
    onnx_source = model_path.with_suffix(".onnx")
    move_export(onnx_source, onnx_path)
    move_export(engine_source, engine_path)

    print(f"ONNX:   {onnx_path}")
    print(f"Engine: {engine_path}")
    print(f"Deployment requires imgsz={args.imgsz} for {engine_path.name}")


if __name__ == "__main__":
    main()
