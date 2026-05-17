"""Inspect a PyTorch .pth file and print parameter/input-output details.

Usage:
	python chcek_pth.py /path/to/model.pth

The script loads a checkpoint or state_dict, prints each tensor's shape,
dtype, and number of elements, and infers common layer input/output sizes
from parameter names and tensor shapes.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import pickle


def _load_checkpoint(path: str | Path, *, unsafe: bool) -> Any:
	if unsafe:
		return torch.load(path, map_location="cpu", weights_only=False)
	try:
		return torch.load(path, map_location="cpu")
	except pickle.UnpicklingError:
		try:
			from dotmap import DotMap
		except Exception:
			raise
		with torch.serialization.safe_globals([DotMap]):
			return torch.load(path, map_location="cpu")


def _extract_state_dict(obj: Any) -> dict[str, torch.Tensor]:
	if isinstance(obj, Mapping):
		if obj and all(torch.is_tensor(v) for v in obj.values()):
			return dict(obj)

		for key in (
			"state_dict",
			"model_state_dict",
			"model",
			"net",
			"generator",
			"ema_state_dict",
		):
			val = obj.get(key)
			if isinstance(val, Mapping) and val and all(torch.is_tensor(v) for v in val.values()):
				return dict(val)

	raise ValueError("Could not find a tensor state_dict in the checkpoint.")


def _format_shape(tensor: torch.Tensor) -> str:
	return "x".join(str(dim) for dim in tensor.shape) if tensor.ndim else "scalar"


def _layer_io_from_tensor(name: str, tensor: torch.Tensor) -> str | None:
	lname = name.lower()
	shape = tuple(tensor.shape)

	if tensor.ndim == 2 and "weight" in lname:
		out_features, in_features = shape
		return f"Linear-like: input={in_features}, output={out_features}"

	if tensor.ndim in (3, 4, 5) and "weight" in lname:
		out_channels = shape[0]
		in_channels = shape[1]
		kernel_size = shape[2:]
		return (
			"Conv-like: "
			f"input_channels={in_channels}, output_channels={out_channels}, kernel_size={kernel_size}"
		)

	if tensor.ndim == 2 and ("embed" in lname or "embedding" in lname):
		num_embeddings, embedding_dim = shape
		return f"Embedding-like: num_embeddings={num_embeddings}, embedding_dim={embedding_dim}"

	if tensor.ndim == 1 and any(k in lname for k in ("bn", "norm", "layernorm", "ln")):
		features = shape[0]
		return f"Norm-like: input_features={features}, output_features={features}"

	return None


def inspect_pth(path: str | Path, *, unsafe: bool) -> None:
	checkpoint = _load_checkpoint(path, unsafe=unsafe)
	state_dict = _extract_state_dict(checkpoint)

	total_params = 0
	print(f"Loaded: {path}")
	print(f"Tensors found: {len(state_dict)}")
	print("-" * 100)

	for name, tensor in state_dict.items():
		numel = tensor.numel()
		total_params += numel
		io_info = _layer_io_from_tensor(name, tensor)
		print(f"{name}")
		print(f"  shape: {_format_shape(tensor)}")
		print(f"  dtype: {tensor.dtype}")
		print(f"  elements: {numel}")
		if io_info:
			print(f"  inferred: {io_info}")
		print()

	print("-" * 100)
	print(f"Total tensor elements: {total_params}")


def main() -> None:
	parser = argparse.ArgumentParser(description="Inspect a .pth file and print parameter sizes.")
	parser.add_argument("pth_file", type=str, help="Path to the .pth file")
	parser.add_argument(
		"--unsafe",
		action="store_true",
		help="Load with weights_only=False (unsafe unless file is trusted)",
	)
	args = parser.parse_args()

	inspect_pth(args.pth_file, unsafe=args.unsafe)


if __name__ == "__main__":
	main()
