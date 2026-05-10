#!/usr/bin/env python3
"""Live IQA with multiple PyIQA metrics on frames from the Android camera client."""

import argparse
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from camera_client import AWB_MODES, AWB_NAMES, EXPOSURE_NS, FOCUS_DIOPTERS, ISO_VALUES, CameraClient

try:
	import pyiqa
except ImportError:
	pyiqa = None


def adb_connect():
	import subprocess

	try:
		subprocess.run(["adb", "forward", "tcp:5555", "tcp:5555"], check=True)
		subprocess.run(["adb", "forward", "tcp:5556", "tcp:5556"], check=True)
		print("[+] ADB port forwarding set up successfully")
	except subprocess.CalledProcessError as exc:
		print(f"[!] ADB port forwarding failed: {exc}")
		sys.exit(1)


def _resize_short_side(image: Image.Image, short_side: int) -> Image.Image:
	width, height = image.size
	if short_side <= 0 or min(width, height) == short_side:
		return image
	scale = short_side / min(width, height)
	new_w = int(round(width * scale))
	new_h = int(round(height * scale))
	return image.resize((new_w, new_h), resample=Image.BICUBIC)


def preprocess_frame(frame_bgr: np.ndarray, short_side: int, center_crop: int, device: str) -> torch.Tensor:
	frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
	image = Image.fromarray(frame_rgb)
	image = _resize_short_side(image, short_side)
	if center_crop > 0:
		image = transforms.CenterCrop((center_crop, center_crop))(image)
	batch = transforms.ToTensor()(image).unsqueeze(0).to(device)
	return batch


def create_metrics(metric_names: list[str], device: str):
	metrics = {}
	for name in metric_names:
		metrics[name] = pyiqa.create_metric(name, device=device)
	return metrics


def run_metrics(metrics: dict, batch: torch.Tensor) -> dict:
	scores = {}
	for name, metric in metrics.items():
		try:
			value = metric(batch)
			scores[name] = float(value.item())
		except Exception as exc:
			scores[name] = None
			print(f"[!] {name} failed: {exc}")
	return scores


def main():
	parser = argparse.ArgumentParser(description="Live IQA with PyIQA metrics on Android camera frames")
	parser.add_argument(
		"--metrics",
		default="niqe,brisque,piqe,maniqa",
		help="Comma-separated PyIQA metric names (no-reference recommended)",
	)
	parser.add_argument("--interval_ms", type=int, default=200, help="Inference interval in ms")
	parser.add_argument("--short_side", type=int, default=640, help="Resize shorter side (0 keeps original)")
	parser.add_argument("--center_crop", type=int, default=0, help="Center crop size (0 disables)")
	args = parser.parse_args()

	if pyiqa is None:
		print("[!] pyiqa is not installed. Install it with: pip install pyiqa")
		sys.exit(1)

	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Using device: {device}")

	metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()]
	metrics = create_metrics(metric_names, device)

	adb_connect()
	client = CameraClient()
	client.connect()

	last_scores = {}
	last_infer = 0.0
	last_inf_time = 0.0

	cv2.namedWindow("Live PyIQA", cv2.WINDOW_NORMAL)

	exp_idx = 3
	iso_idx = 0
	focus_idx = 0
	wb_idx = 0
	auto_exp = True
	auto_af = True
	torch_control = False

	try:
		while client.connected:
			frame = client.get_frame()

			if frame is None:
				key = cv2.waitKey(5) & 0xFF
			else:
				now = time.time()
				if (now - last_infer) * 1000.0 >= args.interval_ms:
					start = time.time()
					batch = preprocess_frame(frame, args.short_side, args.center_crop, device)
					last_scores = run_metrics(metrics, batch)
					last_inf_time = time.time() - start
					last_infer = time.time()

				line_y = 30
				for name in metric_names:
					value = last_scores.get(name)
					label = f"{name}: {value:.4f}" if value is not None else f"{name}: error"
					cv2.putText(
						frame,
						label,
						(10, line_y),
						cv2.FONT_HERSHEY_SIMPLEX,
						0.7,
						(0, 255, 0),
						2,
					)
					line_y += 26

				cv2.putText(
					frame,
					f"PyIQA Inf_Time: {last_inf_time:.4f}",
					(10, line_y + 10),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.6,
					(0, 255, 0),
					2,
				)

				cv2.imshow("Live PyIQA", frame)
				key = cv2.waitKey(1) & 0xFF

			if key == ord("q"):
				break
			elif key == ord("e"):
				exp_idx = (exp_idx + 1) % len(EXPOSURE_NS)
				client.send_command("set_exposure", EXPOSURE_NS[exp_idx])
				print(f"  Exposure -> {EXPOSURE_NS[exp_idx] / 1e6:.1f} ms")
			elif key == ord("r"):
				exp_idx = (exp_idx - 1 + len(EXPOSURE_NS)) % len(EXPOSURE_NS)
				client.send_command("set_exposure", EXPOSURE_NS[exp_idx])
				print(f"  Exposure -> {EXPOSURE_NS[exp_idx] / 1e6:.1f} ms")
			elif key == ord("i"):
				iso_idx = (iso_idx + 1) % len(ISO_VALUES)
				client.send_command("set_iso", ISO_VALUES[iso_idx])
				print(f"  ISO -> {ISO_VALUES[iso_idx]}")
			elif key == ord("o"):
				iso_idx = (iso_idx - 1 + len(ISO_VALUES)) % len(ISO_VALUES)
				client.send_command("set_iso", ISO_VALUES[iso_idx])
				print(f"  ISO -> {ISO_VALUES[iso_idx]}")
			elif key == ord("f"):
				focus_idx = (focus_idx + 1) % len(FOCUS_DIOPTERS)
				client.send_command("set_focus", FOCUS_DIOPTERS[focus_idx])
				print(f"  Focus -> {FOCUS_DIOPTERS[focus_idx]} diopters")
			elif key == ord("g"):
				focus_idx = (focus_idx - 1 + len(FOCUS_DIOPTERS)) % len(FOCUS_DIOPTERS)
				client.send_command("set_focus", FOCUS_DIOPTERS[focus_idx])
				print(f"  Focus -> {FOCUS_DIOPTERS[focus_idx]} diopters")
			elif key == ord("a"):
				auto_exp = not auto_exp
				cmd = "enable_auto_exposure" if auto_exp else "disable_auto_exposure"
				client.send_command(cmd)
				print(f"  Auto-exposure -> {'ON' if auto_exp else 'OFF'}")
			elif key == ord("d"):
				auto_af = not auto_af
				cmd = "enable_auto_focus" if auto_af else "disable_auto_focus"
				client.send_command(cmd)
				print(f"  Auto-focus -> {'ON' if auto_af else 'OFF'}")
			elif key == ord("w"):
				wb_idx = (wb_idx + 1) % len(AWB_MODES)
				client.send_command("set_white_balance", AWB_MODES[wb_idx])
				print(f"  White balance -> {AWB_NAMES[wb_idx]}")
			elif key == ord("t"):
				torch_control = not torch_control
				client.send_command("set_torch", torch_control)
				print(f"  Torch -> {'ON' if torch_control else 'OFF'}")

		if not client.connected:
			print("[!] Connection lost")
	except KeyboardInterrupt:
		pass
	finally:
		client.disconnect()
		cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
