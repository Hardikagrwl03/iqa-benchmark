#!/usr/bin/env python3
"""Live HyperIQA on frames from the Android camera client."""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from camera_client import AWB_MODES, AWB_NAMES, EXPOSURE_NS, FOCUS_DIOPTERS, ISO_VALUES, CameraClient

HYPER_IQA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hyperIQA")
sys.path.append(HYPER_IQA_DIR)

import models


def load_model(model_path: str, device: str):
	model = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).to(device)
	state = torch.load(model_path, map_location=device)
	model.load_state_dict(state)
	model.eval()
	return model


def build_normalize():
	return transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


def preprocess_frame(frame_bgr: np.ndarray, normalize, patch_size: int, device: str):
	frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
	image = Image.fromarray(frame_rgb)
	resized = transforms.Resize((512, 384))(image)
	center_crop = transforms.CenterCrop((patch_size, patch_size))
	patch = center_crop(resized)
	patch = transforms.ToTensor()(patch)
	patch = normalize(patch)
	batch = patch.unsqueeze(0).to(device)
	resized_w, resized_h = resized.size
	left = max((resized_w - patch_size) // 2, 0)
	top = max((resized_h - patch_size) // 2, 0)
	box = (left, top, patch_size, patch_size)
	return batch, resized.size, box


@torch.no_grad()
def infer(model_hyper, batch: torch.Tensor) -> float:
	paras = model_hyper(batch)
	model_target = models.TargetNet(paras).to(batch.device)
	for param in model_target.parameters():
		param.requires_grad = False
	pred = model_target(paras["target_in_vec"])
	return float(pred.mean().item())


def adb_connect():
	import subprocess

	try:
		subprocess.run(["adb", "forward", "tcp:5555", "tcp:5555"], check=True)
		subprocess.run(["adb", "forward", "tcp:5556", "tcp:5556"], check=True)
		print("[+] ADB port forwarding set up successfully")
	except subprocess.CalledProcessError as exc:
		print(f"[!] ADB port forwarding failed: {exc}")
		sys.exit(1)


def main():
	parser = argparse.ArgumentParser(description="Live HyperIQA on Android camera frames")
	parser.add_argument("--model_path", required=True, help="Path to the HyperIQA pretrained model")
	parser.add_argument("--patch_size", type=int, default=224, help="Patch size for HyperIQA")
	parser.add_argument("--interval_ms", type=int, default=200, help="Inference interval in ms")
	args = parser.parse_args()
    
	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Using device: {device}")
	model = load_model(args.model_path, device)
	normalize = build_normalize()

	adb_connect()
	client = CameraClient()
	client.connect()

	last_score = None
	last_infer = 0.0
	last_inf_time = 0.0
	last_box = None

	cv2.namedWindow("Live HyperIQA", cv2.WINDOW_NORMAL)

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
					batch, resized_size, first_box = preprocess_frame(
						frame, normalize, args.patch_size, device
					)
					score = infer(model, batch)
					last_inf_time = time.time() - start
					last_score = score
					last_infer = time.time()
					last_box = (first_box, resized_size)

				if last_box is not None:
					(box, (resized_w, resized_h)) = last_box
					left, top, width, height = box
					orig_h, orig_w = frame.shape[:2]
					scale_x = orig_w / resized_w
					scale_y = orig_h / resized_h
					x1 = int(left * scale_x)
					y1 = int(top * scale_y)
					x2 = int((left + width) * scale_x)
					y2 = int((top + height) * scale_y)
					cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

				if last_score is not None:
					cv2.putText(
						frame,
						f"HyperIQA: {last_score:.2f}",
						(10, 30),
						cv2.FONT_HERSHEY_SIMPLEX,
						0.9,
						(0, 255, 0),
						2,
					)
					cv2.putText(
						frame,
						f"HyperIQA Inf_Time: {last_inf_time:.4f}",
						(10, 60),
						cv2.FONT_HERSHEY_SIMPLEX,
						0.9,
						(0, 255, 0),
						2,
					)

				cv2.imshow("Live HyperIQA", frame)
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
