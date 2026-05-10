#!/usr/bin/env python3
"""Live IQA on frames from the Android camera client."""

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

LAR_IQA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "LAR-IQA")
sys.path.append(LAR_IQA_DIR)

from models.mobilenet_merged_with_kan import MobileNetMergedWithKAN
from models.mobilenet_merged import MobileNetMerged


def load_model(model_path: str, use_kan: bool, device: str):
	model = MobileNetMergedWithKAN() if use_kan else MobileNetMerged()
	state = torch.load(model_path, map_location=device)
	model.load_state_dict(state)
	model.to(device)
	model.eval()
	return model


def _ensure_min_size(image: Image.Image, min_side: int) -> Image.Image:
	width, height = image.size
	if min(width, height) >= min_side:
		return image
	resize = transforms.Resize(min_side)
	return resize(image)


def preprocess_frame(frame_bgr: np.ndarray, color_space: str, device: str):
	frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

	if color_space == "HSV":
		frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
	elif color_space == "LAB":
		frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
	elif color_space == "YUV":
		frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2YUV)

	image = Image.fromarray(frame_rgb)
	image = _ensure_min_size(image, 1280)

	transform_authentic = transforms.Compose(
		[
			transforms.Resize((384, 384)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		]
	)
	transform_synthetic = transforms.Compose(
		[
			transforms.CenterCrop((1280, 1280)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		]
	)

	image_authentic = transform_authentic(image).unsqueeze(0).to(device)
	image_synthetic = transform_synthetic(image).unsqueeze(0).to(device)

	return image_authentic, image_synthetic


@torch.no_grad()
def infer(model, image_authentic, image_synthetic) -> float:
	output = model(image_authentic, image_synthetic)
	return float(output.item())

def adb_connect():
	import subprocess

	try:
		subprocess.run(["adb", "forward", "tcp:5555", "tcp:5555"], check=True)
		subprocess.run(["adb", "forward", "tcp:5556", "tcp:5556"], check=True)
		print("[+] ADB port forwarding set up successfully")
	except subprocess.CalledProcessError as e:
		print(f"[!] ADB port forwarding failed: {e}")
		sys.exit(1)

def main():
	parser = argparse.ArgumentParser(description="Live IQA on Android camera frames")
	parser.add_argument("--model_path", required=True, help="Path to the trained model")
	parser.add_argument("--use_kan", action="store_true", help="Use MobileNetMergedWithKAN")
	parser.add_argument(
		"--color_space",
		choices=["RGB", "HSV", "LAB", "YUV"],
		default="RGB",
		help="Color space for inference",
	)
	parser.add_argument("--interval_ms", type=int, default=200, help="Inference interval in ms")
	# parser.add_argument("--no_display", action="store_true", help="Disable live preview")

	args = parser.parse_args()

	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Using device: {device}")
	model = load_model(args.model_path, args.use_kan, device)

	adb_connect()
	client = CameraClient()
	client.connect()

	last_score = None
	last_infer = 0.0
	# show_window = not args.no_display
	cv2.namedWindow("Live IQA", cv2.WINDOW_NORMAL)
		
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
				start = time.time()
				image_authentic, image_synthetic = preprocess_frame(
					frame, args.color_space, device
				)
				score = infer(model, image_authentic, image_synthetic)
				inf_time = time.time() - start
				print(f"IQA: {score:.4f}, Time: {inf_time:.4f} s")
				cv2.putText(
					frame,
					f"LAR-IQA: {score:.4f}",
					(10, 30),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.9,
					(0, 255, 0),
					2,
				)
				cv2.putText(
					frame,
					f"LAR-IQA Inf_Time: {inf_time:.4f}",
					(10, 60),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.9,
					(0, 255, 0),
					2,
				)
				
				cv2.imshow("Live IQA", frame)
				key = cv2.waitKey(1) & 0xFF

			if key == ord("q"):
				break
			elif key == ord("e"):
				exp_idx = (exp_idx + 1) % len(EXPOSURE_NS)
				client.send_command("set_exposure", EXPOSURE_NS[exp_idx])
				print(f"  Exposure → {EXPOSURE_NS[exp_idx] / 1e6:.1f} ms")
			elif key == ord("r"):
				exp_idx = (exp_idx - 1 + len(EXPOSURE_NS)) % len(EXPOSURE_NS)
				client.send_command("set_exposure", EXPOSURE_NS[exp_idx])
				print(f"  Exposure → {EXPOSURE_NS[exp_idx] / 1e6:.1f} ms")
			elif key == ord("i"):
				iso_idx = (iso_idx + 1) % len(ISO_VALUES)
				client.send_command("set_iso", ISO_VALUES[iso_idx])
				print(f"  ISO → {ISO_VALUES[iso_idx]}")
			elif key == ord("o"):
				iso_idx = (iso_idx - 1 + len(ISO_VALUES)) % len(ISO_VALUES)
				client.send_command("set_iso", ISO_VALUES[iso_idx])
				print(f"  ISO → {ISO_VALUES[iso_idx]}")
			elif key == ord("f"):
				focus_idx = (focus_idx + 1) % len(FOCUS_DIOPTERS)
				client.send_command("set_focus", FOCUS_DIOPTERS[focus_idx])
				print(f"  Focus → {FOCUS_DIOPTERS[focus_idx]} diopters")
			elif key == ord("g"):
				focus_idx = (focus_idx - 1 + len(FOCUS_DIOPTERS)) % len(FOCUS_DIOPTERS)
				client.send_command("set_focus", FOCUS_DIOPTERS[focus_idx])
				print(f"  Focus → {FOCUS_DIOPTERS[focus_idx]} diopters")
			elif key == ord("a"):
				auto_exp = not auto_exp
				cmd = "enable_auto_exposure" if auto_exp else "disable_auto_exposure"
				client.send_command(cmd)
				print(f"  Auto-exposure → {'ON' if auto_exp else 'OFF'}")
			elif key == ord("d"):
				auto_af = not auto_af
				cmd = "enable_auto_focus" if auto_af else "disable_auto_focus"
				client.send_command(cmd)
				print(f"  Auto-focus → {'ON' if auto_af else 'OFF'}")
			elif key == ord("w"):
				wb_idx = (wb_idx + 1) % len(AWB_MODES)
				client.send_command("set_white_balance", AWB_MODES[wb_idx])
				print(f"  White balance → {AWB_NAMES[wb_idx]}")
			elif key == ord("t"):
				torch_control = not torch_control
				client.send_command("set_torch", torch_control)
				print(f"  Torch → {'ON' if torch_control else 'OFF'}")
		if not client.connected:
			print("[!] Connection lost")
	except KeyboardInterrupt:
		pass
	finally:
		client.disconnect()
		cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
