#!/usr/bin/env python3
"""
Capture frames from the Android camera client across all exposure, ISO,
and focus-distance combinations, saving each frame to a new folder.
"""

from __future__ import annotations

import argparse
import os
import time
from camera_client import CameraClient, adb_connect
import cv2

EXPOSURE_NS = [
	166_666,         # 1/6000 s
    500_000,         # 1/2000 s
    1_333_333,       # 1/750 s 
    4_000_000,       # 1/250 s
    11_111_111,      # 1/90 s
    33_333_333,      # 1/30 s
]

ISO_VALUES = [50, 100, 200, 400, 800]

FOCUS_DIOPTERS = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0]


def wait_for_frame(client: CameraClient, timeout_s: float) -> tuple[bool, object]:
	deadline = time.time() + timeout_s
	frame = None
	while time.time() < deadline:
		frame = client.get_frame()
		if frame is not None:
			return True, frame
		time.sleep(0.01)
	return False, frame


def capture_all(output_dir: str, settle_s: float, timeout_s: float) -> None:
	os.makedirs(output_dir, exist_ok=True)
	adb_connect()
	client = CameraClient()
	client.connect()

	try:
		client.send_command("disable_auto_exposure")
		client.send_command("disable_auto_focus")
		client.send_command("set_torch", False)

		total = len(EXPOSURE_NS) * len(ISO_VALUES) * len(FOCUS_DIOPTERS)
		idx = 0

		for exposure_ns in EXPOSURE_NS:
			client.send_command("set_exposure", exposure_ns)
			time.sleep(settle_s)

			for iso in ISO_VALUES:
				client.send_command("set_iso", iso)
				time.sleep(settle_s)

				for focus in FOCUS_DIOPTERS:
					client.send_command("set_focus", focus)
					time.sleep(settle_s)

					ok, frame = wait_for_frame(client, timeout_s)
					idx += 1
					if not ok:
						print(f"[WARN] No frame for exp={exposure_ns} iso={iso} focus={focus}")
						continue

					fname = (
						f"exp_{exposure_ns}ns_iso_{iso}_focus_{focus:.1f}.jpg"
					)
					path = os.path.join(output_dir, fname)
					cv2.imwrite(path, frame)
					print(f"[{idx}/{total}] Saved {path}")

	finally:
		client.disconnect()


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Capture camera frames across all exposure/ISO/focus combos."
	)
	parser.add_argument(
		"--output-dir",
		default=None,
		help="Output directory (default: captures_YYYYMMDD_HHMMSS)",
	)
	parser.add_argument(
		"--settle-s",
		type=float,
		default=0.3,
		help="Seconds to wait after each setting change",
	)
	parser.add_argument(
		"--timeout-s",
		type=float,
		default=2.0,
		help="Max seconds to wait for a frame",
	)
	args = parser.parse_args()

	if args.output_dir:
		out_dir = args.output_dir
	else:
		ts = time.strftime("%Y%m%d_%H%M%S")
		out_dir = f"captures_{ts}"

	capture_all(out_dir, args.settle_s, args.timeout_s)


if __name__ == "__main__":
	main()
