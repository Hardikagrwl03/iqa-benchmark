#!/usr/bin/env python3
"""
USB Camera Controller — Python Client  (low-latency edition)
=============================================================
Receives frames from the Android Camera Controller app over ADB-forwarded USB.

A dedicated receiver thread always holds the *latest* frame, so the display
loop never stalls behind stale data.  Pre-allocated buffers and zero-copy
decoding keep per-frame overhead minimal.

Setup (run once per USB session):
    adb forward tcp:5555 tcp:5555
    adb forward tcp:5556 tcp:5556

Then:
    python camera_client.py
"""

import json
import socket
import struct
import sys
import threading
import time

import cv2
import numpy as np

FRAME_PORT = 5555
COMMAND_PORT = 5556
HOST = "127.0.0.1"
RECV_CHUNK = 256 * 1024          # 256 KB per recv() call
INITIAL_BUF = 2 * 1024 * 1024   # 2 MB pre-allocated receive buffer


class CameraClient:
    """Low-latency dual-TCP client for the Android camera app."""

    def __init__(self, host=HOST, frame_port=FRAME_PORT, command_port=COMMAND_PORT):
        self.host = host
        self.frame_port = frame_port
        self.command_port = command_port
        self.frame_socket: socket.socket | None = None
        self.command_socket: socket.socket | None = None

        # Threaded receiver state
        self._latest_frame: np.ndarray | None = None
        self._frame_lock = threading.Lock()
        self._receiver_thread: threading.Thread | None = None
        self._running = False
        self._connected = False

        # Pre-allocated receive buffer (grows if needed)
        self._recv_buf = bytearray(INITIAL_BUF)

    @property
    def connected(self) -> bool:
        return self._connected

    # ── Connection ──────────────────────────────────────────────────

    def connect(self):
        self.frame_socket = self._make_socket(self.frame_port, "frame")
        self.command_socket = self._make_socket(self.command_port, "command")
        self._connected = True
        self._start_receiver()

    @staticmethod
    def _make_socket(port: int, label: str) -> socket.socket:
        print(f"[*] Connecting to {label} server 127.0.0.1:{port} ...")
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)
        s.connect((HOST, port))
        print(f"[+] {label.capitalize()} connection OK")
        return s

    def disconnect(self):
        self._running = False
        for s in (self.frame_socket, self.command_socket):
            if s:
                try:
                    s.close()
                except OSError:
                    pass
        if self._receiver_thread and self._receiver_thread.is_alive():
            self._receiver_thread.join(timeout=2)
        self.frame_socket = None
        self.command_socket = None
        self._connected = False
        print("[*] Disconnected")

    # ── Threaded frame receiver ─────────────────────────────────────

    def _start_receiver(self):
        self._running = True
        self._receiver_thread = threading.Thread(
            target=self._receive_loop, daemon=True, name="FrameReceiver"
        )
        self._receiver_thread.start()

    def _receive_loop(self):
        """Continuously reads frames and stores only the most recent one."""
        sock = self.frame_socket
        while self._running:
            # --- header (4 bytes) ---
            hdr = self._recv_exact(sock, 4)
            if hdr is None:
                break
            frame_size = struct.unpack(">I", hdr)[0]
            if frame_size <= 0 or frame_size > 10 * 1024 * 1024:
                continue

            # --- payload (JPEG bytes) ---
            data = self._recv_exact(sock, frame_size)
            if data is None:
                break

            frame = cv2.imdecode(
                np.frombuffer(data, dtype=np.uint8, count=frame_size),
                cv2.IMREAD_COLOR,
            )
            if frame is not None:
                with self._frame_lock:
                    self._latest_frame = frame

        self._connected = False

    def get_frame(self) -> np.ndarray | None:
        """Return the latest frame (or *None* if nothing new since last call)."""
        with self._frame_lock:
            f = self._latest_frame
            self._latest_frame = None
            return f

    # ── Pre-allocated receive ───────────────────────────────────────

    def _recv_exact(self, sock: socket.socket, size: int) -> memoryview | None:
        """Read exactly *size* bytes into the pre-allocated buffer."""
        if size > len(self._recv_buf):
            self._recv_buf = bytearray(size)
        view = memoryview(self._recv_buf)[:size]
        pos = 0
        while pos < size:
            try:
                n = sock.recv_into(view[pos:])
            except (ConnectionError, OSError):
                return None
            if n == 0:
                return None
            pos += n
        return view

    # ── Commands ────────────────────────────────────────────────────

    def send_command(self, cmd: str, value=None):
        if not self.command_socket:
            print("[!] Command socket not connected")
            return
        payload = {"cmd": cmd}
        if value is not None:
            payload["value"] = value
        raw = json.dumps(payload).encode("utf-8")
        header = struct.pack(">I", len(raw))
        try:
            self.command_socket.sendall(header + raw)
            print(f"    >>> {payload}")
        except (BrokenPipeError, ConnectionError) as exc:
            print(f"[!] Send failed: {exc}")


# ── Keyboard presets ────────────────────────────────────────────────

EXPOSURE_NS = [
    83_333,          # 1/12000 s
    125_000,         # 1/8000 s
    166_666,         # 1/6000 s
    250_000,         # 1/4000 s
    333_333,         # 1/3000 s
    500_000,         # 1/2000 s
    666_666,         # 1/1500 s
    1_000_000,       # 1/1000 s
    1_333_333,       # 1/750 s
    2_000_000,       # 1/500 s
    2_857_143,       # 1/350 s
    4_000_000,       # 1/250 s
    5_555_556,       # 1/180 s
    8_000_000,       # 1/125 s
    11_111_111,       # 1/90 s
    16_666_666,      # 1/60 s
    20_000_000,      # 1/50 s
    22_222_222,      # 1/45 s
    33_333_333,      # 1/30 s
    50_000_000,      # 1/20 s
    66_666_666,      # 1/15 s
    100_000_000,     # 100 ms
    500_000_000,     # 500 ms
]

ISO_VALUES = [50, 64, 80, 100, 125, 160, 200, 250, 320, 400, 500, 640, 800, 1600, 3200]

FOCUS_DIOPTERS = [i / 10 for i in range(101)]

AWB_MODES = [1, 2, 3, 4, 5]
AWB_NAMES = ["AUTO", "INCANDESCENT", "FLUORESCENT", "WARM_FLUORESCENT", "DAYLIGHT"]


def print_help():
    print(
        "\n"
        "╔══════════════════════════════════════════╗\n"
        "║     Camera Controller — Key Bindings     ║\n"
        "╠══════════════════════════════════════════╣\n"
        "║  e   Cycle exposure time                 ║\n"
        "║  i   Cycle ISO                           ║\n"
        "║  f   Cycle focus distance                ║\n"
        "║  a   Toggle auto-exposure                ║\n"
        "║  d   Toggle auto-focus                   ║\n"
        "║  w   Cycle white-balance mode            ║\n"
        "║  t   Toggle torch / flash                ║\n"
        "║  p   Capture photo (saves locally too)   ║\n"
        "║  h   Show this help                      ║\n"
        "║  q   Quit                                ║\n"
        "╚══════════════════════════════════════════╝\n"
    )


def main():
    client = CameraClient()

    try:
        client.connect()
    except ConnectionRefusedError:
        print(
            "\n[ERROR] Connection refused.\n"
            "  1) Make sure the Android app is running.\n"
            "  2) Set up ADB forwarding:\n"
            "       adb forward tcp:5555 tcp:5555\n"
            "       adb forward tcp:5556 tcp:5556\n"
        )
        sys.exit(1)
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)

    print_help()

    # State indices
    exp_idx = 3
    iso_idx = 0
    focus_idx = 0
    wb_idx = 0
    auto_exp = True
    auto_af = True
    torch = False

    # FPS counter
    frame_count = 0
    fps_start = time.perf_counter()
    fps_label = "— FPS"

    cv2.namedWindow("Camera Controller", cv2.WINDOW_NORMAL)

    try:
        while client.connected:
            frame = client.get_frame()

            if frame is None:
                # No new frame yet — let the window pump events without busy-spinning
                key = cv2.waitKey(5) & 0xFF
            else:
                # FPS bookkeeping
                frame_count += 1
                now = time.perf_counter()
                elapsed = now - fps_start
                if elapsed >= 1.0:
                    fps_label = f"{frame_count / elapsed:.1f} FPS"
                    frame_count = 0
                    fps_start = now

                # HUD overlay drawn directly on the received buffer (no copy)
                h, w = frame.shape[:2]
                cv2.putText(frame, fps_label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame,
                            f"AE:{'ON' if auto_exp else 'OFF'}  AF:{'ON' if auto_af else 'OFF'}"
                            f"  Torch:{'ON' if torch else 'OFF'}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame,
                            f"Exp:{EXPOSURE_NS[exp_idx] / 1e6:.1f}ms  ISO:{ISO_VALUES[iso_idx]}"
                            f"  Focus:{FOCUS_DIOPTERS[focus_idx]}  WB:{AWB_NAMES[wb_idx]}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, "h=help  q=quit", (10, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cv2.imshow("Camera Controller", frame)
                key = cv2.waitKey(1) & 0xFF

            # ── key handling ────────────────────────────────────────
            if key == ord("q"):
                break
            elif key == ord("h"):
                print_help()
            elif key == ord("e"):
                exp_idx = (exp_idx + 1) % len(EXPOSURE_NS)
                client.send_command("set_exposure", EXPOSURE_NS[exp_idx])
                print(f"  Exposure → {EXPOSURE_NS[exp_idx] / 1e6:.1f} ms")
            elif key == ord("i"):
                iso_idx = (iso_idx + 1) % len(ISO_VALUES)
                client.send_command("set_iso", ISO_VALUES[iso_idx])
                print(f"  ISO → {ISO_VALUES[iso_idx]}")
            elif key == ord("f"):
                focus_idx = (focus_idx + 1) % len(FOCUS_DIOPTERS)
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
                torch = not torch
                client.send_command("set_torch", torch)
                print(f"  Torch → {'ON' if torch else 'OFF'}")
            elif key == ord("p"):
                client.send_command("capture_photo")
                ts = time.strftime("%Y%m%d_%H%M%S")
                fname = f"capture_{ts}.jpg"
                if frame is not None:
                    cv2.imwrite(fname, frame)
                    print(f"  Photo saved → {fname}")

        if not client.connected:
            print("[!] Connection lost.")

    except KeyboardInterrupt:
        print("\n[*] Interrupted")
    finally:
        client.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
