"""
VinylPlayer v0.1.6 (single-page + collapsible sections + scroll)

Order (top → bottom):
AUDIO IN/OUT
INPUT LEVEL
MONITOR CONTROLS
EQ
RECORD
RUN ON WINDOWS STARTUP

Notes:
- All sections start CLOSED by default.
- Device selection is saved STABLY by device name + host API (so it survives Windows re-ordering).
- Device dropdowns show NAMES ONLY (no #numbers).
- Filters out useless "Primary Sound Capture Driver" / "Microsoft Sound Mapper" entries.
- Prefers WASAPI devices on Windows (more reliable).
- Wave Bars column count auto-scales to window width (no manual editing).

Run:
  py vinyl_listener.py

Build EXE:
  py -m PyInstaller --noconfirm --clean --onefile --windowed --name "VinylPlayer" --icon "vinyl.ico" --add-data "vinyl.ico;." "vinyl_listener.py"
"""

# ===== CHUNK 1 / 2 =====

import json
import math
import os
import platform
import sys
import threading
import queue
import time
import wave
from dataclasses import dataclass
from collections import deque

import numpy as np
import sounddevice as sd

from PySide6.QtCore import Qt, QTimer, QSize, QRect, QByteArray
from PySide6.QtGui import QPainter, QColor, QPen, QIcon, QLinearGradient, QBrush
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QSlider,
    QPushButton,
    QMessageBox,
    QCheckBox,
    QGroupBox,
    QScrollArea,
    QToolButton,
    QFrame,
    QFileDialog,
    QLineEdit,
    QSizePolicy,
)

VERSION = "0.1.6"
def settings_file_path() -> str:
    # if packaged (PyInstaller), sys.executable is the .exe
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "vinyl_player_settings.json")

SETTINGS_FILE = settings_file_path()

GEQ_FREQS = [31, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
GEQ_LABELS = ["31", "63", "125", "250", "500", "1k", "2k", "4k", "8k", "16k"]

ACCENT_GREEN = "#00d278"

APP_STYLE = f"""
QWidget {{
    background-color: #101010;
    color: #eaeaea;
    font-size: 10.5pt;
}}

QGroupBox {{
    border: 1px solid #2b2b2b;
    border-radius: 10px;
    margin-top: 10px;
    padding: 10px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #cfcfcf;
}}

QPushButton {{
    background: #1a1a1a;
    border: 1px solid #2b2b2b;
    padding: 8px 12px;
    border-radius: 10px;
}}
QPushButton:hover {{
    border-color: {ACCENT_GREEN};
}}
QPushButton:pressed {{
    background: #0f0f0f;
}}

QComboBox {{
    background: #151515;
    border: 1px solid #2b2b2b;
    border-radius: 10px;
    padding: 6px 10px;
}}
QComboBox:hover {{
    border-color: {ACCENT_GREEN};
}}

QCheckBox::indicator {{
    width: 18px;
    height: 18px;
}}
QCheckBox::indicator:unchecked {{
    border: 1px solid #2b2b2b;
    background: #151515;
    border-radius: 4px;
}}
QCheckBox::indicator:checked {{
    border: 1px solid {ACCENT_GREEN};
    background: {ACCENT_GREEN};
    border-radius: 4px;
}}

QSlider::groove:horizontal {{
    height: 10px;
    border-radius: 5px;
    background: #2a2a2a;
}}
QSlider::sub-page:horizontal {{
    border-radius: 5px;
    background: {ACCENT_GREEN};
}}
QSlider::add-page:horizontal {{
    border-radius: 5px;
    background: #3a3a3a;
}}
QSlider::handle:horizontal {{
    width: 18px;
    margin: -6px 0;
    border-radius: 9px;
    background: {ACCENT_GREEN};
}}

QSlider::groove:vertical {{
    width: 10px;
    border-radius: 5px;
    background: #2a2a2a;
}}
QSlider::sub-page:vertical {{
    border-radius: 5px;
    background: {ACCENT_GREEN};
}}
QSlider::add-page:vertical {{
    border-radius: 5px;
    background: #3a3a3a;
}}
QSlider::handle:vertical {{
    height: 18px;
    margin: 0 -6px;
    border-radius: 9px;
    background: {ACCENT_GREEN};
}}
QScrollBar:vertical, QScrollBar:horizontal {{
    background: #0f0f0f;
    border: 1px solid #2b2b2b;
    border-radius: 8px;
    margin: 2px;
}}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background: #1f1f1f;
    border: 1px solid #2b2b2b;
    border-radius: 8px;
    min-height: 30px;
    min-width: 30px;
}}
QScrollBar::handle:hover {{
    border-color: #00d278;
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    background: none;
    border: none;
}}
QScrollBar::add-page, QScrollBar::sub-page {{
    background: none;
}}

"""


def resource_path(relative_name: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, relative_name)


def load_settings():
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_settings(data):
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def is_windows() -> bool:
    return platform.system().lower() == "windows"


def startup_run_key_name() -> str:
    return "VinylPlayer"


def is_startup_enabled() -> bool:
    if not is_windows():
        return False
    try:
        import winreg

        key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_READ) as key:
            try:
                winreg.QueryValueEx(key, startup_run_key_name())
                return True
            except FileNotFoundError:
                return False
    except Exception:
        return False


def set_startup_enabled(enable: bool):
    if not is_windows():
        return
    import winreg

    key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path) as key:
        name = startup_run_key_name()
        if enable:
            exe = sys.executable
            if exe.lower().endswith("python.exe") or exe.lower().endswith("pythonw.exe"):
                script = os.path.abspath(__file__)
                cmd = f"\"{exe}\" \"{script}\""
            else:
                cmd = f"\"{exe}\""
            winreg.SetValueEx(key, name, 0, winreg.REG_SZ, cmd)
        else:
            try:
                winreg.DeleteValue(key, name)
            except FileNotFoundError:
                pass


def set_windows_appusermodelid(app_id: str):
    if not is_windows():
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
        pass


def list_devices_with_hostapi():
    devs = sd.query_devices()
    hostapis = sd.query_hostapis()
    items = []
    for i, d in enumerate(devs):
        hostapi_idx = int(d.get("hostapi", -1))
        hostapi_name = hostapis[hostapi_idx]["name"] if 0 <= hostapi_idx < len(hostapis) else "Unknown"
        items.append(
            (
                i,
                d.get("name", f"Device {i}"),
                int(d.get("max_input_channels", 0)),
                int(d.get("max_output_channels", 0)),
                float(d.get("default_samplerate", 48000.0)),
                hostapi_idx,
                hostapi_name,
            )
        )
    return items


# -----------------------------
# Collapsible Section
# -----------------------------

class CollapsibleBox(QWidget):
    def __init__(self, title: str, start_open: bool = False, parent=None):
        super().__init__(parent)

        self.toggle_button = QToolButton(text=title, checkable=True, checked=start_open)
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.DownArrow if start_open else Qt.RightArrow)
        self.toggle_button.setStyleSheet(
            f"""
            QToolButton {{
                border: 1px solid #2b2b2b;
                border-radius: 10px;
                padding: 10px 12px;
                background: #121212;
                color: #ffffff;
                font-weight: 600;
                text-align: left;
            }}
            QToolButton:hover {{
                border-color: {ACCENT_GREEN};
            }}
            """
        )

        self.content = QFrame()
        self.content.setFrameShape(QFrame.NoFrame)
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 10, 0, 0)
        self.content_layout.setSpacing(10)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        lay.addWidget(self.toggle_button)
        lay.addWidget(self.content)

        self.content.setVisible(start_open)
        self.toggle_button.toggled.connect(self._on_toggled)

    def _on_toggled(self, checked: bool):
        self.content.setVisible(checked)
        self.toggle_button.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)

    def body_layout(self) -> QVBoxLayout:
        return self.content_layout


# -----------------------------
# Meter Widget
# -----------------------------

class LevelMeter(QWidget):
    MODES = ["EQ Bars", "Wave Bars"]

    def __init__(self, history_len: int = 320, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(160)
        self.setMaximumHeight(240)

        self._mode = "EQ Bars"
        self._history = deque([0.0] * history_len, maxlen=history_len)

        self._last_db_text = "-inf dB"
        self._hot = False
        self._hot_th = 0.98

        self._bg = QColor(0, 0, 0)
        self._grid = QColor(35, 35, 35)
        self._text = QColor(235, 235, 235)

        self._green = QColor(0, 230, 120)
        self._yellow = QColor(255, 210, 0)
        self._red = QColor(255, 70, 70)

        # EQ Bars
        self._eq_cols = 30
        self._eq_segments = 16
        self._eq_gap_px = 6
        self._eq_seg_gap_px = 2
        self._reflection = True

        # Wave Bars (auto columns)
        self._wave_segments = 25
        self._wave_seg_gap_px = 3
        self._wave_bar_gap_px = 4
        self._wave_bar_w_px = 6

        self._display_level = 0.0

    def sizeHint(self) -> QSize:
        return QSize(760, 200)

    def set_mode(self, mode: str):
        if mode in self.MODES:
            self._mode = mode
            self.update()

    def push_peak(self, peak: float):
        p = float(max(0.0, min(1.0, peak)))

        if p > self._display_level:
            self._display_level = 0.75 * self._display_level + 0.25 * p
        else:
            self._display_level = 0.90 * self._display_level + 0.10 * p

        self._history.append(self._display_level)

        if p <= 1e-6:
            self._last_db_text = "-inf dB"
        else:
            self._last_db_text = f"{(20.0 * math.log10(p)):.1f} dB"

        self._hot = p >= self._hot_th
        self.update()

    def _color_for_segment(self, seg_index: int, segments: int) -> QColor:
        pos = (seg_index + 1) / float(segments)
        if pos >= 0.60:
            return self._red
        if pos >= 0.35:
            return self._yellow
        return self._green

    def paintEvent(self, event):
        w = self.width()
        h = self.height()

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        p.fillRect(0, 0, w, h, self._bg)
        p.setPen(QPen(self._grid, 1))
        p.drawLine(0, 28, w, 28)

        if self._mode == "EQ Bars":
            self._paint_eq_bars(p, w, h)
        else:
            self._paint_wave_bars(p, w, h)

        p.setPen(QPen(self._text))
        text = self._last_db_text + ("  HOT" if self._hot else "")
        p.drawText(10, 20, text)
        p.end()

    def _paint_eq_bars(self, p: QPainter, w: int, h: int):
        top_pad = 32
        bottom_pad = 10
        total_h = h - top_pad - bottom_pad

        if self._reflection:
            main_h = int(total_h * 0.62)
            refl_h = total_h - main_h
        else:
            main_h = total_h
            refl_h = 0

        cols = self._eq_cols
        gap = self._eq_gap_px
        usable_w = w - (cols + 1) * gap
        bar_w = max(6, usable_w // cols)

        if bar_w < 8:
            gap = 3
            usable_w = w - (cols + 1) * gap
            bar_w = max(5, usable_w // cols)

        hist = list(self._history)
        if not hist:
            return

        take = min(len(hist), cols * 10)
        recent = hist[-take:]

        values = []
        group = max(1, len(recent) // cols)
        for i in range(cols):
            start = max(0, len(recent) - (cols - i) * group)
            end = max(0, len(recent) - (cols - i - 1) * group)
            chunk = recent[start:end] if end > start else recent[-1:]
            base = float(max(chunk)) if chunk else 0.0
            tilt = 0.90 + 0.20 * math.sin((i / max(1, cols - 1)) * math.pi)
            jitter = 0.02 * math.sin((len(recent) + i) * 0.7)
            v = max(0.0, min(1.0, base * tilt + jitter))
            values.append(v)

        segments = self._eq_segments
        seg_gap = self._eq_seg_gap_px
        seg_h = max(3, (main_h - (segments - 1) * seg_gap) // segments)

        x = gap
        y0 = top_pad + main_h

        for v in values:
            lit = int(round(v * segments))
            lit = max(0, min(segments, lit))
            for s in range(lit):
                seg_y = y0 - (s + 1) * seg_h - s * seg_gap
                p.fillRect(QRect(x, seg_y, bar_w, seg_h), self._color_for_segment(s, segments))
            x += bar_w + gap

        if self._reflection and refl_h > 6:
            fade = QLinearGradient(0, top_pad + main_h, 0, top_pad + main_h + refl_h)
            fade.setColorAt(0.0, QColor(255, 255, 255, 85))
            fade.setColorAt(1.0, QColor(255, 255, 255, 0))

            x = gap
            for v in values:
                lit = int(round(v * segments))
                lit = max(0, min(segments, lit))
                for s in range(lit):
                    seg_y = y0 - (s + 1) * seg_h - s * seg_gap
                    dist_from_bottom = y0 - (seg_y + seg_h)
                    refl_y = (top_pad + main_h) + dist_from_bottom
                    base = self._color_for_segment(s, segments)
                    refl_color = QColor(base.red(), base.green(), base.blue(), 90)
                    p.fillRect(QRect(x, refl_y, bar_w, seg_h), refl_color)
                x += bar_w + gap

            p.fillRect(0, top_pad + main_h, w, refl_h, QBrush(fade))

    def _paint_wave_bars(self, p: QPainter, w: int, h: int):
        top_pad = 32
        bottom_pad = 10

        usable_h = h - top_pad - bottom_pad
        center_y = top_pad + usable_h // 2

        p.setPen(QPen(QColor(0, 60, 40), 1))
        p.drawLine(0, center_y, w, center_y)

        bar_w = self._wave_bar_w_px
        gap = self._wave_bar_gap_px

        usable_w = max(0, w - 20)
        cols = max(40, int(usable_w // (bar_w + gap)))

        total_bar_w = cols * bar_w + (cols - 1) * gap
        x_start = (w - total_bar_w) // 2

        segments = self._wave_segments
        seg_gap = self._wave_seg_gap_px

        half_h = usable_h // 2 - 6
        seg_h = max(2, (half_h - (segments - 1) * seg_gap) // segments)

        hist = list(self._history)
        if not hist:
            return

        take = min(len(hist), cols * 6)
        recent = hist[-take:]
        if len(recent) < cols:
            recent = [0.0] * (cols - len(recent)) + recent

        values = []
        step = max(1, len(recent) // cols)
        for i in range(cols):
            start = i * step
            end = min(len(recent), (i + 1) * step)
            chunk = recent[start:end] if end > start else recent[-1:]
            base = float(max(chunk)) if chunk else 0.0
            t = i / max(1, cols - 1)
            envelope = math.sin(t * math.pi) ** 0.6
            v = base * (0.55 + 0.45 * envelope)
            values.append(max(0.0, min(1.0, v)))

        x = x_start
        for v in values:
            lit = int(round(v * segments))
            lit = max(0, min(segments, lit))

            for s in range(lit):
                color = self._color_for_segment(s, segments)
                y_top = center_y - (s + 1) * seg_h - s * seg_gap
                p.fillRect(QRect(x, y_top, bar_w, seg_h), color)
                y_bot = center_y + s * (seg_h + seg_gap)
                p.fillRect(QRect(x, y_bot, bar_w, seg_h), color)

            x += bar_w + gap


# -----------------------------
# DSP: Biquad filters
# -----------------------------

@dataclass
class BiquadCoeffs:
    b0: float
    b1: float
    b2: float
    a1: float
    a2: float


class Biquad:
    def __init__(self, channels: int):
        self.set_channels(channels)
        self.set_coeffs(BiquadCoeffs(1.0, 0.0, 0.0, 0.0, 0.0))

    def set_channels(self, channels: int):
        self.channels = int(channels)
        if self.channels not in (1, 2):
            raise ValueError("Biquad supports only 1 or 2 channels.")
        self.reset()

    def set_coeffs(self, c: BiquadCoeffs):
        self.b0 = float(c.b0)
        self.b1 = float(c.b1)
        self.b2 = float(c.b2)
        self.a1 = float(c.a1)
        self.a2 = float(c.a2)

    def reset(self):
        if self.channels == 1:
            self.x1 = self.x2 = self.y1 = self.y2 = 0.0
        else:
            self.x1L = self.x2L = self.y1L = self.y2L = 0.0
            self.x1R = self.x2R = self.y1R = self.y2R = 0.0

    def process_inplace(self, buf: np.ndarray):
        b0, b1, b2, a1, a2 = self.b0, self.b1, self.b2, self.a1, self.a2
        nframes = buf.shape[0]

        if self.channels == 1:
            x1, x2, y1, y2 = self.x1, self.x2, self.y1, self.y2
            for n in range(nframes):
                x = float(buf[n, 0])
                y = (b0 * x) + (b1 * x1) + (b2 * x2) - (a1 * y1) - (a2 * y2)
                x2, x1 = x1, x
                y2, y1 = y1, y
                buf[n, 0] = y
            self.x1, self.x2, self.y1, self.y2 = x1, x2, y1, y2
            return buf

        x1L, x2L, y1L, y2L = self.x1L, self.x2L, self.y1L, self.y2L
        x1R, x2R, y1R, y2R = self.x1R, self.x2R, self.y1R, self.y2R

        for n in range(nframes):
            xL = float(buf[n, 0])
            yL = (b0 * xL) + (b1 * x1L) + (b2 * x2L) - (a1 * y1L) - (a2 * y2L)
            x2L, x1L = x1L, xL
            y2L, y1L = y1L, yL
            buf[n, 0] = yL

            xR = float(buf[n, 1])
            yR = (b0 * xR) + (b1 * x1R) + (b2 * x2R) - (a1 * y1R) - (a2 * y2R)
            x2R, x1R = x1R, xR
            y2R, y1R = y1R, yR
            buf[n, 1] = yR

        self.x1L, self.x2L, self.y1L, self.y2L = x1L, x2L, y1L, y2L
        self.x1R, self.x2R, self.y1R, self.y2R = x1R, x2R, y1R, y2R
        return buf


def biquad_peaking(fs: float, f0: float, q: float, gain_db: float) -> BiquadCoeffs:
    gain_db = float(max(-30.0, min(30.0, gain_db)))
    A = 10 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * f0 / fs
    alpha = math.sin(w0) / (2.0 * q)
    cosw0 = math.cos(w0)

    b0 = 1 + alpha * A
    b1 = -2 * cosw0
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * cosw0
    a2 = 1 - alpha / A

    b0 /= a0
    b1 /= a0
    b2 /= a0
    a1 /= a0
    a2 /= a0
    return BiquadCoeffs(b0, b1, b2, a1, a2)


# -----------------------------
# Recording (WAV) - async writer (no callback disk writes)
# -----------------------------

class WavRecorder:
    """
    Threaded WAV recorder:
    - audio callback enqueues int16 bytes
    - writer thread writes to disk
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._q = queue.Queue(maxsize=200)
        self._wf = None
        self._thread = None
        self._stop_evt = threading.Event()

        self._samplerate = 48000
        self._channels = 2
        self._frames_written = 0
        self._dropped_blocks = 0
        self._start_time = None

    def start(self, path: str, samplerate: int, channels: int):
        # stop() must be called OUTSIDE the lock or it deadlocks
        self.stop()

        with self._lock:
            self._samplerate = int(samplerate)
            self._channels = int(channels)
            self._frames_written = 0
            self._dropped_blocks = 0
            self._start_time = time.time()

            while not self._q.empty():
                try:
                    self._q.get_nowait()
                except Exception:
                    break

            wf = wave.open(path, "wb")
            wf.setnchannels(self._channels)
            wf.setsampwidth(2)  # int16
            wf.setframerate(self._samplerate)
            self._wf = wf

            self._stop_evt.clear()
            self._thread = threading.Thread(target=self._writer_loop, daemon=True)
            self._thread.start()

    def stop(self):
        with self._lock:
            if self._wf is None:
                return
            self._stop_evt.set()

        if self._thread is not None:
            try:
                self._thread.join(timeout=1.5)
            except Exception:
                pass

        with self._lock:
            try:
                if self._wf is not None:
                    self._wf.close()
            except Exception:
                pass
            self._wf = None
            self._thread = None

            while not self._q.empty():
                try:
                    self._q.get_nowait()
                except Exception:
                    break

    def is_recording(self) -> bool:
        with self._lock:
            return self._wf is not None

    def frames_written(self) -> int:
        with self._lock:
            return int(self._frames_written)

    def dropped_blocks(self) -> int:
        with self._lock:
            return int(self._dropped_blocks)

    def enqueue_float32(self, block: np.ndarray):
        with self._lock:
            wf = self._wf
        if wf is None:
            return

        x = np.clip(block, -1.0, 1.0)
        i16 = (x * 32767.0).astype(np.int16, copy=False)
        data = i16.tobytes()

        try:
            self._q.put_nowait((data, block.shape[0]))
        except queue.Full:
            with self._lock:
                self._dropped_blocks += 1

    def _writer_loop(self):
        while not self._stop_evt.is_set() or not self._q.empty():
            try:
                data, frames = self._q.get(timeout=0.15)
            except Exception:
                continue

            with self._lock:
                wf = self._wf
            if wf is None:
                continue

            try:
                wf.writeframes(data)
                with self._lock:
                    self._frames_written += int(frames)
            except Exception:
                self._stop_evt.set()
                break


# -----------------------------
# Audio Engine
# -----------------------------

class AudioEngine:
    def __init__(self):
        self.stream = None
        self.volume = 0.8
        self.muted = False
        self.blocksize = 1024

        self.in_channels = 1
        self.out_channels = 2
        self.samplerate = 48000

        self._meter_lock = threading.Lock()
        self.input_peak = 0.0

        self._eq_lock = threading.Lock()
        self.eq_enabled = True
        self.eq_preamp_db = 0.0
        self.eq_band_db = [0.0] * 10
        self._geq_filters: list[Biquad] = []

        self.recorder = WavRecorder()

    def get_input_peak(self) -> float:
        with self._meter_lock:
            return float(self.input_peak)

    @staticmethod
    def _pick_channel_counts(input_dev: int, output_dev: int):
        in_info = sd.query_devices(input_dev)
        out_info = sd.query_devices(output_dev)
        max_in = int(in_info.get("max_input_channels", 0))
        max_out = int(out_info.get("max_output_channels", 0))
        if max_in <= 0:
            raise RuntimeError("Selected input device has 0 input channels.")
        if max_out <= 0:
            raise RuntimeError("Selected output device has 0 output channels.")
        in_ch = 2 if max_in >= 2 else 1
        out_ch = 2 if max_out >= 2 else 1
        return in_ch, out_ch

    def _rebuild_geq_locked(self):
        q = 1.0
        if len(self._geq_filters) != 10:
            self._geq_filters = [Biquad(self.out_channels) for _ in range(10)]
        else:
            for flt in self._geq_filters:
                if flt.channels != self.out_channels:
                    flt.set_channels(self.out_channels)

        for flt, f, g in zip(self._geq_filters, GEQ_FREQS, self.eq_band_db):
            flt.set_coeffs(biquad_peaking(float(self.samplerate), float(f), q, float(g)))

    def set_graphic_eq(self, enabled: bool, preamp_db: float, band_db: list[float]):
        if not isinstance(band_db, list) or len(band_db) != 10:
            return
        with self._eq_lock:
            self.eq_enabled = bool(enabled)
            self.eq_preamp_db = float(preamp_db)
            self.eq_band_db = [float(x) for x in band_db]
            if self.stream is not None:
                self._rebuild_geq_locked()

    def start(self, input_dev: int, output_dev: int):
        self.stop()

        in_info = sd.query_devices(input_dev)
        out_info = sd.query_devices(output_dev)
        if int(in_info.get("hostapi", -1)) != int(out_info.get("hostapi", -1)):
            raise RuntimeError(
                "Input and Output are from different Windows audio systems (host APIs). "
                "Pick devices from the same group (prefer WASAPI)."
            )

        preferred_rates = [48000, 44100]
        self.in_channels, self.out_channels = self._pick_channel_counts(input_dev, output_dev)

        last_err = None
        for fs in preferred_rates:
            try:
                self.samplerate = int(fs)
                with self._eq_lock:
                    self._rebuild_geq_locked()

                def callback(indata, outdata, frames, time_info, status):
                    block_peak = float(np.max(np.abs(indata))) if indata.size else 0.0
                    with self._meter_lock:
                        self.input_peak = max(self.input_peak * 0.85, block_peak)

                    x = indata.astype(np.float32, copy=False)
                    if x.ndim == 1:
                        x = x.reshape(-1, 1)

                    if x.shape[1] == self.out_channels:
                        y = x
                    elif x.shape[1] == 1 and self.out_channels == 2:
                        y = np.repeat(x, 2, axis=1)
                    elif x.shape[1] == 2 and self.out_channels == 1:
                        y = np.mean(x, axis=1, keepdims=True)
                    else:
                        if x.shape[1] > self.out_channels:
                            y = x[:, : self.out_channels]
                        else:
                            pad = self.out_channels - x.shape[1]
                            y = np.concatenate([x] + [x[:, :1]] * pad, axis=1)

                    with self._eq_lock:
                        eq_on = self.eq_enabled
                        preamp_db = self.eq_preamp_db
                        filters = self._geq_filters

                    if eq_on:
                        pre = float(10 ** (preamp_db / 20.0))
                        y *= pre
                        for flt in filters:
                            flt.process_inplace(y)

                    # Record POST-EQ, PRE-volume (so volume slider doesn't change recording level)
                    try:
                        if self.recorder.is_recording():
                            self.recorder.enqueue_float32(y)
                    except Exception:
                        pass

                    if self.muted:
                        outdata[:] = 0
                    else:
                        outdata[:] = y * float(self.volume)

                self.stream = sd.Stream(
                    device=(input_dev, output_dev),
                    channels=(self.in_channels, self.out_channels),
                    samplerate=self.samplerate,
                    blocksize=self.blocksize,
                    dtype="float32",
                    callback=callback,
                )
                self.stream.start()
                return

            except Exception as e:
                last_err = e
                self.stream = None

        raise RuntimeError(f"Could not start duplex audio stream. Last error: {last_err}")

    def stop(self):
        try:
            self.recorder.stop()
        except Exception:
            pass

        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
        self.stream = None
# ===== CHUNK 2 / 2 =====

# -----------------------------
# Main Window
# -----------------------------

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        icon_path = resource_path("vinyl.ico")
        if os.path.exists(icon_path):
            ico = QIcon(icon_path)
            app = QApplication.instance()
            if app is not None:
                app.setWindowIcon(ico)
            self.setWindowIcon(ico)

        self.setWindowTitle(f"VinylPlayer v{VERSION}")
        self.engine = AudioEngine()
        self.settings = load_settings()

        self._restore_window_placement()

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)

        # ✅ THIS is what centers the content inside the scroll area
        self.scroll.setAlignment(Qt.AlignHCenter | Qt.AlignTop)

        root.addWidget(self.scroll, 1)

        self.page = QWidget()

        self.scroll.setWidget(self.page)

        page_layout = QVBoxLayout(self.page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(12)
        page_layout.setAlignment(Qt.AlignTop)
        
        def add_step(widget: QWidget):
            widget.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
            row = QHBoxLayout()
            row.addStretch(1)
            row.addWidget(widget, 0, Qt.AlignHCenter)
            row.addStretch(1)
            page_layout.addLayout(row)

        # (optional but helps keep everything “stacked” nicely at the top)
        page_layout.setAlignment(Qt.AlignTop)
        
        # ---- AUDIO IN/OUT ----
        self.sec_dev = CollapsibleBox("AUDIO IN / OUT", start_open=False)
        dev_body = self.sec_dev.body_layout()

        dev_box = QGroupBox("Devices")
        dev_layout = QVBoxLayout(dev_box)

        self.in_combo = QComboBox()
        self.out_combo = QComboBox()
        self.refresh_btn = QPushButton("Refresh devices")

        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Audio IN:"))
        r1.addWidget(self.in_combo, 1)
        dev_layout.addLayout(r1)

        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Audio OUT:"))
        r2.addWidget(self.out_combo, 1)
        dev_layout.addLayout(r2)

        dev_layout.addWidget(self.refresh_btn)
        dev_body.addWidget(dev_box)
        add_step(self.sec_dev)

        # ---- INPUT LEVEL ----
        self.sec_meter = CollapsibleBox("INPUT LEVEL", start_open=False)
        meter_body = self.sec_meter.body_layout()

        meter_box = QGroupBox("Input Level")
        meter_layout = QVBoxLayout(meter_box)

        meter_top = QHBoxLayout()
        meter_top.addWidget(QLabel("Style:"))
        self.meter_style = QComboBox()
        self.meter_style.addItems(LevelMeter.MODES)
        self.meter_style.setCurrentText(self.settings.get("meter_style", "EQ Bars"))
        meter_top.addWidget(self.meter_style)
        meter_top.addStretch(1)
        meter_layout.addLayout(meter_top)

        self.meter = LevelMeter()
        self.meter.set_mode(self.meter_style.currentText())
        meter_layout.addWidget(self.meter)

        meter_body.addWidget(meter_box)
        add_step(self.sec_meter)

        # ---- MONITOR CONTROLS ----
        self.sec_ctrl = CollapsibleBox("MONITOR CONTROLS", start_open=False)
        ctrl_body = self.sec_ctrl.body_layout()

        ctrl_box = QGroupBox("Controls")
        ctrl_layout = QVBoxLayout(ctrl_box)

        vol_row = QHBoxLayout()
        self.vol_slider = QSlider(Qt.Horizontal)
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(int(self.settings.get("volume", 80)))
        self.mute_btn = QPushButton("Mute")
        vol_row.addWidget(QLabel("Volume:"))
        vol_row.addWidget(self.vol_slider, 1)
        vol_row.addWidget(self.mute_btn)
        ctrl_layout.addLayout(vol_row)

        self.status = QLabel("Status: stopped")
        ctrl_layout.addWidget(self.status)

        ctrl_body.addWidget(ctrl_box)
        add_step(self.sec_ctrl)

        # ---- EQ ----
        self.sec_eq = CollapsibleBox("EQ", start_open=False)
        eq_body = self.sec_eq.body_layout()

        eq_box = QGroupBox("Graphic EQ")
        eq_layout = QVBoxLayout(eq_box)

        top_row = QHBoxLayout()
        self.eq_enable = QCheckBox("Enable EQ")
        self.eq_enable.setChecked(bool(self.settings.get("geq_enabled", True)))
        top_row.addWidget(self.eq_enable)
        top_row.addStretch(1)

        top_row.addWidget(QLabel("Preset:"))
        self.eq_presets = QComboBox()
        self.eq_presets.addItems(["Flat", "Bass Boost", "Bass & Treble", "Vocal", "Treble Boost"])
        top_row.addWidget(self.eq_presets)

        self.eq_reset_btn = QPushButton("Reset")
        top_row.addWidget(self.eq_reset_btn)
        eq_layout.addLayout(top_row)

        pre_row = QHBoxLayout()
        pre_row.addWidget(QLabel("Preamp:"))
        self.preamp_slider = QSlider(Qt.Horizontal)
        self.preamp_slider.setRange(-12, 0)
        self.preamp_slider.setValue(int(self.settings.get("geq_preamp_db", 0)))
        self.preamp_val = QLabel(f"{self.preamp_slider.value()} dB")
        pre_row.addWidget(self.preamp_slider, 1)
        pre_row.addWidget(self.preamp_val)
        eq_layout.addLayout(pre_row)

        bands_row = QHBoxLayout()
        bands_row.setSpacing(16)

        saved_bands = self.settings.get("geq_band_db", [0] * 10)
        if not isinstance(saved_bands, list) or len(saved_bands) != 10:
            saved_bands = [0] * 10

        self.band_sliders = []
        self.band_value_labels = []

        for i, lab in enumerate(GEQ_LABELS):
            col = QVBoxLayout()
            col.setSpacing(8)

            vlab = QLabel(f"{int(saved_bands[i])} dB")
            vlab.setAlignment(Qt.AlignCenter)

            s = QSlider(Qt.Vertical)
            s.setRange(-20, 20)
            s.setValue(int(saved_bands[i]))
            s.setTickPosition(QSlider.NoTicks)
            s.setMinimumHeight(260)
            s.setMinimumWidth(34)

            flab = QLabel(lab)
            flab.setAlignment(Qt.AlignCenter)

            col.addWidget(vlab)
            col.addWidget(s, 1)
            col.addWidget(flab)

            bands_row.addLayout(col)
            self.band_sliders.append(s)
            self.band_value_labels.append(vlab)

        eq_layout.addLayout(bands_row)
        eq_body.addWidget(eq_box)
        add_step(self.sec_eq)

        # ---- RECORD ----
        self.sec_rec = CollapsibleBox("RECORD", start_open=False)
        rec_body = self.sec_rec.body_layout()

        rec_box = QGroupBox("WAV Recorder")
        rec_layout = QVBoxLayout(rec_box)

        path_row = QHBoxLayout()
        self.rec_path = QLineEdit()
        self.rec_path.setPlaceholderText("Choose output .wav file…")
        self.rec_path.setText(self.settings.get("record_path", ""))
        self.rec_browse = QPushButton("Browse…")
        path_row.addWidget(self.rec_path, 1)
        path_row.addWidget(self.rec_browse)
        rec_layout.addLayout(path_row)

        btn_row = QHBoxLayout()
        self.rec_start_btn = QPushButton("● Start")
        self.rec_stop_btn = QPushButton("■ Stop")
        self.rec_stop_btn.setEnabled(False)
        btn_row.addWidget(self.rec_start_btn, 1)
        btn_row.addWidget(self.rec_stop_btn, 1)
        rec_layout.addLayout(btn_row)

        info_row = QHBoxLayout()
        self.rec_status = QLabel("Not recording.")
        self.rec_status.setStyleSheet("color: #b7b7b7;")
        self.rec_time = QLabel("0.0s")
        self.rec_time.setStyleSheet("font-weight: 600;")
        self.rec_size = QLabel("0.0 MB")
        self.rec_size.setStyleSheet("color: #b7b7b7;")
        self.rec_drop = QLabel("Drops: 0")
        self.rec_drop.setStyleSheet("color: #b7b7b7;")

        info_row.addWidget(self.rec_status, 1)
        info_row.addWidget(QLabel("Time:"))
        info_row.addWidget(self.rec_time)
        info_row.addSpacing(12)
        info_row.addWidget(QLabel("Size:"))
        info_row.addWidget(self.rec_size)
        info_row.addSpacing(12)
        info_row.addWidget(self.rec_drop)
        rec_layout.addLayout(info_row)

        rec_body.addWidget(rec_box)
        add_step(self.sec_rec)

        # ---- RUN ON WINDOWS STARTUP ----
        self.sec_startup = CollapsibleBox("RUN ON WINDOWS STARTUP", start_open=False)
        startup_body = self.sec_startup.body_layout()

        startup_box = QGroupBox("Startup")
        startup_layout = QVBoxLayout(startup_box)

        self.startup_checkbox = QCheckBox("Run on Windows startup")
        if is_windows():
            self.startup_checkbox.setChecked(is_startup_enabled())
        else:
            self.startup_checkbox.setChecked(False)
            self.startup_checkbox.setEnabled(False)
            self.startup_checkbox.setToolTip("Startup toggle is Windows-only.")
        startup_layout.addWidget(self.startup_checkbox)

        startup_body.addWidget(startup_box)
        add_step(self.sec_startup)

        footer = QHBoxLayout()
        footer.addStretch(1)
        self.version_label = QLabel(f"VinylPlayer v{VERSION}")
        self.version_label.setStyleSheet("color: #9a9a9a;")
        footer.addWidget(self.version_label)
        page_layout.addLayout(footer)
        page_layout.addStretch(1)

        # ---------------- Signals ----------------
        self.refresh_btn.clicked.connect(self.populate_devices)
        self.in_combo.currentIndexChanged.connect(self.on_device_change)
        self.out_combo.currentIndexChanged.connect(self.on_device_change)

        self.vol_slider.valueChanged.connect(self.on_volume_change)
        self.mute_btn.clicked.connect(self.on_mute_toggle)

        self.meter_style.currentTextChanged.connect(self.on_meter_style_change)
        self.startup_checkbox.toggled.connect(self.on_startup_toggled)

        self.rec_browse.clicked.connect(self.on_browse_record_path)
        self.rec_start_btn.clicked.connect(self.on_start_recording)
        self.rec_stop_btn.clicked.connect(self.on_stop_recording)

        self.eq_enable.toggled.connect(self.on_geq_change)
        self.preamp_slider.valueChanged.connect(self.on_geq_change)
        self.eq_reset_btn.clicked.connect(self.on_geq_reset)
        self.eq_presets.currentIndexChanged.connect(self.on_geq_preset)
        for s in self.band_sliders:
            s.valueChanged.connect(self.on_geq_change)

        # ---------------- Timers ----------------
        self.meter_timer = QTimer(self)
        self.meter_timer.setInterval(50)
        self.meter_timer.timeout.connect(self.update_meter)
        self.meter_timer.start()

        self.rec_timer = QTimer(self)
        self.rec_timer.setInterval(250)
        self.rec_timer.timeout.connect(self.update_record_status)
        self.rec_timer.start()

        # ---------------- Init ----------------
        self.populate_devices()

        self.engine.muted = bool(self.settings.get("muted", False))
        self.mute_btn.setText("Unmute" if self.engine.muted else "Mute")

        self.eq_presets.setCurrentIndex(0)
        self.push_geq_to_engine()

        QTimer.singleShot(1200, self.autostart_if_possible)

    # -------- Recording UI --------
    def on_browse_record_path(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save WAV", "recording.wav", "WAV Files (*.wav)")
        if path:
            if not path.lower().endswith(".wav"):
                path += ".wav"
            self.rec_path.setText(path)
            self.settings["record_path"] = path
            save_settings(self.settings)

    def on_start_recording(self):
        if self.engine.stream is None:
            QMessageBox.warning(self, "Not playing", "Start audio monitoring first (open Devices and select IN/OUT).")
            return

        path = self.rec_path.text().strip()
        if not path:
            QMessageBox.warning(self, "No file", "Choose a .wav file path first.")
            return

        try:
            self.engine.recorder.start(path, samplerate=self.engine.samplerate, channels=self.engine.out_channels)
            self.rec_start_btn.setEnabled(False)
            self.rec_stop_btn.setEnabled(True)
            self.rec_status.setText("Recording…")
        except Exception as e:
            QMessageBox.critical(self, "Record error", str(e))

    def on_stop_recording(self):
        try:
            self.engine.recorder.stop()
        except Exception:
            pass
        self.rec_start_btn.setEnabled(True)
        self.rec_stop_btn.setEnabled(False)
        self.rec_status.setText("Stopped.")

    def update_record_status(self):
        if not hasattr(self, "rec_status"):
            return

        if self.engine.recorder.is_recording():
            frames = self.engine.recorder.frames_written()
            secs = frames / float(self.engine.samplerate or 48000)
            ch = int(self.engine.out_channels or 2)
            bytes_written = int(frames * ch * 2)  # int16
            mb = bytes_written / (1024 * 1024)

            self.rec_status.setText("Recording…")
            self.rec_time.setText(f"{secs:0.1f}s")
            self.rec_size.setText(f"{mb:0.1f} MB")
            self.rec_drop.setText(f"Drops: {self.engine.recorder.dropped_blocks()}")
        else:
            if self.rec_stop_btn.isEnabled():
                self.rec_start_btn.setEnabled(True)
                self.rec_stop_btn.setEnabled(False)

    # -------- Window placement (same monitor) --------
    def _restore_window_placement(self):
        """
        Restore window size/position AND the monitor it was on.
        Works like browsers: wherever you last closed it, it reopens there.
        """
        geo_hex = self.settings.get("window_geometry_hex")
        if geo_hex:
            try:
                self.restoreGeometry(QByteArray.fromHex(geo_hex.encode("ascii")))
            except Exception:
                pass

        screen_name = self.settings.get("window_screen_name")
        if screen_name:
            screens = QApplication.screens()
            target = None
            for s in screens:
                try:
                    if s.name() == screen_name:
                        target = s
                        break
                except Exception:
                    pass

            if target is not None:
                try:
                    fg = self.frameGeometry()
                    center = fg.center()
                    if not target.geometry().contains(center):
                        self.move(target.availableGeometry().topLeft())
                except Exception:
                    pass

        try:
            fg = self.frameGeometry()
            any_screen_ok = any(s.availableGeometry().intersects(fg) for s in QApplication.screens())
            if not any_screen_ok:
                self.move(QApplication.primaryScreen().availableGeometry().topLeft())
        except Exception:
            pass

    def _save_window_placement(self):
        try:
            self.settings["window_geometry_hex"] = bytes(self.saveGeometry().toHex()).decode("ascii")
        except Exception:
            pass
        try:
            scr = self.screen()
            if scr is not None:
                self.settings["window_screen_name"] = scr.name()
        except Exception:
            pass

    # -------- Device refresh that actually updates (Windows/PortAudio) --------
    def _force_device_refresh(self):
        """
        PortAudio on Windows can keep stale device lists unless re-initialized.
        This forces a refresh without restarting your app.
        """
        try:
            if hasattr(sd, "_terminate"):
                sd._terminate()
            if hasattr(sd, "_initialize"):
                sd._initialize()
        except Exception:
            pass

    # -------- Stable restore helper --------
    def _restore_combo(self, combo: QComboBox, name_key: str, hostapi_key: str):
        wanted_name = (self.settings.get(name_key) or "").strip().lower()
        wanted_api = (self.settings.get(hostapi_key) or "").strip().lower()
        if not wanted_name:
            return

        for i in range(combo.count()):
            d = combo.itemData(i)
            if isinstance(d, dict):
                if d.get("name", "").lower() == wanted_name and d.get("hostapi", "").lower() == wanted_api:
                    combo.setCurrentIndex(i)
                    return

        for i in range(combo.count()):
            d = combo.itemData(i)
            if isinstance(d, dict) and d.get("name", "").lower() == wanted_name:
                combo.setCurrentIndex(i)
                return

    # -------- Devices --------
    def populate_devices(self):
        was_running = self.engine.stream is not None
        if was_running:
            self.engine.stop()

        self._force_device_refresh()

        self.in_combo.blockSignals(True)
        self.out_combo.blockSignals(True)

        prev_in_name = self.settings.get("input_device_name")
        prev_in_api = self.settings.get("input_device_hostapi")
        prev_out_name = self.settings.get("output_device_name")
        prev_out_api = self.settings.get("output_device_hostapi")

        self.in_combo.clear()
        self.out_combo.clear()

        devices = list_devices_with_hostapi()

        if is_windows():
            wasapi = [d for d in devices if "WASAPI" in str(d[6]).upper()]
            if wasapi:
                devices = wasapi

        hide_substrings = [
            "microsoft sound mapper",
            "primary sound capture driver",
            "primary sound driver",
        ]

        seen_in = set()
        seen_out = set()

        for idx, name, in_ch, out_ch, fs, hostapi_idx, hostapi_name in devices:
            clean = " ".join(str(name).split())
            key = clean.lower()

            if any(s in key for s in hide_substrings):
                continue

            if in_ch > 0 and key not in seen_in:
                seen_in.add(key)
                self.in_combo.addItem(clean, {"id": idx, "name": clean, "hostapi": hostapi_name})

            if out_ch > 0 and key not in seen_out:
                seen_out.add(key)
                self.out_combo.addItem(clean, {"id": idx, "name": clean, "hostapi": hostapi_name})

        if prev_in_name is not None:
            self.settings["input_device_name"] = prev_in_name
        if prev_in_api is not None:
            self.settings["input_device_hostapi"] = prev_in_api
        if prev_out_name is not None:
            self.settings["output_device_name"] = prev_out_name
        if prev_out_api is not None:
            self.settings["output_device_hostapi"] = prev_out_api

        self._restore_combo(self.in_combo, "input_device_name", "input_device_hostapi")
        self._restore_combo(self.out_combo, "output_device_name", "output_device_hostapi")

        self.in_combo.blockSignals(False)
        self.out_combo.blockSignals(False)

        if was_running and self.in_combo.count() and self.out_combo.count():
            self.start_audio()

    def autostart_if_possible(self):
        if self.in_combo.count() == 0 or self.out_combo.count() == 0:
            QMessageBox.warning(self, "No devices", "No input/output devices found.")
            return
        self.start_audio()

    def start_audio(self):
        in_data = self.in_combo.currentData()
        out_data = self.out_combo.currentData()

        if in_data is None or out_data is None:
            return

        in_dev = in_data["id"] if isinstance(in_data, dict) else in_data
        out_dev = out_data["id"] if isinstance(out_data, dict) else out_data

        self.engine.volume = self.vol_slider.value() / 100.0

        try:
            self.engine.start(in_dev, out_dev)
            self.status.setText(
                f"Status: playing (auto) • {self.engine.samplerate} Hz • "
                f"IN {self.engine.in_channels}ch → OUT {self.engine.out_channels}ch • "
                f"block {self.engine.blocksize}"
            )

            self.settings["input_device_name"] = in_data.get("name") if isinstance(in_data, dict) else None
            self.settings["input_device_hostapi"] = in_data.get("hostapi") if isinstance(in_data, dict) else None
            self.settings["output_device_name"] = out_data.get("name") if isinstance(out_data, dict) else None
            self.settings["output_device_hostapi"] = out_data.get("hostapi") if isinstance(out_data, dict) else None

            self.settings["volume"] = self.vol_slider.value()
            self.settings["muted"] = self.engine.muted
            self.settings["meter_style"] = self.meter_style.currentText()
            save_settings(self.settings)

        except Exception as e:
            self.status.setText("Status: error")
            QMessageBox.critical(self, "Audio error", "Could not start audio.\n\nDetails:\n" + str(e))

    def on_device_change(self):
        self.start_audio()

    # -------- Volume/Mute --------
    def on_volume_change(self, v):
        self.engine.volume = v / 100.0
        self.settings["volume"] = v
        save_settings(self.settings)

    def on_mute_toggle(self):
        self.engine.muted = not self.engine.muted
        self.settings["muted"] = self.engine.muted
        save_settings(self.settings)
        self.mute_btn.setText("Unmute" if self.engine.muted else "Mute")

    # -------- Meter --------
    def update_meter(self):
        peak = self.engine.get_input_peak()
        self.meter.push_peak(peak)

    def on_meter_style_change(self, mode: str):
        self.meter.set_mode(mode)
        self.settings["meter_style"] = mode
        save_settings(self.settings)

    # -------- Startup --------
    def on_startup_toggled(self, checked: bool):
        if not is_windows():
            return
        try:
            set_startup_enabled(bool(checked))
        except Exception as e:
            QMessageBox.critical(self, "Startup error", str(e))
            self.startup_checkbox.blockSignals(True)
            self.startup_checkbox.setChecked(is_startup_enabled())
            self.startup_checkbox.blockSignals(False)

    # -------- EQ --------
    def push_geq_to_engine(self):
        enabled = bool(self.eq_enable.isChecked())
        pre = int(self.preamp_slider.value())
        bands = [int(s.value()) for s in self.band_sliders]

        self.preamp_val.setText(f"{pre} dB")
        for vlab, s in zip(self.band_value_labels, self.band_sliders):
            vlab.setText(f"{s.value()} dB")

        self.engine.set_graphic_eq(enabled, pre, bands)

        self.settings["geq_enabled"] = enabled
        self.settings["geq_preamp_db"] = pre
        self.settings["geq_band_db"] = bands
        save_settings(self.settings)

    def on_geq_change(self, *_):
        self.push_geq_to_engine()

    def on_geq_reset(self):
        self.preamp_slider.setValue(0)
        for s in self.band_sliders:
            s.setValue(0)
        self.eq_presets.setCurrentIndex(0)
        self.push_geq_to_engine()

    def on_geq_preset(self, idx: int):
        presets = {
            0: {"pre": 0, "bands": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
            1: {"pre": -3, "bands": [6, 5, 4, 2, 0, 0, 0, 0, 0, 0]},
            2: {"pre": -4, "bands": [4, 3, 2, 0, 0, 0, 0, 2, 3, 4]},
            3: {"pre": -3, "bands": [-2, -1, 0, 2, 4, 4, 3, 1, 0, -1]},
            4: {"pre": -4, "bands": [0, 0, 0, 0, 0, 2, 4, 6, 6, 6]},
        }
        p = presets.get(idx, presets[0])
        self.preamp_slider.setValue(int(p["pre"]))
        for s, v in zip(self.band_sliders, p["bands"]):
            s.setValue(int(v))
        self.push_geq_to_engine()

    def closeEvent(self, event):
        self._save_window_placement()
        save_settings(self.settings)
        self.engine.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    set_windows_appusermodelid("C.Howe.VinylPlayer")
    app = QApplication([])
    app.setStyle("Fusion")  # stops Windows theme colouring bleeding in
    app.setStyleSheet(APP_STYLE)
    w = MainWindow()
    if not w.settings.get("window_geometry_hex"):
        w.resize(600, 375)   # default first-run size
    w.show()
    app.exec()
