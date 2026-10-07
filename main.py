from __future__ import annotations

import csv
import json
import math
import sys
import time
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

import cv2
from PySide6.QtCore import QProcess, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton, QSlider,
    QSpinBox, QVBoxLayout, QWidget,
)

APP_DIR = Path(__file__).parent
CHARACTER_FILE = APP_DIR / "characters.json"
CSV_FIELDS = ["id", "tag", "character", "begin", "end"]


class PtsProbeProcess(QProcess):
    """Asynchronous ffprobe reader so metadata probing never stalls the UI thread."""
    completed = Signal(object, object, float)

    def __init__(self, path: Path, frame_count: int, parent=None):
        super().__init__(parent)
        self.path = path
        self.frame_count = frame_count
        self.started_at = time.perf_counter()
        self.timestamps: list[float] = []
        self.pending_line = bytearray()
        self.invalid_output = False
        self.finished_once = False
        self.readyReadStandardOutput.connect(self._collect_output)
        self.finished.connect(self._on_finished)
        self.errorOccurred.connect(self._on_error)
        self.timeout = QTimer(self)
        self.timeout.setSingleShot(True)
        self.timeout.timeout.connect(self._on_timeout)

    def start_probe(self):
        self.start("ffprobe", [
            "-v", "error", "-select_streams", "v:0", "-show_frames",
            "-show_entries", "frame=best_effort_timestamp_time", "-of", "csv=p=0",
            str(self.path),
        ])
        self.timeout.start(30_000)

    def _collect_output(self):
        self.pending_line.extend(bytes(self.readAllStandardOutput()))
        lines = self.pending_line.split(b"\n")
        self.pending_line = bytearray(lines.pop())
        for line in lines: self._parse_line(line)

    def _parse_line(self, line: bytes):
        value = line.strip().split(b",", 1)[0]
        if not value or value == b"N/A": return
        try: self.timestamps.append(float(value))
        except ValueError: self.invalid_output = True

    def _on_finished(self, exit_code: int, _exit_status):
        self._collect_output()
        if self.pending_line:
            self._parse_line(bytes(self.pending_line)); self.pending_line.clear()
        timestamps = self.timestamps if exit_code == 0 and not self.invalid_output else None
        if timestamps is not None and (len(timestamps) != self.frame_count or any(b < a for a, b in zip(timestamps, timestamps[1:]))):
            timestamps = None
        self._complete(timestamps)

    def _on_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self._complete(None)

    def _on_timeout(self):
        self.kill()
        self._complete(None)

    def _complete(self, timestamps):
        if self.finished_once:
            return
        self.finished_once = True
        self.timeout.stop()
        self.completed.emit(self.path, timestamps, (time.perf_counter() - self.started_at) * 1000)


def load_characters() -> list[str]:
    try:
        values = json.loads(CHARACTER_FILE.read_text(encoding="utf-8"))
        return [str(value) for value in values if str(value).strip()]
    except (OSError, ValueError, TypeError):
        return list("あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん") + list("ぁぃぅぇぉゃゅょっがぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽ") + list("アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン") + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


@dataclass
class Annotation:
    id: int
    tag: str
    character: str
    begin: int
    end: int


class VideoSource:
    """Frame decoding and metadata, isolated from annotations for future observers."""

    def __init__(self, path: Path):
        self.path = path
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise RuntimeError(f"動画を開けません: {path}")
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.fps <= 0 or self.frame_count <= 0:
            self.capture.release()
            raise RuntimeError("動画のFPSまたは総フレーム数を取得できませんでした。")
        self.pts_times = [i / self.fps for i in range(self.frame_count)]
        self.pts_available = False
        self.vfr: bool | None = None
        self._next_frame_index = -1

    @staticmethod
    def parse_pts(output: str | bytes, expected_frames: int) -> list[float] | None:
        if isinstance(output, bytes): output = output.decode("utf-8", errors="replace")
        try:
            times = []
            for line in output.splitlines():
                value = line.strip().split(",", 1)[0]
                if value and value != "N/A": times.append(float(value))
            if len(times) != expected_frames or len(times) < 2:
                return None
            if any(b < a for a, b in zip(times, times[1:])):
                return None
            return times
        except ValueError:
            return None

    def set_presentation_times(self, timestamps: list[float]) -> bool:
        if len(timestamps) != self.frame_count or any(b < a for a, b in zip(timestamps, timestamps[1:])):
            return False
        self.pts_times = timestamps
        self.pts_available = True
        deltas = [b - a for a, b in zip(timestamps, timestamps[1:]) if b > a]
        expected = 1.0 / self.fps
        self.vfr = bool(deltas) and max(abs(delta - expected) for delta in deltas) > max(0.0005, expected * 0.02)
        return True

    def frame(self, index: int):
        index = max(0, min(self.frame_count - 1, index))
        if index != self._next_frame_index:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = self.capture.read()
        if not ok:
            return None
        self._next_frame_index = index + 1
        return image

    def close(self):
        self.capture.release()


class Timeline(QWidget):
    selected = Signal(int)
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(48)
        self.setMaximumHeight(48)
        self.annotations: list[Annotation] = []
        self.total = 1
        self.current_frame = 0
        self.annotation_layer = QPixmap()

    def set_data(self, annotations: list[Annotation], total: int):
        self.annotations, self.total = annotations, max(1, total)
        details = "\n".join(f"{a.begin}–{a.end}: {a.character if a.tag == 'fingerspelling' else 'Transition'}"
                             for a in annotations)
        self.setToolTip(details or "アノテーション区間はありません")
        self._rebuild_annotation_layer()
        self.update()

    def set_frame(self, frame: int):
        self.current_frame = max(0, min(self.total - 1, frame))
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rebuild_annotation_layer()

    def _rebuild_annotation_layer(self):
        if self.width() <= 0 or self.height() <= 0:
            self.annotation_layer = QPixmap()
            return
        from PySide6.QtGui import QPainter, QColor
        layer = QPixmap(self.size())
        layer.fill(QColor("#202632"))
        painter = QPainter(layer)
        w, h = self.width(), self.height()
        for item in self.annotations:
            x1 = round(item.begin / self.total * w)
            x2 = round((item.end + 1) / self.total * w)
            color = QColor("#d69e36") if item.tag == "transition" else QColor("#3b91a6")
            painter.fillRect(x1, 5, max(3, x2 - x1), h - 10, color)
            if x2 - x1 > 28:
                painter.setPen(QColor("white"))
                painter.drawText(x1 + 3, h - 13, item.character if item.tag == "fingerspelling" else "Transition")
        painter.end()
        self.annotation_layer = layer

    def paintEvent(self, _event):
        from PySide6.QtGui import QPainter, QColor
        painter = QPainter(self)
        if not self.annotation_layer.isNull():
            painter.drawPixmap(0, 0, self.annotation_layer)
        playhead_x = round(self.current_frame / self.total * self.width())
        painter.setPen(QColor("#f4f6f8"))
        painter.drawLine(playhead_x, 0, playhead_x, self.height())
        painter.end()

    def mousePressEvent(self, event):
        if not self.annotations:
            return
        frame = int(event.position().x() / max(1, self.width()) * self.total)
        matches = [a for a in self.annotations if a.begin <= frame <= a.end]
        if matches:
            self.selected.emit(matches[-1].id)


class TagDialog(QDialog):
    def __init__(self, characters: list[str], annotation: Annotation | None = None,
                 recent: list[tuple[str, str]] | None = None, current_frame: int = 0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("区間タグ")
        layout = QFormLayout(self)
        self.tag = QComboBox()
        self.tag.addItem("指文字", "fingerspelling")
        self.tag.addItem("Transition", "transition")
        self.character = QComboBox()
        self.character.setEditable(True)
        self.character.addItems(characters)
        self.character.insertSeparator(0)
        self.character.insertItem(0, "最近使用した指文字", "")
        for char in reversed([char for tag, char in (recent or []) if tag == "fingerspelling"]):
            if char:
                self.character.insertItem(1, f"最近: {char}", char)
        self.character.setCurrentIndex(-1)
        self.recent_combo = QComboBox()
        self.recent_combo.addItem("最近のタグを選択…", None)
        for recent_tag, recent_char in (recent or []):
            label = recent_char if recent_tag == "fingerspelling" else "Transition"
            self.recent_combo.addItem(f"最近: {label}", (recent_tag, recent_char))
        self.recent_combo.setEnabled(self.recent_combo.count() > 1)
        self.recent_combo.activated.connect(self.apply_recent)
        self.begin = QSpinBox(); self.begin.setRange(0, 2_147_483_647)
        self.end = QSpinBox(); self.end.setRange(0, 2_147_483_647)
        layout.addRow("種類", self.tag); layout.addRow("指文字", self.character)
        layout.addRow("最近使用", self.recent_combo)
        layout.addRow("開始フレーム", self.begin); layout.addRow("終了フレーム", self.end)
        if annotation is not None:
            boundary = QHBoxLayout()
            set_begin = QPushButton("現在フレームを開始位置にする")
            set_end = QPushButton("現在フレームを終了位置にする")
            set_begin.clicked.connect(lambda: self.begin.setValue(current_frame))
            set_end.clicked.connect(lambda: self.end.setValue(current_frame))
            boundary.addWidget(set_begin); boundary.addWidget(set_end)
            layout.addRow("境界", boundary)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        self.tag.currentIndexChanged.connect(self._toggle_character)
        if annotation:
            self.tag.setCurrentIndex(0 if annotation.tag == "fingerspelling" else 1)
            self.character.setCurrentText(annotation.character)
            self.begin.setValue(annotation.begin); self.end.setValue(annotation.end)
        self._toggle_character()

    def _toggle_character(self):
        self.character.setEnabled(self.tag.currentData() == "fingerspelling")

    def apply_recent(self, index: int):
        value = self.recent_combo.itemData(index)
        if value is None: return
        tag, character = value
        self.tag.setCurrentIndex(0 if tag == "fingerspelling" else 1)
        if tag == "fingerspelling": self.character.setCurrentText(character)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Framiton — 指文字動画アノテーション")
        self.resize(1120, 760)
        self.video: VideoSource | None = None
        self.frame_index = 0
        self.annotations: list[Annotation] = []
        self.csv_path: Path | None = None
        self.pending_begin: int | None = None
        self.dirty = False
        self.recent_tags: list[tuple[str, str]] = []
        self.playback_started_at = 0.0
        self.playback_started_pts = 0.0
        self.pts_probe: PtsProbeProcess | None = None
        self.characters = load_characters()
        self.timer = QTimer(self); self.timer.timeout.connect(self.advance_playback)

        root = QWidget(); self.setCentralWidget(root); outer = QVBoxLayout(root)
        toolbar = QHBoxLayout()
        self.open_button = QPushButton("動画を開く"); self.open_button.clicked.connect(self.open_video)
        self.csv_button = QPushButton("CSV読み込み"); self.csv_button.clicked.connect(self.open_csv)
        self.save_button = QPushButton("保存"); self.save_button.clicked.connect(self.save_csv)
        self.save_as_button = QPushButton("名前を付けて保存"); self.save_as_button.clicked.connect(self.save_csv_as)
        for button in (self.open_button, self.csv_button, self.save_button, self.save_as_button): toolbar.addWidget(button)
        toolbar.addStretch(); outer.addLayout(toolbar)

        self.preview = QLabel("動画を開いてください")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter); self.preview.setMinimumSize(640, 400)
        self.preview.setStyleSheet("background:#11151b;color:#b7c0cc;font-size:20px")
        outer.addWidget(self.preview, 1)
        self.timeline = Timeline(); self.timeline.selected.connect(self.select_annotation); outer.addWidget(self.timeline)
        self.legend = QLabel('<span style="color:#3b91a6">■ 指文字</span>　'
                             '<span style="color:#d69e36">■ Transition</span>　タイムラインの区間をクリックすると編集')
        outer.addWidget(self.legend)
        seekrow = QHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal); self.slider.setRange(0, 0); self.slider.valueChanged.connect(self.seek)
        seekrow.addWidget(self.slider, 1)
        self.frame_label = QLabel("Frame: —"); self.time_label = QLabel("Time: —"); self.meta_label = QLabel("FPS: — | Frames: —")
        seekrow.addWidget(self.frame_label); seekrow.addWidget(self.time_label)
        outer.addLayout(seekrow)
        outer.addWidget(self.meta_label)
        controls = QHBoxLayout()
        self.back = QPushButton("⏮ 1フレーム"); self.back.clicked.connect(lambda: self.step(-1))
        self.play = QPushButton("▶ 再生"); self.play.clicked.connect(self.toggle_play)
        self.forward = QPushButton("1フレーム ⏭"); self.forward.clicked.connect(lambda: self.step(1))
        self.start_button = QPushButton("タグスタート"); self.start_button.clicked.connect(self.mark_start)
        self.start_status = QLabel("開始位置未指定")
        self.end_button = QPushButton("タグゴール"); self.end_button.clicked.connect(self.mark_end)
        self.speed = QComboBox()
        for value in (0.25, 0.5, 1.0, 1.5, 2.0): self.speed.addItem(f"{value:.2g}×", value)
        self.speed.setCurrentIndex(2); self.speed.currentIndexChanged.connect(self.update_playback_rate)
        for widget in (self.back, self.play, self.forward, QLabel("速度"), self.speed, self.start_button, self.start_status, self.end_button): controls.addWidget(widget)
        controls.addStretch(); outer.addLayout(controls)
        jump_row = QHBoxLayout()
        for title, boundary, direction in (("|◀ 前の開始", "begin", -1), ("◀ 前の終了", "end", -1),
                                           ("次の開始 ▶|", "begin", 1), ("次の終了 ▶", "end", 1)):
            button = QPushButton(title); button.clicked.connect(lambda _checked=False, b=boundary, d=direction: self.jump_boundary(b, d))
            jump_row.addWidget(button)
        jump_row.addStretch(); outer.addLayout(jump_row)
        self.listing = QLabel("アノテーション 0 区間")
        outer.addWidget(self.listing)
        self.hint = QLabel("Space: 再生/一時停止　← →: 1フレーム移動　S: タグスタート　E: タグゴール　Ctrl+S: 保存　[:前の開始　]:次の開始　Shift+[ / ]:終了位置")
        self.hint.setStyleSheet("color:#748093"); outer.addWidget(self.hint)
        for key, fn in [("Space", self.toggle_play), ("Left", lambda: self.step(-1)), ("Right", lambda: self.step(1))]:
            QShortcut(QKeySequence(key), self, activated=fn)
        for key, fn in [("S", self.mark_start), ("E", self.mark_end), ("Ctrl+S", self.save_csv),
                        ("[", lambda: self.jump_boundary("begin", -1)), ("Shift+[", lambda: self.jump_boundary("end", -1)),
                        ("]", lambda: self.jump_boundary("begin", 1)), ("Shift+]", lambda: self.jump_boundary("end", 1))]:
            QShortcut(QKeySequence(key), self, activated=fn)
        self._enable_video_controls(False)

    def _enable_video_controls(self, enabled: bool):
        for widget in (self.slider, self.back, self.play, self.forward, self.start_button, self.end_button): widget.setEnabled(enabled)

    def open_video(self):
        if self.dirty and not self.confirm_discard_or_save(): return
        path, _ = QFileDialog.getOpenFileName(self, "動画を開く", "", "Video files (*.mp4 *.mov *.mkv *.avi);;All files (*)")
        if not path: return
        try:
            new_video = VideoSource(Path(path))
        except Exception as exc:
            QMessageBox.critical(self, "動画を開けません", str(exc)); return
        if self.timer.isActive():
            self.timer.stop(); self.play.setText("▶ 再生")
        if self.video: self.video.close()
        self.video = new_video; self.frame_index = 0; self.pending_begin = None
        self.csv_path = Path(path).with_suffix(".csv")
        self.annotations = []
        if self.csv_path.exists():
            try:
                loaded_annotations = self.read_csv(self.csv_path)
                if any(a.end >= self.video.frame_count for a in loaded_annotations):
                    raise ValueError("区間が動画の総フレーム数を超えています")
                self.annotations = loaded_annotations
            except Exception as exc: QMessageBox.warning(self, "CSV読み込み", f"関連CSVを読み込めませんでした: {exc}")
        self.dirty = False
        self.slider.setRange(0, self.video.frame_count - 1); self._enable_video_controls(True)
        self.timeline.set_data(self.annotations, self.video.frame_count)
        self._update_video_status("PTS解析中 (FPSで再生)")
        self.update_window_title()
        self.refresh()
        self._start_pts_probe(self.video)

    def _start_pts_probe(self, video: VideoSource):
        if self.pts_probe and self.pts_probe.state() != QProcess.ProcessState.NotRunning:
            self.pts_probe.kill()
        probe = PtsProbeProcess(video.path, video.frame_count, self)
        self.pts_probe = probe
        probe.completed.connect(self._on_pts_probe_completed)
        probe.start_probe()

    def _on_pts_probe_completed(self, path: Path, timestamps, _duration_ms: float):
        probe = self.sender()
        if self.pts_probe is probe: self.pts_probe = None
        if probe: probe.deleteLater()
        if not self.video or path != self.video.path: return
        if timestamps is not None and self.video.set_presentation_times(timestamps):
            if self.timer.isActive():
                self.reset_playback_clock(); self.timer.start(1)
            self._update_video_status()
            self.refresh()
        else:
            self._update_video_status("判定不能 (FPSで再生)")

    def _update_video_status(self, fallback: str | None = None):
        if not self.video: return
        if fallback:
            status = fallback
        elif self.video.vfr is True:
            status = "⚠ VFR検出: FPSは平均値です (フレーム間隔が一定ではありません)"
        elif self.video.vfr is False:
            status = "CFR: 固定フレームレート"
        else:
            status = "VFR: 判定不能 (FPSで再生)"
        self.meta_label.setText(f"FPS: {self.video.fps:.3f} | Frames: {self.video.frame_count:,} | {status}")
        self.meta_label.setStyleSheet("color:#ff8d70;font-weight:bold" if self.video.vfr is True else "")

    def open_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "CSV読み込み", "", "CSV (*.csv)")
        if not path: return
        if self.dirty and not self.confirm_discard_or_save(): return
        try:
            annotations = self.read_csv(Path(path))
            if self.video and any(a.end >= self.video.frame_count for a in annotations):
                raise ValueError("区間が動画の総フレーム数を超えています")
            self.annotations, self.csv_path = annotations, Path(path)
            self.dirty = False; self.update_window_title()
            if self.video: self.timeline.set_data(self.annotations, self.video.frame_count)
            self.refresh()
        except Exception as exc: QMessageBox.critical(self, "CSVを読み込めません", str(exc))

    @staticmethod
    def read_csv(path: Path) -> list[Annotation]:
        items = []
        with path.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or not set(CSV_FIELDS).issubset(reader.fieldnames):
                raise ValueError("必要な列 id,tag,character,begin,end がありません")
            ids = set()
            for row in reader:
                item = Annotation(int(row["id"]), row["tag"], row["character"] or "", int(row["begin"]), int(row["end"]))
                if item.id in ids: raise ValueError(f"IDが重複しています: {item.id}")
                if item.tag not in ("fingerspelling", "transition"): raise ValueError(f"未知のタグ: {item.tag}")
                if item.begin < 0 or item.end < item.begin: raise ValueError(f"不正な区間: {item.id}")
                if item.tag == "transition": item.character = ""
                elif not item.character: raise ValueError(f"指文字が未設定です: {item.id}")
                ids.add(item.id); items.append(item)
        return sorted(items, key=lambda a: (a.begin, a.end, a.id))

    def save_csv(self):
        if not self.csv_path: self.save_csv_as(); return
        self.write_csv(self.csv_path)

    def save_csv_as(self):
        path, _ = QFileDialog.getSaveFileName(self, "CSVに保存", str(self.csv_path or "annotations.csv"), "CSV (*.csv)")
        if path:
            self.csv_path = Path(path if path.lower().endswith(".csv") else path + ".csv")
            self.write_csv(self.csv_path)

    def write_csv(self, path: Path):
        try:
            with path.open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS); writer.writeheader()
                for a in sorted(self.annotations, key=lambda x: (x.begin, x.end, x.id)):
                    writer.writerow({"id": a.id, "tag": a.tag, "character": a.character, "begin": a.begin, "end": a.end})
            self.dirty = False; self.update_window_title()
            self.hint.setText(f"保存しました: {path}")
        except OSError as exc: QMessageBox.critical(self, "保存できません", str(exc))

    def seek(self, value: int):
        if self.video and value != self.frame_index:
            self.frame_index = value; self.refresh()
            self.reset_playback_clock()

    def refresh(self):
        if not self.video: return
        bgr = self.video.frame(self.frame_index)
        if bgr is not None:
            h, w, _ = bgr.shape
            image = QImage(bgr.data, w, h, bgr.strides[0], QImage.Format.Format_BGR888)
            self.preview.setPixmap(QPixmap.fromImage(image).scaled(self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.slider.blockSignals(True); self.slider.setValue(self.frame_index); self.slider.blockSignals(False)
        self.frame_label.setText(f"Frame: {self.frame_index:,} / {self.video.frame_count - 1:,}")
        elapsed_pts = self.video.pts_times[self.frame_index] - self.video.pts_times[0]
        self.time_label.setText(f"Time: {elapsed_pts:.3f}s")
        self.timeline.set_frame(self.frame_index)
        self.listing.setText(f"アノテーション {len(self.annotations)} 区間" + (f"　|　CSV: {self.csv_path.name}" if self.csv_path else ""))

    def step(self, amount: int):
        if self.video:
            self.frame_index = max(0, min(self.video.frame_count - 1, self.frame_index + amount)); self.refresh()
            self.reset_playback_clock()

    def toggle_play(self):
        if not self.video: return
        if self.timer.isActive():
            self.timer.stop(); self.play.setText("▶ 再生")
        else:
            self.reset_playback_clock()
            self.timer.start(1); self.play.setText("⏸ 一時停止")

    def reset_playback_clock(self):
        if self.video:
            self.playback_started_at = time.perf_counter()
            self.playback_started_pts = self.video.pts_times[self.frame_index]

    def update_playback_rate(self):
        if self.video and self.timer.isActive():
            self.reset_playback_clock()
            self.timer.start(1)

    def advance_playback(self):
        if not self.video: return
        speed = float(self.speed.currentData())
        elapsed = time.perf_counter() - self.playback_started_at
        target_pts = self.playback_started_pts + elapsed * speed
        timestamps = self.video.pts_times
        target_index = min(self.video.frame_count - 1, max(self.frame_index, bisect_right(timestamps, target_pts) - 1))
        if target_index != self.frame_index:
            self.frame_index = target_index
            self.refresh()
        last_delta = timestamps[-1] - timestamps[-2] if len(timestamps) > 1 else 1 / self.video.fps
        media_end = timestamps[-1] + max(last_delta, 1 / self.video.fps)
        if target_pts >= media_end:
            self.timer.stop(); self.play.setText("▶ 再生"); return
        next_index = min(self.frame_index + 1, self.video.frame_count - 1)
        next_due = (timestamps[next_index] - self.playback_started_pts) / speed
        remaining = next_due - elapsed
        if self.frame_index == self.video.frame_count - 1:
            remaining = (media_end - target_pts) / speed
        self.timer.start(max(1, math.ceil(remaining * 1000)))

    def mark_start(self):
        if self.video:
            self.pending_begin = self.frame_index; self.start_status.setText(f"開始: {self.frame_index:,} frame")

    def mark_end(self):
        if self.pending_begin is None:
            QMessageBox.information(self, "タグスタート", "先にタグスタートを押してください。"); return
        dialog = TagDialog(self.characters, recent=self.recent_tags, current_frame=self.frame_index, parent=self)
        dialog.begin.setMaximum(self.video.frame_count - 1)
        dialog.end.setMaximum(self.video.frame_count - 1)
        dialog.begin.setValue(self.pending_begin); dialog.end.setValue(self.frame_index)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            begin, end = dialog.begin.value(), dialog.end.value()
            if end < begin:
                QMessageBox.warning(self, "区間", "終了フレームは開始フレーム以降にしてください。"); return
            tag = dialog.tag.currentData(); char = dialog.character.currentText().strip() if tag == "fingerspelling" else ""
            if tag == "fingerspelling" and not char:
                QMessageBox.warning(self, "指文字", "指文字を1文字入力してください。"); return
            overlaps = [a for a in self.annotations if begin <= a.end and end >= a.begin]
            if overlaps and QMessageBox.question(self, "区間の重複",
                    f"{len(overlaps)}件の既存アノテーションと重複しています。保存しますか？",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
                return
            self.annotations.append(Annotation(max((a.id for a in self.annotations), default=0) + 1, tag, char, begin, end))
            self.remember_tag(tag, char); self.set_dirty()
            self.annotations.sort(key=lambda a: (a.begin, a.end, a.id))
            self.timeline.set_data(self.annotations, self.video.frame_count)
            self.refresh()
        self.pending_begin = None; self.start_status.setText("開始位置未指定")

    def select_annotation(self, annotation_id: int):
        item = next((a for a in self.annotations if a.id == annotation_id), None)
        if not item: return
        dialog = TagDialog(self.characters, item, self.recent_tags, self.frame_index, self)
        if self.video:
            dialog.begin.setMaximum(self.video.frame_count - 1)
            dialog.end.setMaximum(self.video.frame_count - 1)
        dialog.setWindowTitle(f"区間 #{item.id} を編集")
        delete = QPushButton("削除"); dialog.layout().addWidget(delete)
        deleted = False
        def remove():
            nonlocal deleted
            answer = QMessageBox.question(dialog, "削除確認", f"区間 #{item.id} を削除しますか？")
            if answer == QMessageBox.StandardButton.Yes: deleted = True; dialog.accept()
        delete.clicked.connect(remove)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            if deleted: self.annotations.remove(item); self.set_dirty()
            else:
                tag = dialog.tag.currentData()
                character = dialog.character.currentText().strip() if tag == "fingerspelling" else ""
                begin, end = dialog.begin.value(), dialog.end.value()
                if end < begin or (self.video and end >= self.video.frame_count):
                    QMessageBox.warning(self, "区間", "区間は動画内に収め、終了フレームを開始フレーム以降にしてください。"); return
                if tag == "fingerspelling" and not character:
                    QMessageBox.warning(self, "指文字", "指文字を1文字入力してください。"); return
                if (item.tag, item.character, item.begin, item.end) != (tag, character, begin, end):
                    item.tag, item.character, item.begin, item.end = tag, character, begin, end
                    self.remember_tag(tag, character); self.set_dirty()
        self.annotations.sort(key=lambda a: (a.begin, a.end, a.id))
        self.timeline.set_data(self.annotations, self.video.frame_count if self.video else 1)
        self.refresh()

    def closeEvent(self, event):
        if self.dirty and not self.confirm_discard_or_save():
            event.ignore(); return
        if self.pts_probe and self.pts_probe.state() != QProcess.ProcessState.NotRunning:
            self.pts_probe.kill(); self.pts_probe.waitForFinished(500)
        if self.video: self.video.close()
        event.accept()

    def remember_tag(self, tag: str, character: str):
        value = (tag, character)
        self.recent_tags = [value] + [old for old in self.recent_tags if old != value]
        self.recent_tags = self.recent_tags[:8]

    def set_dirty(self):
        self.dirty = True
        self.update_window_title()

    def update_window_title(self):
        marker = " *" if self.dirty else ""
        self.setWindowTitle(f"Framiton — 指文字動画アノテーション{marker}")

    def confirm_discard_or_save(self) -> bool:
        answer = QMessageBox.question(self, "未保存の変更", "未保存の変更があります。保存しますか？",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save)
        if answer == QMessageBox.StandardButton.Cancel: return False
        if answer == QMessageBox.StandardButton.Discard: return True
        self.save_csv()
        return not self.dirty

    def jump_boundary(self, boundary: str, direction: int):
        if not self.video or not self.annotations: return
        values = sorted({getattr(a, boundary) for a in self.annotations})
        candidates = [value for value in values if value < self.frame_index] if direction < 0 else [value for value in values if value > self.frame_index]
        if candidates:
            self.frame_index = candidates[-1] if direction < 0 else candidates[0]
            self.refresh()


def main():
    app = QApplication(sys.argv)
    window = MainWindow(); window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
