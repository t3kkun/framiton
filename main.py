from __future__ import annotations

import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
    QFrame, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

APP_DIR = Path(__file__).parent
CHARACTER_FILE = APP_DIR / "characters.json"
CSV_FIELDS = ["id", "tag", "character", "begin", "end"]


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
        self.vfr = self._detect_vfr()

    def _detect_vfr(self) -> bool:
        """Use ffprobe frame timestamps when available; tolerate missing ffprobe."""
        try:
            result = subprocess.run([
                "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
                "-show_entries", "frame=best_effort_timestamp_time", "-of", "csv=p=0",
                str(self.path),
            ], capture_output=True, text=True, timeout=30, check=True)
            times = [float(line.strip()) for line in result.stdout.splitlines() if line.strip() and line.strip() != "N/A"]
            if len(times) < 3:
                return False
            deltas = [b - a for a, b in zip(times, times[1:]) if b > a]
            expected = 1.0 / self.fps
            return bool(deltas) and max(abs(delta - expected) for delta in deltas) > max(0.0005, expected * 0.02)
        except (OSError, ValueError, subprocess.SubprocessError):
            return False

    def frame(self, index: int):
        index = max(0, min(self.frame_count - 1, index))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = self.capture.read()
        if not ok:
            return None
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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

    def set_data(self, annotations: list[Annotation], total: int):
        self.annotations, self.total = annotations, max(1, total)
        self.update()

    def paintEvent(self, _event):
        from PySide6.QtGui import QPainter, QColor
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#202632"))
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

    def mousePressEvent(self, event):
        if not self.annotations:
            return
        frame = int(event.position().x() / max(1, self.width()) * self.total)
        matches = [a for a in self.annotations if a.begin <= frame <= a.end]
        if matches:
            self.selected.emit(matches[-1].id)


class TagDialog(QDialog):
    def __init__(self, characters: list[str], annotation: Annotation | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("区間タグ")
        layout = QFormLayout(self)
        self.tag = QComboBox()
        self.tag.addItem("指文字", "fingerspelling")
        self.tag.addItem("Transition", "transition")
        self.character = QComboBox()
        self.character.setEditable(True)
        self.character.addItems(characters)
        self.begin = QSpinBox(); self.begin.setRange(0, 2_147_483_647)
        self.end = QSpinBox(); self.end.setRange(0, 2_147_483_647)
        layout.addRow("種類", self.tag); layout.addRow("指文字", self.character)
        layout.addRow("開始フレーム", self.begin); layout.addRow("終了フレーム", self.end)
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
        seekrow = QHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal); self.slider.setRange(0, 0); self.slider.valueChanged.connect(self.seek)
        seekrow.addWidget(self.slider, 1)
        self.frame_label = QLabel("Frame: —"); self.time_label = QLabel("Time: —"); self.meta_label = QLabel("FPS: — | Frames: —")
        for label in (self.frame_label, self.time_label, self.meta_label): seekrow.addWidget(label)
        outer.addLayout(seekrow)
        controls = QHBoxLayout()
        self.back = QPushButton("⏮ 1フレーム"); self.back.clicked.connect(lambda: self.step(-1))
        self.play = QPushButton("▶ 再生"); self.play.clicked.connect(self.toggle_play)
        self.forward = QPushButton("1フレーム ⏭"); self.forward.clicked.connect(lambda: self.step(1))
        self.start_button = QPushButton("タグスタート"); self.start_button.clicked.connect(self.mark_start)
        self.start_status = QLabel("開始位置未指定")
        self.end_button = QPushButton("タグゴール"); self.end_button.clicked.connect(self.mark_end)
        for widget in (self.back, self.play, self.forward, self.start_button, self.start_status, self.end_button): controls.addWidget(widget)
        controls.addStretch(); outer.addLayout(controls)
        self.listing = QLabel("アノテーション 0 区間")
        outer.addWidget(self.listing)
        self.hint = QLabel("Space: 再生/一時停止　← →: 1フレーム移動　タイムラインの区間をクリックして編集")
        self.hint.setStyleSheet("color:#748093"); outer.addWidget(self.hint)
        for key, fn in [("Space", self.toggle_play), ("Left", lambda: self.step(-1)), ("Right", lambda: self.step(1))]:
            QShortcut(QKeySequence(key), self, activated=fn)
        self._enable_video_controls(False)

    def _enable_video_controls(self, enabled: bool):
        for widget in (self.slider, self.back, self.play, self.forward, self.start_button, self.end_button): widget.setEnabled(enabled)

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "動画を開く", "", "Video files (*.mp4 *.mov *.mkv *.avi);;All files (*)")
        if not path: return
        try:
            new_video = VideoSource(Path(path))
        except Exception as exc:
            QMessageBox.critical(self, "動画を開けません", str(exc)); return
        if self.video: self.video.close()
        self.video = new_video; self.frame_index = 0; self.pending_begin = None
        self.csv_path = Path(path).with_suffix(".csv")
        self.annotations = []
        if self.csv_path.exists():
            try: self.annotations = self.read_csv(self.csv_path)
            except Exception as exc: QMessageBox.warning(self, "CSV読み込み", f"関連CSVを読み込めませんでした: {exc}")
        self.slider.setRange(0, self.video.frame_count - 1); self._enable_video_controls(True)
        warning = "  ⚠ VFRの可能性" if self.video.vfr else ""
        self.meta_label.setText(f"FPS: {self.video.fps:.3f}{warning} | Frames: {self.video.frame_count:,}")
        self.refresh()

    def open_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "CSV読み込み", "", "CSV (*.csv)")
        if not path: return
        try:
            annotations = self.read_csv(Path(path))
            if self.video and any(a.end >= self.video.frame_count for a in annotations):
                raise ValueError("区間が動画の総フレーム数を超えています")
            self.annotations, self.csv_path = annotations, Path(path)
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
            self.hint.setText(f"保存しました: {path}")
        except OSError as exc: QMessageBox.critical(self, "保存できません", str(exc))

    def seek(self, value: int):
        if self.video and value != self.frame_index:
            self.frame_index = value; self.refresh()

    def refresh(self):
        if not self.video: return
        rgb = self.video.frame(self.frame_index)
        if rgb is not None:
            h, w, _ = rgb.shape
            image = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
            self.preview.setPixmap(QPixmap.fromImage(image).scaled(self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.slider.blockSignals(True); self.slider.setValue(self.frame_index); self.slider.blockSignals(False)
        self.frame_label.setText(f"Frame: {self.frame_index:,} / {self.video.frame_count - 1:,}")
        self.time_label.setText(f"Time: {self.frame_index / self.video.fps:.3f}s")
        self.timeline.set_data(self.annotations, self.video.frame_count)
        self.listing.setText(f"アノテーション {len(self.annotations)} 区間" + (f"　|　CSV: {self.csv_path.name}" if self.csv_path else ""))

    def step(self, amount: int):
        if self.video:
            self.frame_index = max(0, min(self.video.frame_count - 1, self.frame_index + amount)); self.refresh()

    def toggle_play(self):
        if not self.video: return
        if self.timer.isActive():
            self.timer.stop(); self.play.setText("▶ 再生")
        else:
            self.timer.start(max(1, round(1000 / self.video.fps))); self.play.setText("⏸ 一時停止")

    def advance_playback(self):
        if not self.video: return
        if self.frame_index >= self.video.frame_count - 1:
            self.timer.stop(); self.play.setText("▶ 再生"); return
        self.step(1)

    def mark_start(self):
        if self.video:
            self.pending_begin = self.frame_index; self.start_status.setText(f"開始: {self.frame_index:,} frame")

    def mark_end(self):
        if self.pending_begin is None:
            QMessageBox.information(self, "タグスタート", "先にタグスタートを押してください。"); return
        dialog = TagDialog(self.characters, parent=self)
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
            self.annotations.append(Annotation(max((a.id for a in self.annotations), default=0) + 1, tag, char, begin, end))
            self.annotations.sort(key=lambda a: (a.begin, a.end, a.id)); self.refresh()
        self.pending_begin = None; self.start_status.setText("開始位置未指定")

    def select_annotation(self, annotation_id: int):
        item = next((a for a in self.annotations if a.id == annotation_id), None)
        if not item: return
        dialog = TagDialog(self.characters, item, self)
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
            if deleted: self.annotations.remove(item)
            else:
                tag = dialog.tag.currentData()
                character = dialog.character.currentText().strip() if tag == "fingerspelling" else ""
                begin, end = dialog.begin.value(), dialog.end.value()
                if end < begin or (self.video and end >= self.video.frame_count):
                    QMessageBox.warning(self, "区間", "区間は動画内に収め、終了フレームを開始フレーム以降にしてください。"); return
                if tag == "fingerspelling" and not character:
                    QMessageBox.warning(self, "指文字", "指文字を1文字入力してください。"); return
                item.tag, item.character, item.begin, item.end = tag, character, begin, end
        self.annotations.sort(key=lambda a: (a.begin, a.end, a.id)); self.refresh()

    def closeEvent(self, event):
        if self.video: self.video.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow(); window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
