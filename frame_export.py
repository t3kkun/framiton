"""Frame range resolution, export I/O, and provenance for Framiton."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Callable, Sequence

import cv2


class ExportError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResolvedRange:
    mode: str
    requested_start: int | str
    requested_end: int | str
    start_frame: int
    end_frame: int
    pts_start_seconds: float | None = None
    pts_end_seconds: float | None = None

    @property
    def total_frames(self) -> int:
        return self.end_frame - self.start_frame + 1


@dataclass
class ExportResult:
    output_directory: str
    total_selected: int
    exported: int = 0
    skipped: int = 0
    failed: int = 0
    cancelled: bool = False
    metadata_saved: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def export_directory_for(source_path: Path) -> Path:
    source_path = Path(source_path)
    return source_path.with_name(f"{source_path.stem}_exported_frames")


_TIME_RE = re.compile(r"^(\d+):(\d{2}):(\d{2})(?:\.(\d{1,6}))?$")


def parse_timestamp(value: str) -> float:
    match = _TIME_RE.fullmatch(value.strip())
    if not match:
        raise ValueError("時刻は HH:MM:SS または HH:MM:SS.ffffff 形式で入力してください")
    hours, minutes, seconds = map(int, match.group(1, 2, 3))
    if minutes >= 60 or seconds >= 60:
        raise ValueError("分と秒は00〜59で入力してください")
    fraction = float(f"0.{match.group(4)}") if match.group(4) else 0.0
    return hours * 3600 + minutes * 60 + seconds + fraction


def resolve_frame_range(start: int, end: int, frame_count: int,
                        pts_times: Sequence[float | None] | None = None) -> ResolvedRange:
    if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
        raise ValueError("フレーム範囲は整数で指定してください")
    if frame_count <= 0:
        raise ValueError("動画のフレーム数が不明です")
    if start < 0 or end < 0 or start > end:
        raise ValueError("StartとEndは0以上で、StartはEnd以下にしてください")
    if end >= frame_count:
        raise ValueError(f"Endは最終フレーム {frame_count - 1} 以下にしてください")
    pts_start = pts_end = None
    if pts_times is not None and len(pts_times) == frame_count:
        first, last = pts_times[start], pts_times[end]
        if first is not None and last is not None:
            pts_start, pts_end = float(first), float(last)
    return ResolvedRange("frame", start, end, start, end, pts_start, pts_end)


def resolve_time_range(
    start: str,
    end: str,
    pts_times: Sequence[float | None] | None,
    frame_count: int,
    mapping_reliable: bool,
    duration_seconds: float | None,
) -> ResolvedRange:
    start_seconds, end_seconds = parse_timestamp(start), parse_timestamp(end)
    if start_seconds > end_seconds:
        raise ValueError("StartはEnd以下にしてください")
    if not mapping_reliable or pts_times is None:
        raise ValueError("信頼できるPTSと元フレームの対応がありません。時刻範囲は解決できません")
    if len(pts_times) != frame_count or any(value is None or not math.isfinite(float(value)) for value in pts_times):
        raise ValueError("PTS件数または値が元動画のフレーム列と一致しません")
    numeric_pts = [float(value) for value in pts_times]
    if any(b < a for a, b in zip(numeric_pts, numeric_pts[1:])):
        raise ValueError("PTSが非単調のため時刻範囲を安全に解決できません")
    if duration_seconds is None or not math.isfinite(duration_seconds):
        raise ValueError("動画の長さを確定できないため時刻範囲を検証できません")
    if start_seconds < 0 or end_seconds > duration_seconds:
        raise ValueError(f"時刻範囲は動画の長さ 00:00:{duration_seconds:.3f} 秒以内にしてください")

    origin = numeric_pts[0]
    relative_pts = [value - origin for value in numeric_pts]
    first = bisect_left(relative_pts, start_seconds)
    stop = bisect_right(relative_pts, end_seconds)
    if first >= stop:
        raise ValueError("指定した包含範囲内にフレームPTSがありません")
    return ResolvedRange(
        "time", start.strip(), end.strip(), first, stop - 1,
        numeric_pts[first], numeric_pts[stop - 1],
    )


def frame_index_at_pts(pts_times: Sequence[float], target_pts: float) -> int:
    """Return the last source frame presented at or before an absolute PTS."""
    if not pts_times or any(b < a for a, b in zip(pts_times, pts_times[1:])):
        raise ValueError("PTS sequence is empty or non-monotonic")
    return max(0, min(len(pts_times) - 1, bisect_right(pts_times, target_pts) - 1))


def _file_identity(path: Path) -> dict:
    stat = path.stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def source_metadata_for(path: Path, video) -> dict:
    path = Path(path)
    stat = path.stat()
    modified = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
    warnings = []
    if getattr(video, "pts_warning", None): warnings.append(video.pts_warning)
    if getattr(video, "vfr", None) is None and not getattr(video, "pts_warning", None):
        warnings.append("CFR/VFR classification is unknown because reliable PTS are unavailable")
    return {
        "original_filename": path.name,
        "original_path": str(path.resolve()),
        "file_size_bytes": stat.st_size,
        "file_modified_utc": modified,
        "file_identity": getattr(video, "file_identity", None) or {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
        "container": getattr(video, "container", path.suffix.lstrip(".").lower() or "unknown"),
        "codec": getattr(video, "codec", "unknown"),
        "width": getattr(video, "width", None) or "unknown",
        "height": getattr(video, "height", None) or "unknown",
        "nominal_or_average_fps": getattr(video, "fps", None) or "unknown",
        "rate_classification": "VFR" if getattr(video, "vfr", None) is True else "CFR" if getattr(video, "vfr", None) is False else "unknown",
        "duration_seconds": getattr(video, "duration_seconds", None) if getattr(video, "duration_seconds", None) is not None else "unknown",
        "duration_method": getattr(video, "duration_method", "unknown"),
        "total_source_frames": video.frame_count,
        "frame_count_method": "OpenCV CAP_PROP_FRAME_COUNT; requested frame decodes are checked during export",
        "pts_available": bool(getattr(video, "pts_available", False)),
        "pts_mapping_reliable": bool(getattr(video, "pts_mapping_reliable", False)),
        "probed_pts_frame_count": len(video.probed_pts_times) if getattr(video, "probed_pts_times", None) is not None else "unavailable",
        "warnings": warnings,
    }


def _atomic_write_metadata(path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temp_name = stream.name
            json.dump(metadata, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def _load_metadata(path: Path, source_metadata: dict) -> dict:
    if path.exists():
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ExportError(f"既存メタデータを読み込めません: {exc}") from exc
        if not isinstance(metadata, dict) or not isinstance(metadata.get("export_history"), list):
            raise ExportError("既存メタデータの形式が不正です。安全のため上書きしません")
    else:
        metadata = {"schema_version": 1, "source": {}, "export_history": []}
    metadata["schema_version"] = 1
    metadata["source"] = source_metadata
    return metadata


def _upsert_history(metadata: dict, entry: dict) -> None:
    history = metadata["export_history"]
    for index, previous in enumerate(history):
        if previous.get("operation_id") == entry["operation_id"]:
            history[index] = entry
            return
    history.append(entry)


def _encode_png(frame) -> bytes:
    ok, encoded = cv2.imencode(".png", frame)
    if not ok:
        raise ExportError("PNGエンコードに失敗しました")
    return encoded.tobytes()


def execute_export(
    source_path: Path,
    output_directory: Path,
    source_metadata: dict,
    resolved: ResolvedRange,
    operation_id: str,
    cancel_event: Event,
    progress: Callable[[int, int, int], None] | None = None,
    capture_factory=None,
    png_encoder: Callable | None = None,
) -> ExportResult:
    source_path, output_directory = Path(source_path), Path(output_directory)
    result = ExportResult(str(output_directory), resolved.total_frames)
    history_path = output_directory / "export_metadata.txt"
    entry = {
        "operation_id": operation_id,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_identity": source_metadata.get("file_identity", {}),
        "source_path": source_metadata.get("original_path", str(source_path)),
        "range_mode": resolved.mode,
        "requested_range": {
            "start": resolved.requested_start, "end": resolved.requested_end,
            "unit": "source_frame_index" if resolved.mode == "frame" else "HH:MM:SS.fraction",
            "endpoint_rule": "inclusive",
        },
        "actual_range": {
            "source_frame_start": resolved.start_frame,
            "source_frame_end": resolved.end_frame,
            "pts_start_seconds_absolute": resolved.pts_start_seconds,
            "pts_end_seconds_absolute": resolved.pts_end_seconds,
        },
        "total_frames_selected": resolved.total_frames,
        "newly_exported": 0,
        "skipped_existing": 0,
        "failed": 0,
        "cancelled": False,
        "status": "in_progress",
        "warnings": list(source_metadata.get("warnings", [])),
        "errors": [],
    }
    metadata_started = False
    capture = None
    try:
        if not source_path.is_file():
            raise ExportError(f"ソース動画が見つからないか、読み込めません: {source_path}")
        output_directory.mkdir(parents=True, exist_ok=True)
        metadata = _load_metadata(history_path, source_metadata)
        _upsert_history(metadata, entry)
        try:
            _atomic_write_metadata(history_path, metadata)
        except OSError as exc:
            raise ExportError(f"メタデータを書き込めません: {exc}") from exc
        metadata_started = True

        identity = source_metadata.get("file_identity")
        if identity and _file_identity(source_path) != identity:
            raise ExportError("動画ファイルが読み込み後に変更されました。動画を開き直してください")
        factory = capture_factory or cv2.VideoCapture
        capture = factory(str(source_path))
        if not capture.isOpened():
            raise ExportError("エクスポート用の動画デコーダを開けません")
        capture.set(cv2.CAP_PROP_POS_FRAMES, resolved.start_frame)
        encode = png_encoder or _encode_png
        for offset, source_index in enumerate(range(resolved.start_frame, resolved.end_frame + 1), 1):
            if cancel_event.is_set():
                result.cancelled = True
                break
            if identity and _file_identity(source_path) != identity:
                result.errors.append("動画ファイルがエクスポート中に変更されました。以降のフレームを停止しました")
                result.failed += resolved.total_frames - offset + 1
                break
            destination = output_directory / f"{source_index}.png"
            already_exists = destination.exists()
            ok, frame = capture.read()
            if not ok:
                result.failed += 1
                result.errors.append(f"元動画のフレーム {source_index} をデコードできません")
            elif already_exists:
                result.skipped += 1
            else:
                try:
                    payload = encode(frame)
                    with destination.open("xb") as stream:
                        stream.write(payload)
                        stream.flush()
                except FileExistsError:
                    result.skipped += 1
                except Exception as exc:
                    result.failed += 1
                    result.errors.append(f"{destination.name}: {exc}")
                    try: destination.unlink(missing_ok=True)
                    except OSError: pass
                else:
                    result.exported += 1
            if progress: progress(offset, resolved.total_frames, source_index)
        if not result.cancelled and not result.errors and result.exported + result.skipped < resolved.total_frames:
            result.errors.append("選択された全フレームが処理されませんでした")
    except Exception as exc:
        result.errors.append(str(exc))
        result.failed = max(result.failed, result.total_selected - result.exported - result.skipped)
    finally:
        if capture is not None:
            capture.release()
        if metadata_started:
            entry.update({
                "newly_exported": result.exported,
                "skipped_existing": result.skipped,
                "failed": result.failed,
                "cancelled": result.cancelled,
                "status": "cancelled" if result.cancelled else "failed" if result.errors and not result.exported else "partial" if result.errors else "completed",
                "warnings": result.warnings,
                "errors": result.errors,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            })
            try:
                metadata = _load_metadata(history_path, source_metadata)
                _upsert_history(metadata, entry)
                _atomic_write_metadata(history_path, metadata)
                result.metadata_saved = True
            except Exception as exc:
                result.metadata_saved = False
                result.errors.append(f"メタデータを保存できませんでした: {exc}")
    return result
