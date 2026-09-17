"""visual_signal.py — 视觉信号层：画面主体（说话人）检测。

移植自 VideoHighlighter/action_recognition.py 的 SmartActionDetector + PersonTracker，
按 doubao-live-clip 项目需求重写：

**移植保留**（评分思想与跟踪机制）：
- IoUTracker：跨帧 IOU 跟踪，给每个检测目标分配稳定 track_id
- SubjectScorer 五维评分：center_prox(离中央) / size(占画面) / motion(跨帧动)
  / temporal(持续出现) / conf(检测置信)，加权取主体
- selection_history 粘性帧：主体切换不过度抖动

**按项目改造**：
- 检测器：YOLO 行人检测 → OpenCV Haar 人脸级联（多级联融合，零模型下载，CPU 可跑）
- 目标：检测"脸"而非"全身"——直播连麦/展示场景脸是核心主体
- 输出：采样时间轴上的主体事件 → face_events.json（判断层输入）
- motion 归一化改为相对位移（位移/框宽），跨采样率稳定

用法（CLI）：
    python liveclip/visual_signal.py --video <素材.mp4> [--sample 0.5] [--out face_events.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]  # (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# IoU 工具
# ---------------------------------------------------------------------------
def compute_iou(box1: Box, box2: Box) -> float:
    """两个框的交并比；不重叠返回 0.0。"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    x1_i, y1_i = max(x1_1, x1_2), max(y1_1, y1_2)
    x2_i, y2_i = min(x2_1, x2_2), min(y2_1, y2_2)
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def merge_boxes(boxes: List[Box]) -> Optional[Box]:
    """合并多个框为一个外接框（无框返回 None）。"""
    if not boxes:
        return None
    if len(boxes) == 1:
        return boxes[0]
    return (
        min(b[0] for b in boxes), min(b[1] for b in boxes),
        max(b[2] for b in boxes), max(b[3] for b in boxes),
    )


# ---------------------------------------------------------------------------
# IoU 跟踪器（移植 PersonTracker）
# ---------------------------------------------------------------------------
class IoUTracker:
    """基于 IOU 的目标跟踪：给每个检测目标分配稳定 track_id。

    行为：新框与现存 track 匹配（IOU 最大且超阈值）→ 更新该 track；
    无匹配 → 新建 track；丢失帧数超限 → 删除 track。
    """

    def __init__(self, iou_threshold: float = 0.3, max_lost_frames: int = 10):
        self.tracks: Dict[int, Dict[str, Any]] = {}
        self.next_id = 0
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames

    def update(self, detected_boxes: List[Box]) -> List[int]:
        """输入本帧检测框列表，返回与 boxes **顺序一致** 的 track_id 列表。

        （顺序一致性是调用方 zip(boxes, track_ids) 配对的前提，不能按 id 排序。）
        无检测框时返回 []。
        """
        if not detected_boxes:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]["lost_frames"] += 1
                if self.tracks[track_id]["lost_frames"] > self.max_lost_frames:
                    del self.tracks[track_id]
            return []

        matched_tracks: set = set()
        result_ids: List[int] = []
        for det_box in detected_boxes:
            best_iou, best_track_id = 0.0, None
            for track_id, track_data in self.tracks.items():
                if track_id in matched_tracks:
                    continue
                iou = compute_iou(det_box, track_data["box"])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou, best_track_id = iou, track_id
            if best_track_id is not None:
                self.tracks[best_track_id]["box"] = det_box
                self.tracks[best_track_id]["lost_frames"] = 0
                matched_tracks.add(best_track_id)
                result_ids.append(best_track_id)
            else:
                new_id = self.next_id
                self.next_id += 1
                self.tracks[new_id] = {"box": det_box, "lost_frames": 0}
                matched_tracks.add(new_id)
                result_ids.append(new_id)

        for track_id in list(self.tracks.keys()):
            if track_id not in matched_tracks:
                self.tracks[track_id]["lost_frames"] += 1
                if self.tracks[track_id]["lost_frames"] > self.max_lost_frames:
                    del self.tracks[track_id]

        return result_ids


# ---------------------------------------------------------------------------
# 主体评分（移植 SmartActionDetector，适配 Haar 人脸检测）
# ---------------------------------------------------------------------------
@dataclass
class SubjectHit:
    """一帧中的一个检测目标（已评分）。"""
    track_id: int
    box: Box
    center: Tuple[float, float]
    area: float
    conf: float
    center_prox: float      # 离画面中心近（0-1，越近越大）
    size: float             # 占画面比例（0-1，越大越大）
    motion: float           # 跨帧位移（归一化到框宽，0-1）
    temporal: float         # 是否持续出现（与历史选中框重叠）
    score: float = 0.0      # 综合主体分


class SubjectScorer:
    """五维主体评分：conf/center_prox/size/motion/temporal 各 0.2。

    高分 = 画面中央 + 占比大 + 正在动 + 持续存在 = "当前主角"（说话人候选）。
    """

    WEIGHTS = dict(conf=0.2, center_prox=0.2, size=0.2, motion=0.2, temporal=0.2)

    def __init__(self, sticky_frames: int = 15, conf_default: float = 0.7):
        self.sticky_frames = sticky_frames
        self.conf_default = conf_default          # Haar 无真置信度，用默认值
        self.prev_frame_data: List[Dict[str, Any]] = []
        self.frame_count = 0
        self.selection_history: deque = deque(maxlen=sticky_frames)
        self.track_boxes: Dict[int, Box] = {}     # track_id -> 上一帧框（由外部同步）

    def reset(self) -> None:
        self.prev_frame_data = []
        self.frame_count = 0
        self.selection_history.clear()
        self.track_boxes.clear()

    def score(
        self,
        frame: np.ndarray,
        boxes: List[Box],
        track_ids: List[int],
        *,
        max_people: int = 2,
    ) -> List[SubjectHit]:
        """对一帧的检测框（含 track_id）打分，返回按 score 降序的列表。"""
        h, w = frame.shape[:2]
        center_x, center_y = w / 2.0, h / 2.0
        frame_area = h * w
        hits: List[SubjectHit] = []

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            box_cx, box_cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            area = (x2 - x1) * (y2 - y1)

            # center_prox：归一化到画面中心距离（对角为 1）
            dist = float(np.hypot(box_cx - center_x, box_cy - center_y))
            max_dist = float(np.hypot(center_x, center_y))
            center_score = 1.0 - (dist / max_dist if max_dist > 0 else 0.0)

            # size：占画面比例，阈值 30% 即满分
            size_score = min(area / (frame_area * 0.3), 1.0)

            # motion：相对上一帧同目标框的位移（归一化到本框宽），clamp 0-1
            motion_score = 0.0
            prev_box = self.track_boxes.get(track_id)
            if prev_box is not None:
                prev_cx, prev_cy = (prev_box[0] + prev_box[2]) / 2.0, (prev_box[1] + prev_box[3]) / 2.0
                shift = float(np.hypot(box_cx - prev_cx, box_cy - prev_cy))
                box_w = max(x2 - x1, 1)
                motion_score = min(shift / box_w, 1.0)

            # temporal：与近若干帧"选中主体"框重叠则算持续出现
            temporal_score = 0.0
            for hist in self.selection_history:
                if any(compute_iou(box, hb) > 0.5 for hb in hist):
                    temporal_score = 1.0
                    break

            hit = SubjectHit(
                track_id=track_id, box=box, center=(box_cx, box_cy), area=area,
                conf=self.conf_default,
                center_prox=center_score, size=size_score,
                motion=motion_score, temporal=temporal_score,
            )
            hit.score = (
                hit.conf * self.WEIGHTS["conf"]
                + hit.center_prox * self.WEIGHTS["center_prox"]
                + hit.size * self.WEIGHTS["size"]
                + hit.motion * self.WEIGHTS["motion"]
                + hit.temporal * self.WEIGHTS["temporal"]
            )
            hits.append(hit)

        # 更新状态
        self.prev_frame_data = hits
        self.frame_count += 1
        for hit in hits:
            self.track_boxes[hit.track_id] = hit.box
        hits.sort(key=lambda x: x.score, reverse=True)
        selected = hits[:max_people]
        self.selection_history.append([h.box for h in selected])
        return hits


# ---------------------------------------------------------------------------
# Haar 人脸检测（替代 YOLO：多级联融合，零下载）
# ---------------------------------------------------------------------------
def _default_cascade_dir() -> str:
    """OpenCV 自带级联文件目录。"""
    return os.path.join(os.path.dirname(cv2.__file__), "data")


def build_face_cascades(
    names: Optional[List[str]] = None,
    cascade_dir: Optional[str] = None,
) -> List[cv2.CascadeClassifier]:
    """加载多个人脸级联（正脸 + 侧脸，提高召回）。

    默认：[haarcascade_frontalface_default, haarcascade_frontalface_alt2,
           haarcascade_profileface]
    """
    if names is None:
        names = [
            "haarcascade_frontalface_default.xml",
            "haarcascade_frontalface_alt2.xml",
            "haarcascade_profileface.xml",
        ]
    cascade_dir = cascade_dir or _default_cascade_dir()
    cascades = []
    for n in names:
        p = os.path.join(cascade_dir, n)
        cc = cv2.CascadeClassifier(p)
        if not cc.empty():
            cascades.append(cc)
    if not cascades:
        raise RuntimeError(f"未找到任何人脸级联文件，请检查 OpenCV data 目录: {cascade_dir}")
    return cascades


def detect_faces_haar(
    frame: np.ndarray,
    cascades: List[cv2.CascadeClassifier],
    *,
    min_size: int = 40,
    scale_factor: float = 1.1,
    min_neighbors: int = 4,
    iou_dedup: float = 0.4,
) -> List[Box]:
    """多级联人脸检测 + IOU 去重。

    返回框列表 [(x1, y1, x2, y2), ...]，按面积降序（大脸优先）。
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # 光照归一化，提升 Haar 召回
    all_boxes: List[Box] = []
    for cc in cascades:
        faces = cc.detectMultiScale(
            gray, scaleFactor=scale_factor, minNeighbors=min_neighbors,
            minSize=(min_size, min_size),
        )
        for (x, y, w, h) in faces:
            all_boxes.append((int(x), int(y), int(x + w), int(y + h)))

    # IOU 去重（同一张脸被多个级联检出）
    dedup: List[Box] = []
    all_boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    for b in all_boxes:
        if not any(compute_iou(b, d) > iou_dedup for d in dedup):
            dedup.append(b)
    return dedup


# ---------------------------------------------------------------------------
# 分屏区域划分（说话人-位置绑定的视觉依据）
# ---------------------------------------------------------------------------
def region_of(box_center: Tuple[float, float], frame_w: int, frame_h: int,
              mode: str = "auto", n_faces: int = 1) -> str:
    """框中心点 → 区域名。

    mode:
      - "lr"   ：左右二分（连麦布局），left / right
      - "quad" ：四象限（PK 布局），tl / tr / bl / br
      - "auto" ：单人 whole；2 人以下 lr；3 人及以上 quad
    返回 "whole" 表示单人全画面。
    """
    cx, cy = box_center
    if mode == "auto":
        mode = "lr" if n_faces <= 2 else "quad"
    if mode == "lr":
        return "whole" if n_faces <= 1 else ("left" if cx < frame_w / 2.0 else "right")
    if mode == "quad":
        if n_faces <= 1:
            return "whole"
        h = "l" if cx < frame_w / 2.0 else "r"
        v = "t" if cy < frame_h / 2.0 else "b"
        return h + v
    return "whole"


# ---------------------------------------------------------------------------
# 采样检测主流程
# ---------------------------------------------------------------------------
def extract_face_signal(
    video_path: str,
    *,
    sample_interval: float = 0.5,
    max_people: int = 2,
    min_face_size: int = 40,
    region_mode: str = "lr",  # lr 稳定（1人=whole/2人=left-right）；PK 用 quad
    cascades: Optional[List[cv2.CascadeClassifier]] = None,
    tracker: Optional[IoUTracker] = None,
    scorer: Optional[SubjectScorer] = None,
) -> Dict[str, Any]:
    """按采样间隔遍历视频，输出画面主体（人脸）时间轴信号。

    返回 face_events.json 结构：
    {
      "video": "...", "duration": 194.8, "fps": 30.0, "sample_interval": 0.5,
      "events": [
        {"t": 0.5, "n_faces": 1,
         "subjects": [{"track_id":0, "score":0.8, "center_prox":0.9, "size":0.3,
                       "motion":0.05, "temporal":1.0, "box":[x1,y1,x2,y2]}],
         "dominant": {"track_id":0, "score":0.8}}, ...
      ]
    }
    """
    if cascades is None:
        cascades = build_face_cascades()
    if tracker is None:
        tracker = IoUTracker()
    if scorer is None:
        scorer = SubjectScorer()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps if fps > 0 else 0.0

    events: List[Dict[str, Any]] = []
    frame_id = 0
    next_sample_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id < next_sample_frame:
            frame_id += 1
            continue
        next_sample_frame = frame_id + max(1, int(round(fps * sample_interval)))

        # 降采样加速（长边 <= 960，Haar 在较小图更快且足够）
        h, w = frame.shape[:2]
        if max(h, w) > 960:
            scale = 960.0 / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)

        boxes = detect_faces_haar(frame, cascades, min_size=min_face_size)
        track_ids = tracker.update(boxes)
        hits = scorer.score(frame, boxes, track_ids, max_people=max_people)

        t = frame_id / fps
        subjects = []
        dominant = None
        n_faces = len(hits)
        for hit in hits:
            rec = {
                "track_id": hit.track_id,
                "region": region_of(hit.center, frame.shape[1], frame.shape[0],
                                    region_mode, n_faces),
                "score": round(hit.score, 3),
                "center_prox": round(hit.center_prox, 3),
                "size": round(hit.size, 3),
                "motion": round(hit.motion, 3),
                "temporal": round(hit.temporal, 3),
                "box": list(hit.box),
            }
            subjects.append(rec)
            if dominant is None or hit.score > dominant["score"]:
                dominant = rec
        events.append({
            "t": round(t, 3),
            "n_faces": n_faces,
            "subjects": subjects,
            "dominant": dominant,
        })

        frame_id += 1

    cap.release()
    return {
        "video": str(video_path),
        "duration": round(duration, 3),
        "fps": round(fps, 3),
        "sample_interval": sample_interval,
        "frame_max_side": 960,  # 人脸检测前的降采样边长（box 坐标基于此坐标系）
        "events": events,
    }


# ---------------------------------------------------------------------------
# visual_events 整合：scdet + freezedetect（ffmpeg 子进程包装）
# ---------------------------------------------------------------------------
def find_ffmpeg() -> str:
    """优先项目 tools/ffmpeg，其次 PATH。"""
    proj = Path(__file__).resolve().parent.parent
    local = proj / "tools" / "ffmpeg" / "ffmpeg.exe"
    if local.is_file():
        return str(local)
    import shutil
    p = shutil.which("ffmpeg")
    if p:
        return p
    raise RuntimeError("未找到 ffmpeg")


def extract_scene_changes(
    video_path: str,
    *,
    score_threshold: float = 8.0,
    merge_window: float = 1.0,
) -> List[Dict[str, float]]:
    """ffmpeg scdet：镜头切换点列表 [{t, score}]（同窗口取最高分）。"""
    ffmpeg = find_ffmpeg()
    r = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", video_path,
         "-vf", "scdet=threshold=0.3,metadata=print:file=-",
         "-an", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    pts = re.findall(r"lavfi\.scd\.score: ([\d.]+), lavfi\.scd\.time: ([\d.]+)",
                     r.stderr + r.stdout)
    raw = [(float(t), float(s)) for s, t in pts if float(s) > score_threshold]
    raw.sort()
    merged = []
    for t, s in raw:
        if merged and t - merged[-1]["t"] < merge_window:
            if s > merged[-1]["score"]:
                merged[-1] = {"t": t, "score": s}
        else:
            merged.append({"t": t, "score": s})
    return merged


def extract_frozen_segments(
    video_path: str,
    *,
    noise_db: float = -30.0,
    min_duration: float = 2.0,
) -> List[Dict[str, float]]:
    """ffmpeg freezedetect：画面冻结段 [{start, end, duration}]。"""
    ffmpeg = find_ffmpeg()
    r = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", video_path,
         "-vf", f"freezedetect=n={noise_db}dB:d={min_duration},metadata=print:file=-",
         "-an", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    out = r.stderr + r.stdout
    starts = re.findall(r"freeze_start: ([\d.]+)", out)
    ends = re.findall(r"freeze_end: ([\d.]+)", out)
    segs = []
    for s, e in zip(starts, ends):
        s_f, e_f = float(s), float(e)
        segs.append({"start": round(s_f, 3), "end": round(e_f, 3),
                     "duration": round(e_f - s_f, 3)})
    return segs


def extract_visual_events(
    video_path: str,
    *,
    sample_interval: float = 0.5,
    scene_threshold: float = 8.0,
    frozen_noise_db: float = -30.0,
    frozen_min_duration: float = 2.0,
    region_mode: str = "lr",
) -> Dict[str, Any]:
    """整合三信号 → visual_events.json（判断层输入）。

    {video, duration, fps, sample_interval,
     scene_changes: [...], frozen_segments: [...], face_events: [...]}
    """
    face = extract_face_signal(video_path, sample_interval=sample_interval,
                               region_mode=region_mode)
    return {
        "video": str(video_path),
        "duration": face["duration"],
        "fps": face["fps"],
        "sample_interval": sample_interval,
        "scene_changes": extract_scene_changes(video_path, score_threshold=scene_threshold),
        "frozen_segments": extract_frozen_segments(
            video_path, noise_db=frozen_noise_db, min_duration=frozen_min_duration),
        "face_events": face["events"],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(prog="visual_signal.py", description="视觉信号层：主体/切换/冻结检测")
    ap.add_argument("--video", required=True, help="素材视频路径")
    ap.add_argument("--sample", type=float, default=0.5, help="人脸采样间隔（秒）")
    ap.add_argument("--scene-threshold", type=float, default=8.0, help="场景切换分数阈值")
    ap.add_argument("--region-mode", choices=["auto", "lr", "quad"], default="lr",
                    help="分屏区域划分：lr=左右连麦（默认）/ quad=四象限PK / auto=按人数自适应")
    ap.add_argument("--no-face", action="store_true", help="跳过人脸检测（只跑 scdet/freezedetect）")
    ap.add_argument("--out", default=None, help="输出 JSON 路径（默认 outputs/work/visual_events.json）")
    args = ap.parse_args()

    out = args.out
    if out is None:
        out = str(Path(__file__).resolve().parent.parent / "outputs" / "work" / "visual_events.json")

    if args.no_face:
        result = {
            "video": str(args.video),
            "scene_changes": extract_scene_changes(args.video, score_threshold=args.scene_threshold),
            "frozen_segments": extract_frozen_segments(args.video),
        }
    else:
        result = extract_visual_events(
            args.video, sample_interval=args.sample,
            scene_threshold=args.scene_threshold,
            region_mode=args.region_mode)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({
        "ok": True,
        "video": str(args.video),
        "duration": result.get("duration", "?"),
        "scene_changes": len(result.get("scene_changes", [])),
        "frozen_segments": len(result.get("frozen_segments", [])),
        "face_samples": len(result.get("face_events", [])),
        "out": out,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
