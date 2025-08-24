import argparse
import os
from pathlib import Path

import cv2
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector


def save_frame(cap: cv2.VideoCapture, frame_idx: int, out_path: Path) -> bool:
    """Seek tới frame_idx và lưu ảnh JPG ra out_path. Trả về True/False."""
    # Clamp phòng trường hợp frame_idx vượt giới hạn.
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        return False
    frame_idx = max(0, min(frame_idx, total - 1))

    # Seek & đọc khung hình.
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret or frame is None:
        # Thử lại một lần (một số backend cần read thêm sau khi set).
        ret, frame = cap.read()
        if not ret or frame is None:
            return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(out_path), frame)


def extract_keyframes(video_path: Path, out_dir: Path, threshold: float = 27.0) -> None:
    """
    Phát hiện scene bằng PySceneDetect (ContentDetector),
    và với mỗi scene, lưu 3 khung hình: start / mid / end (JPG).
    """
    # 1) Dò scene bằng PySceneDetect.
    video = open_video(str(video_path))
    scene_manager = SceneManager()
    # threshold càng thấp -> càng nhạy (nhiều cảnh hơn). Mặc định 27 là hợp lý cho đa số video.
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)
    scenes = scene_manager.get_scene_list()

    if not scenes:
        print("Không phát hiện scene nào. Thử giảm --threshold.")
        return

    # 2) Mở bằng OpenCV để trích frame theo chỉ số khung.
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Tổng số scene tìm được: {len(scenes)} | Tổng frame: {total_frames}")

    for i, (start_time, end_time) in enumerate(scenes, start=1):
        # FrameTimecode -> chỉ số khung.
        start_f = start_time.get_frames()
        # end_time trong PySceneDetect là mốc bắt đầu của cảnh kế tiếp (exclusive),
        # nên frame cuối thực sự của scene hiện tại là end_f - 1.
        end_f = end_time.get_frames() - 1

        # Phòng trường hợp giá trị bất thường:
        if end_f < start_f:
            end_f = start_f

        mid_f = (start_f + end_f) // 2

        # Đường dẫn file xuất.
        base = f"scene_{i:03d}"
        out_start = out_dir / f"{base}_start.jpg"
        out_mid = out_dir / f"{base}_mid.jpg"
        out_end = out_dir / f"{base}_end.jpg"

        ok1 = save_frame(cap, start_f, out_start)
        ok2 = save_frame(cap, mid_f, out_mid)
        ok3 = save_frame(cap, end_f, out_end)

        print(
            f"[Scene {i:03d}] frames [{start_f}, {mid_f}, {end_f}] "
            f"-> saved: start={ok1}, mid={ok2}, end={ok3}"
        )

    cap.release()
    print(f"Xong! Ảnh đã lưu tại: {out_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract 3 key frames (start/mid/end) cho mỗi scene từ 1 video MP4."
    )
    parser.add_argument("video", type=Path, help="Đường dẫn video .mp4")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("images"),
        help="Thư mục lưu ảnh (mặc định: images)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=27.0,
        help="Ngưỡng ContentDetector (thấp hơn -> nhạy hơn; mặc định 27.0).",
    )
    args = parser.parse_args()

    extract_keyframes(args.video, args.out, threshold=args.threshold)


if __name__ == "__main__":
    main()

#python3 extract_keyframes.py 1.mp4 --out images