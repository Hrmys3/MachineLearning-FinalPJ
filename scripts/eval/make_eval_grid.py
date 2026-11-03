"""将本次实验按每20轮合成4行×5列同步GIF，不足20轮保留空格。"""
import argparse
import bisect
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROWS, COLS = 4, 5
SIZE, LABEL_HEIGHT, FRAME_MS = 192, 30, 50


def episode_order(root):
    """优先使用本次实验的计划顺序，保留缺失/失败轮次的位置。"""
    config_path = root / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        if "episode_ids" in config:
            return config["episode_ids"]
    status_path = root / "process_status.json"
    if status_path.exists():
        return [s["episode_id"] for s in json.loads(status_path.read_text())]
    return [int(p.name.removeprefix("episode_")) for p in sorted(root.glob("episode_*"))
            if p.is_dir() and p.name.removeprefix("episode_").isdigit()]


def render_group(root, episode_ids, output):
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), 13) if font_path.exists() else ImageFont.load_default()
    clips = []
    for episode_id in episode_ids:
        directory = root / f"episode_{episode_id:04d}"
        frames, end_times, elapsed = [], [], 0
        label = f"ep {episode_id:03d} MISSING/FAILED"
        color = (230, 80, 70)
        try:
            result = json.loads((directory / "result.json").read_text())
            with Image.open(directory / "rollout.gif") as gif:
                for i in range(gif.n_frames):
                    gif.seek(i)
                    frames.append(gif.convert("RGB").resize((SIZE, SIZE), Image.Resampling.LANCZOS))
                    elapsed += max(1, int(gif.info.get("duration", FRAME_MS)))
                    end_times.append(elapsed)
            hold = result.get("first_lift_hold_success", result.get("first_lift_hold_2s_success"))
            stable = result.get("stable_native_success_10_steps", False)
            final = result.get("success_final", False)
            label = f"ep {episode_id:03d} H={'-' if hold is None else int(hold)} S={int(stable)} F={int(final)}"
            color = (40, 180, 80) if final else (220, 150, 40) if stable else color
        except (OSError, ValueError, EOFError) as exc:
            # 保留原位，不将后面的成功样本挪到失败位置。
            print(f"episode={episode_id}: {exc}", flush=True)
            frames, end_times, elapsed = [], [], 0
        clips.append((frames, end_times, elapsed, label, color))

    total_ms = max(FRAME_MS, max(clip[2] for clip in clips))
    output_frames = []
    for time_ms in range(0, total_ms, FRAME_MS):
        canvas = Image.new("RGB", (COLS * SIZE, ROWS * (SIZE + LABEL_HEIGHT)), (20, 20, 20))
        draw = ImageDraw.Draw(canvas)
        for index, (frames, end_times, _, label, color) in enumerate(clips):
            row, col = divmod(index, COLS)
            x, y = col * SIZE, row * (SIZE + LABEL_HEIGHT)
            if frames:
                frame_index = min(bisect.bisect_right(end_times, time_ms), len(frames) - 1)
                canvas.paste(frames[frame_index], (x, y + LABEL_HEIGHT))
            draw.text((x + 5, y + 7), label, fill=color, font=font)
        output_frames.append(canvas)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_frames[0].save(output, save_all=True, append_images=output_frames[1:],
                          duration=FRAME_MS, loop=0, optimize=False, disposal=2)
    print("GIF:", output.resolve(), flush=True)


def make_grids(root, output=None):
    root = Path(root)
    episode_ids = episode_order(root)
    if not episode_ids:
        raise ValueError(f"{root} 中没有本次实验的 episode 记录")
    groups = [episode_ids[i:i + ROWS * COLS] for i in range(0, len(episode_ids), ROWS * COLS)]
    if output is None:
        paths = [root / f"grid_{i + 1:03d}.gif" for i in range(len(groups))]
    else:
        output = Path(output)
        paths = [output] if len(groups) == 1 else [
            output.with_name(f"{output.stem}_{i + 1:03d}{output.suffix}") for i in range(len(groups))]
    # 一次性检查所有目标，避免部分写入后才发现后面的文件已存在。
    for path in paths:
        if path.exists():
            raise FileExistsError(f"{path} 已存在，不覆盖已有动画")
    for group, path in zip(groups, paths):
        render_group(root, group, path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="本次实验目录")
    parser.add_argument("--output", type=Path, help="可选输出文件名；多组自动附加_001等编号")
    args = parser.parse_args()
    make_grids(args.input, args.output)


if __name__ == "__main__":
    main()
