"""将验证集 20 份 GIF 合成 4 行 × 5 列的同步动画。"""

import bisect
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path("outputs/act_valid100")
OUTPUT = Path("outputs/act_valid100_grid.gif")

ROWS = 4
COLS = 5
SIZE = 192
LABEL_HEIGHT = 30
FRAME_MS = 50  # 输出 20 FPS

splits = json.loads(Path("data/lift_act/splits.json").read_text())
episode_ids = splits["valid"]
assert len(episode_ids) == ROWS * COLS

font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
font = (
    ImageFont.truetype(str(font_path), 13)
    if font_path.exists()
    else ImageFont.load_default()
)

clips = []

for episode_id in episode_ids:
    directory = ROOT / f"episode_{episode_id:04d}"
    gif_path = directory / "rollout.gif"
    result = json.loads((directory / "result.json").read_text())

    frames = []
    end_times = []
    elapsed = 0

    with Image.open(gif_path) as gif:
        for i in range(gif.n_frames):
            gif.seek(i)
            # Pillow 解码 GIF 的当前合成画面，再复制为独立 RGB 帧。
            frame = gif.convert("RGB").resize(
                (SIZE, SIZE), Image.Resampling.LANCZOS
            )
            duration = int(gif.info.get("duration", FRAME_MS))
            duration = duration if duration > 0 else FRAME_MS

            frames.append(frame)
            elapsed += duration
            end_times.append(elapsed)

    stable = bool(result["stable_native_success_10_steps"])
    final = bool(result["success_final"])

    clips.append({
        "episode": episode_id,
        "frames": frames,
        "end_times": end_times,
        "duration": elapsed,
        "label": (
            f"ep {episode_id:03d}  "
            f"stable={'Y' if stable else 'N'} "
            f"final={'Y' if final else 'N'}"
        ),
        "color": (
            (40, 180, 80) if final
            else (220, 150, 40) if stable
            else (210, 60, 60)
        ),
    })
    print(
        f"episode={episode_id}, frames={len(frames)}, "
        f"duration={elapsed / 1000:.2f}s"
    )

total_ms = max(clip["duration"] for clip in clips)
width = COLS * SIZE
height = ROWS * (SIZE + LABEL_HEIGHT)
output_frames = []

for time_ms in range(0, total_ms, FRAME_MS):
    canvas = Image.new("RGB", (width, height), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)

    for index, clip in enumerate(clips):
        row, col = divmod(index, COLS)
        x = col * SIZE
        y = row * (SIZE + LABEL_HEIGHT)

        # 较短动画播放结束后保持末帧，不单独循环。
        frame_index = min(
            bisect.bisect_right(clip["end_times"], time_ms),
            len(clip["frames"]) - 1,
        )
        canvas.paste(clip["frames"][frame_index], (x, y + LABEL_HEIGHT))
        draw.text(
            (x + 5, y + 7),
            clip["label"],
            fill=clip["color"],
            font=font,
        )

    output_frames.append(canvas)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
output_frames[0].save(
    OUTPUT,
    save_all=True,
    append_images=output_frames[1:],
    duration=FRAME_MS,
    loop=0,
    optimize=False,
    disposal=2,
)

with Image.open(OUTPUT) as gif:
    print("Saved frames:", gif.n_frames)

print("Layout:", ROWS, "rows ×", COLS, "columns")
print("Resolution:", width, "×", height)
print("Output:", OUTPUT.resolve())
