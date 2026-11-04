"""保存抽帧动画并保留原模拟时间轴，只处理输出。"""


def save_rollout_gif(frames, steps, output):
    if len(frames) != len(steps) or not frames or steps[0] != 0:
        raise ValueError('视频帧须与从0开始的步数匹配')
    if any(b <= a for a, b in zip(steps, steps[1:])):
        raise ValueError('视频步数须严格递增')
    durations = [(b-a)*50 for a, b in zip(steps, steps[1:])] + [50]
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=durations, loop=0)
