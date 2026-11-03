"""只读硬件监控，可在另一终端采样；不安装依赖、不修改设备参数。"""
import argparse
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from performance import HardwareMonitor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='新CSV文件')
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--interval', type=float, default=1)
    args = parser.parse_args()
    if args.seconds <= 0 or args.interval <= 0:
        parser.error('seconds和interval须为正数')
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with HardwareMonitor(args.output, True, args.interval):
            time.sleep(args.seconds)
    except KeyboardInterrupt:
        pass
    print('硬件采样:', args.output.resolve())


if __name__ == '__main__':
    main()
