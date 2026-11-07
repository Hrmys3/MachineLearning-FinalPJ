"""从官方X5模板生成本地规划路径；不改原模板、关节或碰撞参数。"""
import argparse
import hashlib
import json
from pathlib import Path

import yaml


def prepare_x5(assets_dir):
    assets = Path(assets_dir).resolve()
    template = assets / "Robots/x5/curobo_tmp.yml"
    source = template.read_bytes()
    data = yaml.safe_load(source)

    def resolve(value):
        if isinstance(value, dict):
            return {key: resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve(item) for item in value]
        if isinstance(value, str):
            return value.replace("${ASSETS_PATH}", str(assets.parent))
        return value

    generated = resolve(data)
    kinematics = generated["robot_cfg"]["kinematics"]
    if not Path(kinematics["urdf_path"]).is_file():
        raise FileNotFoundError(kinematics["urdf_path"])
    if not Path(kinematics["asset_root_path"]).is_dir():
        raise FileNotFoundError(kinematics["asset_root_path"])
    content = yaml.safe_dump(generated, sort_keys=False).encode()
    target = assets / "Robots/x5/curobo.yml"
    if target.exists():
        if target.read_bytes() != content:
            raise FileExistsError(f"{target} 已存在且内容不同，不覆盖")
    else:
        with target.open("xb") as output:
            output.write(content)
    report = {
        "template": str(template),
        "template_sha256": hashlib.sha256(source).hexdigest(),
        "generated": str(target),
        "generated_sha256": hashlib.sha256(content).hexdigest(),
        "substitution": "${ASSETS_PATH} -> parent of resolved Assets directory",
    }
    manifest = target.with_suffix(".generated.json")
    if not manifest.exists():
        with manifest.open("x") as output:
            json.dump(report, output, indent=2)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_x5(args.assets_dir), indent=2))


if __name__ == "__main__":
    main()
