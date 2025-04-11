import subprocess
import os, sys
from datetime import datetime
from pathlib import Path

def convert_fbx_folder(
    input_folder: str,
    output_folder: str,
    to_glb: bool = False
):
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    fbx_files = list(input_path.glob("*.fbx"))
    if not fbx_files:
        print("FBX 파일이 없습니다.")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    for fbx_file in fbx_files:
        output_file = output_path / (fbx_file.stem + (".glb" if to_glb else ".gltf"))

        cmd = [
            "./fbx2gltf",
            "-i", str(fbx_file),
            "-o", str(output_file)
        ]

        if to_glb:
            cmd.append("-b")  # binary format = glb

        print(f"변환 중: {fbx_file.name} -> {output_file.name}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print("✔ 성공")
        else:
            print("❌ 실패")
            print(result.stderr)

# 사용 예시
if __name__ == "__main__":
    convert_fbx_folder(
        input_folder="./fbx_files",      # FBX 파일이 있는 폴더
        output_folder=f"./converted_{datetime.now().strftime('%Y%m%d_%H%M%S')}",     # 변환된 파일이 저장될 폴더
        to_glb=True if "--glb" in sys.argv else False
    )
