import subprocess
import sys
from datetime import datetime
from pathlib import Path

"""
이 스크립트는 폴더 내의 FBX 파일을 GLTF 또는 GLB 형식으로 변환하는 기능을 제공합니다. FBX2glTF 도구를 사용하여 변환합니다.
This script provides functionality to convert FBX files in a folder to GLTF or GLB format using the `fbx2gltf` tool.

fbx2gltf 파일은 https://github.com/godotengine/FBX2glTF에서 찾을 수 있습니다.
fbx2gltf can be found at https://github.com/godotengine/FBX2glTF.

Functions:
    convert_fbx_folder(input_folder: str, output_folder: str, to_glb: bool = False):
        모든 FBX 파일을 GLTF 또는 GLB 형식으로 변환합니다.
        Converts all FBX files in the specified input folder to GLTF or GLB format and saves them in the output folder.
Usage:
    원하는 FBX 파일이 있는 폴더와 변환된 파일을 저장할 폴더를 지정하여 스크립트를 실행합니다. glb 형식으로 변환하려면 `--glb` 플래그를 사용합니다.
    Run the script with the desired input and output folder paths. Use the `--glb` flag to convert to GLB format instead of GLTF.

Example:
    python converter_fbx2glx.py --glb

Parameters:
    input_folder (str)
    output_folder (str)
    to_glb (bool): glb format(binary)로 변환 여부 설정. 기본값은 False입니다. Whether to convert to GLB format (binary). Defaults to False (GLTF format).
"""

def convert_fbx_folder(
    input_folder: str,
    output_folder: str,
    to_glb: bool = False
):
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    fbx_files = list(input_path.glob("*.fbx"))
    if not fbx_files:
        print("No FBX files in the input folder.")
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

        print(f"converting: {fbx_file.name} -> {output_file.name}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print("✔ Success")
        else:
            print("❌ Failed")
            print(result.stderr)

# 사용 예제
# example usage
if __name__ == "__main__":
    convert_fbx_folder(
        input_folder="./fbx_files",      # 이 폴더에 fbx 파일들을 넣어두세요. input FBX files in this folder
        output_folder=f"./converted_{datetime.now().strftime('%Y%m%d_%H%M%S')}",     # 변환된 파일은 이 폴더에 생성됩니다. output folder for converted files
        to_glb=True if "--glb" in sys.argv else False
    )
