"""
데이터셋 생성 관련 모듈
"""

import os
from pathlib import Path
import numpy as np

def process_animation_files(files, name, frames, rate, output_dir):
    """
    애니메이션 파일들을 처리하여 데이터셋을 생성/수정합니다.
    
    Args:
        files: 입력 애니메이션 파일 목록
        name: 데이터셋 이름
        frames: 클립당 프레임 수
        rate: 프레임 레이트
        output_dir: 출력 디렉토리 경로
        
    Returns:
        HTML 출력, 상태 정보
    """
    if not files or len(files) == 0:
        return (
            """
            <div style="width: 100%; height: 500px; background-color: #333; border-radius: 8px; 
                    display: flex; justify-content: center; align-items: center; color: #ccc;">
                <div style="text-align: center;">
                    <h3>오류</h3>
                    <p>하나 이상의 애니메이션 파일을 업로드해야 합니다</p>
                </div>
            </div>
            """, 
            {"상태": "오류", "원인": "파일이 없음"}
        )
    
    try:
        # 파일 형식별 카운팅
        file_types = {"glb": 0, "bvh": 0, "fbx": 0}
        for f in files:
            ext = Path(f.name).suffix.lower()[1:]  # .을 제외한 확장자
            if ext in file_types:
                file_types[ext] += 1
                
        # 가상의 데이터셋 생성 결과
        final_output_dir = os.path.join(output_dir, "datasets", name)
        os.makedirs(final_output_dir, exist_ok=True)
        
        num_files = len(files)
        estimated_frames = num_files * 300  # 가정: 각 파일당 약 300프레임
        estimated_clips = estimated_frames // frames
        
        # 파일 형식 정보 문자열 생성
        file_type_info = ", ".join([f"{count}개의 {ext.upper()}" for ext, count in file_types.items() if count > 0])
        
        return (
            f"""
            <div style="width: 100%; height: 500px; background-color: #333; border-radius: 8px; padding: 20px; color: #fff; overflow-y: auto;">
                <h3>데이터셋 수정 완료</h3>
                <p>데이터셋이 수정되었습니다.</p>
                <div style="background-color: #444; padding: 15px; border-radius: 5px; margin-top: 15px;">
                    <h4>데이터셋 정보</h4>
                    <ul>
                        <li><b>이름:</b> {name}</li>
                        <li><b>경로:</b> {final_output_dir}</li>
                        <li><b>파일 수:</b> {num_files} ({file_type_info})</li>
                        <li><b>클립당 프레임:</b> {frames}</li>
                        <li><b>프레임 레이트:</b> {rate} FPS</li>
                    </ul>
                </div>
                <div style="background-color: #444; padding: 15px; border-radius: 5px; margin-top: 15px;">
                    <h4>처리된 파일</h4>
                    <ul>
                        {"".join([f"<li>{Path(f.name).name} <small>({Path(f.name).suffix.lower()[1:].upper()})</small></li>" for f in files])}
                    </ul>
                </div>
            </div>
            """,
            {
                "상태": "완료",
                "데이터셋 이름": name,
                "파일 수": num_files,
                "파일 형식": file_types,
                "총 프레임": estimated_frames,
                "클립 수": estimated_clips,
                "클립당 프레임": frames,
                "프레임 레이트": rate
            }
        )
    except Exception as e:
        print(f"데이터셋 처리 오류: {e}")
        import traceback
        traceback.print_exc()
        return (
            f"""
            <div style="width: 100%; height: 500px; background-color: #333; border-radius: 8px; 
                    display: flex; justify-content: center; align-items: center; color: #ccc;">
                <div style="text-align: center;">
                    <h3>오류</h3>
                    <p>데이터셋 수정 중 오류가 발생했습니다: {str(e)}</p>
                </div>
            </div>
            """, 
            {"상태": "오류", "원인": str(e)}
        )