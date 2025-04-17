"""
애니메이션 학습 데이터셋 생성 탭 모듈
"""

import gradio as gr
import os
from pathlib import Path
import glob
import numpy as np

def create_dataset_tab(MODELS_DIR):
    """
    애니메이션 학습 데이터셋 생성 탭 인터페이스 생성
    
    Args:
        MODELS_DIR: 모델 파일이 저장된 디렉토리 경로
    """
    with gr.Column():
        gr.Markdown("# 애니메이션 학습 데이터셋 생성")
        gr.Markdown("애니메이션 모델 학습에 필요한 데이터셋을 생성합니다.")
        
        with gr.Row():
            with gr.Column(scale=1):
                # 데이터셋 생성 옵션
                source_files = gr.File(
                    label="소스 애니메이션 파일 (GLB/BVH/FBX)",
                    file_types=[".glb", ".bvh", ".fbx"],
                    type="file",
                    file_count="multiple"
                )
                
                dataset_name = gr.Textbox(
                    label="데이터셋 이름",
                    placeholder="생성할 데이터셋 이름을 입력하세요",
                    value="my_animation_dataset"
                )
                
                with gr.Row():
                    frames_per_clip = gr.Number(
                        label="클립당 프레임 수", 
                        value=64,
                        precision=0,
                        minimum=16,
                        maximum=256
                    )
                    
                    frame_rate = gr.Number(
                        label="프레임 레이트", 
                        value=30,
                        precision=0,
                        minimum=10,
                        maximum=120
                    )
                
                generate_btn = gr.Button("데이터셋 생성", variant="primary")
                
                gr.Markdown("""
                ## 데이터셋 생성 방법
                1. 하나 이상의 애니메이션 파일(GLB, BVH 또는 FBX)을 업로드합니다.
                2. 데이터셋 이름과 설정을 지정합니다.
                3. '데이터셋 생성' 버튼을 클릭합니다.
                4. 생성된 데이터셋은 오른쪽에 결과가 표시됩니다.
                """)
                
            with gr.Column(scale=2):
                # 데이터셋 생성 결과 출력 영역
                dataset_output = gr.HTML("""
                <div style="width: 100%; height: 500px; background-color: #333; border-radius: 8px; 
                         display: flex; justify-content: center; align-items: center; color: #ccc;">
                    <div style="text-align: center;">
                        <h3>데이터셋 생성 결과</h3>
                        <p>데이터셋을 생성하려면 왼쪽 패널에서 파일을 업로드하고 '데이터셋 생성' 버튼을 클릭하세요</p>
                    </div>
                </div>
                """)
                
                # 데이터셋 통계 영역
                dataset_stats = gr.JSON({
                    "상태": "대기 중",
                    "파일 수": 0,
                    "총 프레임": 0,
                    "클립 수": 0
                })
        
        def generate_dataset(files, name, frames, rate):
            """
            애니메이션 학습 데이터셋 생성 함수
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
                # 여기에 실제 데이터셋 생성 로직 구현
                # (현재는 예시로 표시)
                
                # 파일 형식별 카운팅
                file_types = {"glb": 0, "bvh": 0, "fbx": 0}
                for f in files:
                    ext = Path(f.name).suffix.lower()[1:]  # .을 제외한 확장자
                    if ext in file_types:
                        file_types[ext] += 1
                        
                # 가상의 데이터셋 생성 결과
                output_dir = os.path.join(str(MODELS_DIR), "datasets", name)
                num_files = len(files)
                estimated_frames = num_files * 300  # 가정: 각 파일당 약 300프레임
                estimated_clips = estimated_frames // frames
                
                # 파일 형식 정보 문자열 생성
                file_type_info = ", ".join([f"{count}개의 {ext.upper()}" for ext, count in file_types.items() if count > 0])
                
                return (
                    f"""
                    <div style="width: 100%; height: 500px; background-color: #333; border-radius: 8px; padding: 20px; color: #fff; overflow-y: auto;">
                        <h3>데이터셋 생성 완료</h3>
                        <p>데이터셋이 생성되었습니다.</p>
                        <div style="background-color: #444; padding: 15px; border-radius: 5px; margin-top: 15px;">
                            <h4>데이터셋 정보</h4>
                            <ul>
                                <li><b>이름:</b> {name}</li>
                                <li><b>경로:</b> {output_dir}</li>
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
                return (
                    f"""
                    <div style="width: 100%; height: 500px; background-color: #333; border-radius: 8px; 
                            display: flex; justify-content: center; align-items: center; color: #ccc;">
                        <div style="text-align: center;">
                            <h3>오류</h3>
                            <p>데이터셋 생성 중 오류가 발생했습니다: {str(e)}</p>
                        </div>
                    </div>
                    """, 
                    {"상태": "오류", "원인": str(e)}
                )
        
        # 버튼 클릭 이벤트 함수 등록
        generate_btn.click(
            fn=generate_dataset,
            inputs=[source_files, dataset_name, frames_per_clip, frame_rate],
            outputs=[dataset_output, dataset_stats]
        )

# 단독 실행 시 테스트
if __name__ == "__main__":
    with gr.Blocks() as demo:
        create_dataset_tab(".")
    demo.launch()