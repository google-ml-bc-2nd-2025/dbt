"""
3D 모델 뷰어 웹 애플리케이션의 메인 모듈
"""

import gradio as gr
import os, atexit
import threading
import numpy as np
from pathlib import Path

# 프로젝트 내 사용자 정의 모듈 임포트
from viewer_template import create_viewer_html
from file_utils import apply_animation, send_prompt
from apis import start_api_server  # FastAPI 서버 실행 함수 임포트
# SMPL 애니메이션 처리를 위한 모듈 추가

# 정적 파일 디렉토리 생성
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
MODELS_DIR = STATIC_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# 뷰어 템플릿 HTML 파일 경로
TEMPLATE_PATH = STATIC_DIR / "viewer_template.html"

# HTML 뷰어 파일 생성
VIEWER_PATH = STATIC_DIR / "viewer.html"
create_viewer_html(VIEWER_PATH)

# FastAPI 서버 실행 (별도 스레드에서)
def run_fastapi_server():
    """별도 스레드에서 FastAPI 서버를 실행합니다."""
    start_api_server(port=8000)

# 서버 종료 관리 추가
def cleanup():
    """애플리케이션 종료 시 정리 작업을 수행합니다."""
    print("서버가 종료됩니다. 정리 작업 수행 중...")
    # 필요한 정리 작업 수행
    print("정리 작업 완료.")

# 종료 시 정리 함수 등록
atexit.register(cleanup)


# Gradio 인터페이스 생성
with gr.Blocks(title="애니메이션 뷰어") as demo:
    gr.Markdown("# AssetSmith - (애니메이션 뷰어)")
    gr.Markdown("스킨이 있는 GLB 모델을 지정 후 원하는 애니메이션 데이터를 선택하세요.")
    
    with gr.Row():
        with gr.Column(scale=1):
            # 파일 업로드 영역
            skin_model = gr.File(
                label="스킨 모델 (GLB)",
                file_types=[".glb"],
                type="file"
            )
            
            anim_model = gr.File(
                label="애니메이션 파일 (GLB/BVH/NPY)",
                file_types=[".glb", ".bvh", ".npy"],  # .npy 파일 지원 추가
                type="file"
            )
            
            apply_btn = gr.Button("적용 및 보기", variant="primary")
            
            gr.Markdown("""
            ## 사용 방법
            1. 스킨이 있는 GLB 모델을 첫 번째 필드에 업로드합니다.
            2. 애니메이션 파일(GLB, BVH 또는 SMPL NPY)을 두 번째 필드에 업로드합니다.
            3. '적용 및 보기' 버튼을 클릭합니다.
            4. 오른쪽 패널에서 애니메이션이 적용된 모델을 확인합니다.
            
            **참고**: 
            - GLB 애니메이션: 두 모델의 리깅(뼈대) 구조가 동일해야 합니다. 즉시 확인 가능
            - BVH 애니메이션: 본 이름이 비슷하면 자동으로 매핑을 시도합니다. 본만 래더링 됨.
            - SMPL NPY 애니메이션: SMPL 호환 모델에 적용됩니다. 에러 발생하면 static/models에 생성된 anim_xx 파일을 선택 후 실행
            """)
            
        with gr.Column(scale=2):
            # 3D 모델 뷰어
            viewer = gr.HTML("""
            <div style="width: 100%; height: 500px; background-color: #333; border-radius: 8px; 
                     display: flex; justify-content: center; align-items: center; color: #ccc;">
                <div style="text-align: center;">
                    <h3>모델이 표시될 영역</h3>
                    <p>모델을 업로드하고 '적용 및 보기' 버튼을 클릭하세요</p>
                </div>
            </div>
            """)
            
            # 모델 뷰어 바로 아래에 프롬프트 입력 영역 배치
            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    prompt_input = gr.Textbox(
                        label="프롬프트 입력",
                        placeholder="캐릭터에 적용할 프롬프트를 입력하세요...",
                        lines=3
                    )
                
                with gr.Column(scale=1, min_width=120):
                    prompt_btn = gr.Button("전송", variant="secondary")
            
            # 프롬프트 결과 출력 영역
            prompt_result = gr.HTML(visible=True)
    
    # 버튼 클릭 이벤트 처리를 수정하여 NPY 파일도 처리하도록 함
    def process_animation(skin, anim):
        """애니메이션 파일 형식에 따라 적절한 처리 함수를 호출"""
        if anim is None or skin is None:
            return """
            <div style="width: 100%; height: 500px; background-color: #333; border-radius: 8px; 
                     display: flex; justify-content: center; align-items: center; color: #ccc;">
                <div style="text-align: center;">
                    <h3>오류</h3>
                    <p>스킨 모델과 애니메이션 파일을 모두 업로드해야 합니다</p>
                </div>
            </div>
            """
            
        file_ext = Path(anim.name).suffix.lower()
        
        if file_ext == '.npy':
            # NPY 파일 처리 (SMPL 애니메이션)
            try:
                # apply_smpl_animation 대신 apply_to_glb 사용
                from smpl_animation import apply_to_glb
                return apply_to_glb(skin, anim, VIEWER_PATH, MODELS_DIR)
            except Exception as e:
                return f"""
                <div style="width: 100%; height: 500px; background-color: #333; border-radius: 8px; 
                         display: flex; justify-content: center; align-items: center; color: #ccc;">
                    <div style="text-align: center;">
                        <h3>SMPL 애니메이션 적용 오류</h3>
                        <p>{str(e)}</p>
                    </div>
                </div>
                """
        else:
            # 기존 GLB/BVH 파일 처리
            return apply_animation(skin, anim, VIEWER_PATH, MODELS_DIR)
    
    # 버튼 클릭 이벤트 처리 함수 업데이트
    apply_btn.click(
        fn=process_animation,
        inputs=[skin_model, anim_model],
        outputs=viewer
    )
    
    # 프롬프트 전송 버튼 클릭 이벤트 처리
    prompt_btn.click(
        fn=send_prompt,
        inputs=prompt_input,
        outputs=prompt_result
    )
# 웹 서버 실행
if __name__ == "__main__":
    # FastAPI 서버를 별도 스레드에서 실행
    api_thread = threading.Thread(target=run_fastapi_server, daemon=True)
    api_thread.start()
    
    print("FastAPI 서버가 백그라운드에서 시작되었습니다 (포트: 8000)")
    print("API 문서: http://localhost:8000/docs")
    
    # Gradio 인터페이스 실행
    demo.launch()