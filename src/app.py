"""
3D 모델 뷰어 웹 애플리케이션의 메인 모듈
"""

import gradio as gr
import atexit
from pathlib import Path

# 프로젝트 내 사용자 정의 모듈 임포트
from util.viewer_template import create_viewer_html

# 탭 모듈 임포트
from page.animation_generation_tab import create_animation_generation_tab
from page.animation_viewer_tab import create_animation_viewer_tab
from page.dataset_creation_tab import create_dataset_create_tab  # 새로운 탭 모듈 임포트

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



# 서버 종료 관리 추가
def cleanup():
    """애플리케이션 종료 시 정리 작업을 수행합니다."""
    print("서버가 종료됩니다. 정리 작업 수행 중...")
    # 필요한 정리 작업 수행
    print("정리 작업 완료.")

# 종료 시 정리 함수 등록
atexit.register(cleanup)

# Gradio 인터페이스 생성
with gr.Blocks(title="Animation Tool") as demo:
    gr.Markdown("# Animation Tool")
    
    # 탭 인터페이스 생성
    with gr.Tabs():
        with gr.TabItem("애니메이션 뷰어"):
            create_animation_viewer_tab(VIEWER_PATH, MODELS_DIR)

        with gr.TabItem("애니메이션 생성"):
            create_animation_generation_tab(VIEWER_PATH, MODELS_DIR)
            
        with gr.TabItem("애니메이션 학습 데이터셋 생성"):
            create_dataset_create_tab(MODELS_DIR)

# 웹 서버 실행
if __name__ == "__main__":
    # Gradio 인터페이스 실행
    demo.queue().launch()