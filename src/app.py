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
from util.i18n import translations  # 다국어 지원 모듈 임포트
import os
from dotenv import load_dotenv

load_dotenv()
LANG_CODE = os.getenv("LANGUAGE", "en")  # 환경변수에서 언어 설정을 가져옵니다.

# 정적 파일 디렉토리 생성
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
MODELS_DIR = STATIC_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# 뷰어 템플릿 HTML 파일 경로
TEMPLATE_PATH = STATIC_DIR / "viewer.html"


# 서버 종료 관리 추가
def cleanup():
    """애플리케이션 종료 시 정리 작업을 수행합니다."""
    print("서버가 종료됩니다. 정리 작업 수행 중...")
    # 필요한 정리 작업 수행
    print("정리 작업 완료.")

# 종료 시 정리 함수 등록
atexit.register(cleanup)

# Gradio 인터페이스 생성
with gr.Blocks(title="Dataset Building Tool for MDM") as demo:
    gr.Markdown("# Dataset Building Tool for MDM")

    # 탭 인터페이스 생성
    with gr.Tabs():
        with gr.TabItem(translations[LANG_CODE]["tab_title_01"]):
            create_animation_generation_tab(LANG_CODE, TEMPLATE_PATH, MODELS_DIR)

        with gr.TabItem(translations[LANG_CODE]["tab_title_02"]):
            create_animation_viewer_tab(LANG_CODE, TEMPLATE_PATH, MODELS_DIR)
            
        with gr.TabItem(translations[LANG_CODE]["tab_title_03"]):
            create_dataset_create_tab(LANG_CODE, MODELS_DIR)

# 웹 서버 실행
if __name__ == "__main__":
    # Gradio 인터페이스 실행
    demo.queue().launch()