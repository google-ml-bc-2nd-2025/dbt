"""
3D 모델 뷰어 웹 애플리케이션의 메인 모듈
"""

import gradio as gr
import os, atexit
import threading
from pathlib import Path

# 프로젝트 내 사용자 정의 모듈 임포트
from viewer_template import create_viewer_html
from apis import start_api_server
# 탭 모듈 임포트
from animation_tab import create_animation_tab
from dataset_tab import create_dataset_tab

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
with gr.Blocks(title="AssetSmith") as demo:
    gr.Markdown("# AssetSmith")
    
    # 탭 인터페이스 생성
    with gr.Tabs():
        with gr.TabItem("애니메이션 생성"):
            create_animation_tab(VIEWER_PATH, MODELS_DIR)
            
        with gr.TabItem("애니메이션 훈련 데이터셋 생성"):
            create_dataset_tab(MODELS_DIR)

# 웹 서버 실행
if __name__ == "__main__":
    # FastAPI 서버를 별도 스레드에서 실행
    api_thread = threading.Thread(target=run_fastapi_server, daemon=True)
    api_thread.start()
    
    print("FastAPI 서버가 백그라운드에서 시작되었습니다 (포트: 8000)")
    print("API 문서: http://localhost:8000/docs")
    
    # Gradio 인터페이스 실행
    demo.launch()