"""
3D 모델 뷰어 웹 애플리케이션의 메인 모듈
"""

import gradio as gr
import os, atexit
import threading
from pathlib import Path
import json
from file_utils import send_prompt, poll_task_status

# 프로젝트 내 사용자 정의 모듈 임포트
from viewer_template import create_viewer_html
from apis import start_api_server
# 탭 모듈 임포트
from animation_tab import create_animation_tab
from dataset_tab import create_dataset_tab
from edit_dataset_tab import create_edit_dataset_tab

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

def format_status_html(status_data: dict) -> str:
    """작업 상태를 HTML로 포맷팅"""
    if status_data.get("status") == "error":
        return f"""
        <div style="color: red;">
            <p>오류 발생: {status_data.get('message', '알 수 없는 오류')}</p>
        </div>
        """
    
    elif status_data.get("status") == "processing":
        return f"""
        <div style="color: blue;">
            <p>작업 처리 중...</p>
            <p>잠시만 기다려주세요.</p>
        </div>
        """
    
    elif status_data.get("status") == "completed":
        text_result = status_data.get("text_result", "")
        animation_result = status_data.get("animation_result", {})
        timing = animation_result.get("timing", {})
        
        return f"""
        <div style="color: green;">
            <h3>작업 완료!</h3>
            <h4>정제된 텍스트:</h4>
            <pre>{text_result}</pre>
            <h4>처리 시간:</h4>
            <ul>
                <li>텍스트 정제: {timing.get('text_refinement_duration', 0):.2f}초</li>
                <li>모션 생성: {timing.get('motion_generation_duration', 0):.2f}초</li>
                <li>총 소요 시간: {timing.get('total_duration', 0):.2f}초</li>
            </ul>
        </div>
        """
    
    return f"""
    <div>
        <pre>{json.dumps(status_data, indent=2, ensure_ascii=False)}</pre>
    </div>
    """

def handle_prompt(prompt: str, status_output) -> str:
    """프롬프트 처리 및 상태 업데이트"""
    if not prompt.strip():
        return "프롬프트를 입력해주세요."
    
    # 프롬프트 전송
    result = send_prompt(prompt)
    
    if result.get("status") == "error":
        return format_status_html(result)
    
    task_id = result.get("task_id")
    if not task_id:
        return format_status_html({
            "status": "error",
            "message": "작업 ID를 받지 못했습니다."
        })
    
    # 초기 상태 표시
    initial_status = format_status_html({
        "status": "processing",
        "message": "작업이 시작되었습니다."
    })
    
    # 백그라운드에서 상태 확인
    threading.Thread(
        target=poll_task_status,
        args=(task_id, lambda s: status_output.update(format_status_html(s))),
        daemon=True
    ).start()
    
    return initial_status

# FastAPI 서버 실행 (별도 스레드에서)
def run_fastapi_server():
    """별도 스레드에서 FastAPI 서버를 실행합니다."""
    start_api_server(port=8001)

# 서버 종료 관리
def cleanup():
    """애플리케이션 종료 시 정리 작업을 수행합니다."""
    print("서버가 종료됩니다. 정리 작업 수행 중...")
    print("정리 작업 완료.")

# 종료 시 정리 함수 등록
atexit.register(cleanup)

# 웹 서버 실행
if __name__ == "__main__":
    # FastAPI 서버를 별도 스레드에서 실행
    api_thread = threading.Thread(target=run_fastapi_server, daemon=True)
    api_thread.start()
    
    print("FastAPI 서버가 백그라운드에서 시작되었습니다 (포트: 8001)")
    print("API 문서: http://localhost:8001/docs")
    
    # Gradio 인터페이스 생성
    with gr.Blocks(title="AssetSmith") as demo:
        gr.Markdown("# AssetSmith")
        
        # 탭 인터페이스 생성
        with gr.Tabs():
            with gr.TabItem("애니메이션 생성"):
                create_animation_tab(VIEWER_PATH, MODELS_DIR)
            
            with gr.TabItem("애니메이션 학습 데이터셋 생성"):
                create_edit_dataset_tab(MODELS_DIR)
            
            with gr.TabItem("학습 데이터 수정"):
                create_dataset_tab(MODELS_DIR)
    
    # Gradio 실행
    demo.launch(server_port=8001)