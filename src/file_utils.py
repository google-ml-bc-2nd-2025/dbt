"""
파일 처리와 관련된 유틸리티 함수들
"""

import os
import shutil
import uuid
from pathlib import Path
import requests
import json
import time
from typing import Dict, Any, Optional

def save_model(file_obj, prefix, models_dir):
    """
    모델 파일을 저장하고 URL을 반환합니다.
    
    Args:
        file_obj: 업로드된 파일 객체
        prefix: 파일 접두어 (예: "skin", "anim")
        models_dir: 모델이 저장될 디렉토리 경로
    
    Returns:
        저장된 파일의 URL 경로
    """
    if file_obj is None:
        return None
    
    # 고유 ID 생성 및 파일 저장
    unique_id = str(uuid.uuid4())[:8]
    
    # 원본 파일 확장자 유지
    original_ext = Path(file_obj.name).suffix.lower()
    filename = f"{prefix}_{unique_id}{original_ext}"
    model_path = models_dir / filename
    
    # 파일 복사
    file_path = file_obj.name if hasattr(file_obj, "name") else file_obj
    shutil.copy2(file_path, model_path)
    
    # 파일명만 반환 (전체 경로 대신)
    return f"/file={model_path}"

def apply_animation(skin_model, anim_model, viewer_path, models_dir):
    """
    스킨 모델과 애니메이션 모델을 뷰어에 적용합니다.
    
    Args:
        skin_model: 스킨 모델 파일 객체
        anim_model: 애니메이션 모델 파일 객체
        viewer_path: 뷰어 HTML 파일 경로
        models_dir: 모델이 저장될 디렉토리 경로
    
    Returns:
        HTML 문자열 (iframe으로 뷰어 표시)
    """
    if skin_model is None:
        return "스킨 모델을 먼저 업로드해주세요."
    
    # 모델 저장 및 URL 생성
    skin_url = save_model(skin_model, "skin", models_dir)
    anim_url = None
    anim_type = "glb"  # 기본값
    
    if anim_model:
        anim_url = save_model(anim_model, "anim", models_dir)
        # 파일 확장자로 애니메이션 타입 결정
        anim_ext = Path(anim_model.name).suffix.lower()
        if anim_ext == '.bvh':
            anim_type = "bvh"
    
    # 뷰어 URL 생성 (파일 URL 형식으로)
    viewer_url = f"/file={viewer_path}?skin={skin_url}"
    if anim_url:
        viewer_url += f"&anim={anim_url}&animType={anim_type}"
    
    # iframe으로 뷰어 표시 (JavaScript 통신 기능 추가)
    return f"""
    <div style="width: 100%; height: 500px; border-radius: 8px; overflow: hidden;">
        <iframe id="model-viewer-frame" src="{viewer_url}" style="width: 100%; height: 100%; border: none;"></iframe>
    </div>
    <p style="margin-top: 8px; color: #666; font-size: 0.9em;">
        마우스를 사용하여 모델을 회전하고 확대/축소할 수 있습니다.
    </p>
    <script>
        // iframe에서 메시지 수신
        window.addEventListener('message', function(event) {{
            if (event.data.type === 'modelLoadStatus') {{
                console.log('모델 로드 상태:', event.data.status);
                // 추가 작업 가능
            }}
        }});
    </script>
    """

def send_prompt(prompt: str) -> Dict[str, Any]:
    """
    프롬프트를 백엔드로 전송
    
    Args:
        prompt (str): 사용자 입력 프롬프트
        
    Returns:
        Dict[str, Any]: 응답 데이터
    """
    try:
        response = requests.post(
            "http://localhost:8000/api/prompt",
            json={"prompt": prompt},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.RequestException as e:
        return {
            "status": "error",
            "message": f"백엔드 서버(localhost:8000)에 연결할 수 없습니다: {str(e)}"
        }

def check_task_status(task_id: str) -> Dict[str, Any]:
    """
    작업 상태 확인
    
    Args:
        task_id (str): 확인할 작업의 ID
        
    Returns:
        Dict[str, Any]: 작업 상태 정보
    """
    try:
        response = requests.get(
            f"http://localhost:8000/api/tasks/{task_id}",
            timeout=10
        )
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.RequestException as e:
        return {
            "status": "error",
            "message": f"작업 상태 확인 중 오류 발생: {str(e)}"
        }

def poll_task_status(task_id: str, callback, interval: float = 1.0, timeout: float = 300.0) -> None:
    """
    작업 상태를 주기적으로 확인
    
    Args:
        task_id (str): 확인할 작업의 ID
        callback: 상태 업데이트 시 호출할 콜백 함수
        interval (float): 확인 주기 (초)
        timeout (float): 최대 대기 시간 (초)
    """
    start_time = time.time()
    
    while True:
        # 타임아웃 체크
        if time.time() - start_time > timeout:
            callback({
                "status": "error",
                "message": "작업 시간이 초과되었습니다."
            })
            break
            
        # 상태 확인
        status = check_task_status(task_id)
        callback(status)
        
        # 작업 완료 또는 실패 시 종료
        if status.get("status") in ["completed", "failed", "error"]:
            break
            
        # 대기
        time.sleep(interval)
