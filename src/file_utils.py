"""
파일 처리와 관련된 유틸리티 함수들
"""

import os
import shutil
import uuid
from pathlib import Path

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

def send_prompt(prompt_text):
    """
    프롬프트를 localhost:8000으로 전송합니다.
    
    Args:
        prompt_text: 사용자가 입력한 프롬프트 텍스트
    
    Returns:
        dict: API 응답 데이터
    """
    import json
    import requests
    
    if not prompt_text.strip():
        return {
            "status": "error",
            "message": "프롬프트를 입력해주세요."
        }
    
    try:
        response = requests.post(
            'http://localhost:8000/api/prompt',
            headers={'Content-Type': 'application/json'},
            data=json.dumps({'prompt': prompt_text}),
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "status": "error",
                "message": f"서버 응답 오류: {response.status_code}",
                "details": response.text
            }
            
    except requests.exceptions.ConnectionError:
        return {
            "status": "error",
            "message": "서버 연결 오류",
            "details": "서버가 실행 중인지 확인해주세요 (localhost:8000)"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": "프롬프트 전송 실패",
            "details": str(e)
        }

def poll_task_status(task_id, callback):
    """
    작업 상태를 주기적으로 확인하고 콜백 함수를 통해 결과를 전달합니다.
    
    Args:
        task_id: 확인할 작업의 ID
        callback: 상태 업데이트를 처리할 콜백 함수
    
    Returns:
        None
    """
    import time
    import requests
    
    while True:
        try:
            response = requests.get(f'http://localhost:8000/api/tasks/{task_id}')
            if response.status_code == 200:
                status_data = response.json()
                callback(status_data)
                
                # 작업이 완료되면 종료
                if status_data.get('status') in ['completed', 'failed']:
                    break
            
            # 1초 대기
            time.sleep(1)
            
        except requests.exceptions.ConnectionError:
            callback({
                'status': 'error',
                'message': '서버 연결 오류'
            })
            break
        except Exception as e:
            callback({
                'status': 'error',
                'message': str(e)
            })
            break
