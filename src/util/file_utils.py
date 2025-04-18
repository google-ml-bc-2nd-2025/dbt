"""
파일 처리와 관련된 유틸리티 함수들
"""

import os
import shutil
import uuid
import numpy as np
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
    웹 기반 3D 뷰어에서 스킨 모델(GLB 등)과 애니메이션 모델(GLB, NPY, NPZ, BVH 등)을 
    함께 시각화할 수 있도록 iframe HTML 코드를 생성하는 함수입니다.     

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
    
    print("\n===== 애니메이션 모델 상세 정보 =====")
    print(f"애니메이션 모델 타입: {type(anim_model)}")
    print(f"애니메이션 모델 속성: {dir(anim_model)}")

    # 모델 저장 및 URL 생성
    skin_url = save_model(skin_model, "skin", models_dir)
    anim_url = None
    anim_type = "glb"  # 기본값
    
    # Check for NPZ or NPY animation files and print their contents
    if anim_model:
        print(f"model name = {anim_model.name}")
        # 파일 크기 확인
        try:
            file_size = os.path.getsize(anim_model.name)
            print(f"파일 크기: {file_size} 바이트 ({file_size/1024/1024:.2f} MB)")
        except Exception as e:
            print(f"파일 크기 확인 실패: {e}")

        # 파일 존재 여부 확인
        print(f"파일 존재 여부: {os.path.exists(anim_model.name)}")
        
        # 파일 경로 확인
        print(f"파일 절대 경로: {os.path.abspath(anim_model.name)}")
        
        # 파일 확장자 확인
        anim_ext = Path(anim_model.name).suffix.lower()
        print(f"파일 확장자: {anim_ext}")

        print(f"anim_ext = {anim_ext}")
        # 파일 내용 확인 (NPY/NPZ 파일)
        if anim_ext in ['.npz', '.npy']:
            try:
                data = np.load(anim_model.name)
                print(f"애니메이션 데이터 ({anim_ext}) 내용:")
                
                if anim_ext == '.npz':
                    print(f"NPZ 파일 키: {list(data.keys())}")
                    for key in data.keys():
                        array = data[key]
                        print(f"  - {key}: 형태 {array.shape}, 타입 {array.dtype}")
                        print(f"    값 범위: {np.min(array)} ~ {np.max(array)}")
                        print(f"    첫 5개 요소: {array.flatten()[:5]}")
                else:  # .npy 파일
                    print(f"NPY 배열 형태: {data.shape}, 타입: {data.dtype}")
                    print(f"데이터 범위: {np.min(data)} ~ {np.max(data)}")
                    print(f"첫 5개 요소: {data.flatten()[:5]}")
                    print(f"데이터 통계: 평균 = {np.mean(data)}, 표준편차 = {np.std(data)}")
                    
                    # 만약 shapes 데이터인 경우
                    if len(data.shape) == 2 and data.shape[1] == 10:
                        print("SMPL 형상 파라미터로 추정됩니다.")
                    # 만약 poses 데이터인 경우
                    elif len(data.shape) == 2 and (data.shape[1] == 72 or data.shape[1] % 3 == 0):
                        print("SMPL 포즈 파라미터로 추정됩니다.")
                        print(f"관절 수: {data.shape[1] // 3}")
                
            except Exception as e:
                print(f"애니메이션 데이터 로드 오류: {str(e)}")
        # GLB 파일인 경우
        elif anim_ext == '.glb':
            print("GLB 애니메이션 파일 감지됨")
            try:
                import pygltflib
                gltf = pygltflib.GLTF2().load(anim_model.name)
                print(f"GLB 정보: {len(gltf.nodes)}개 노드, {len(gltf.animations)}개 애니메이션")
                
                if gltf.animations:
                    for i, anim in enumerate(gltf.animations):
                        print(f"애니메이션 {i}: {anim.name}, {len(anim.channels)}개 채널, {len(anim.samplers)}개 샘플러")
                else:
                    print("애니메이션 모델 없음: None")
            except Exception as e:
                print(f"GLB 파일 분석 오류: {str(e)}")
            
            print("====================================\n")

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
        HTML 문자열 (프롬프트 전송 결과 표시)
    """
    import json
    import requests
    
    import json
    import requests
    
    if not prompt_text.strip():
        return "프롬프트를 입력해주세요."
    
    # 디버깅을 위한 로그 추가
    print(f"send_prompt() 호출됨 - 프롬프트: {prompt_text}")
    
    # 단일 호출 보장을 위한 플래그 사용 (필요시)
    # if hasattr(send_prompt, '_is_running') and send_prompt._is_running:
    #     return "이전 요청이 처리 중입니다..."
    # send_prompt._is_running = True
    
    try:
        response = requests.post(
            'http://localhost:8000/api/prompt',
            headers={'Content-Type': 'application/json'},
            data=json.dumps({'prompt': prompt_text}),
            timeout=10
        )
        
        # 실행 완료 후 플래그 해제 (필요시)
        # send_prompt._is_running = False
        
        if response.status_code == 200:
            result_data = response.json()
            return f"""
            <div id="prompt-result">
                <p style="color: green;">프롬프트 전송 성공!</p>
                <pre>{json.dumps(result_data, indent=2, ensure_ascii=False)}</pre>
            </div>
            """
        else:
            return f"""
            <div id="prompt-result">
                <p style="color: red;">프롬프트 전송 실패: 서버 응답 코드 {response.status_code}</p>
                <p>{response.text}</p>
            </div>
            """
    except requests.exceptions.ConnectionError:
        # send_prompt._is_running = False  # 예외 발생 시에도 플래그 해제
        return f"""
        <div id="prompt-result">
            <p style="color: red;">프롬프트 전송 실패: 서버 연결 오류</p>
            <p>서버가 실행 중인지 확인해주세요 (localhost:8000)</p>
        </div>
        """
    except Exception as e:
        # send_prompt._is_running = False  # 예외 발생 시에도 플래그 해제
        return f"""
        <div id="prompt-result">
            <p style="color: red;">프롬프트 전송 실패: {str(e)}</p>
        </div>
        """
    print(f"send prompt() 호출됨 - 프롬프트: {prompt_text}")
    
    # Python에서 직접 API 호출 수행
    try:
        response = requests.post(
            'http://localhost:8000/api/prompt',
            headers={'Content-Type': 'application/json'},
            data=json.dumps({'prompt': prompt_text}),
            timeout=10
        )
        
        if response.status_code == 200:
            result_data = response.json()
            return f"""
            <div id="prompt-result">
                <p style="color: green;">프롬프트 전송 성공!</p>
                <pre>{json.dumps(result_data, indent=2, ensure_ascii=False)}</pre>
            </div>
            """
        else:
            return f"""
            <div id="prompt-result">
                <p style="color: red;">프롬프트 전송 실패: 서버 응답 코드 {response.status_code}</p>
                <p>{response.text}</p>
            </div>
            """
    except requests.exceptions.ConnectionError:
        return f"""
        <div id="prompt-result">
            <p style="color: red;">프롬프트 전송 실패: 서버 연결 오류</p>
            <p>서버가 실행 중인지 확인해주세요 (localhost:8000)</p>
        </div>
        """
    except Exception as e:
        return f"""
        <div id="prompt-result">
            <p style="color: red;">프롬프트 전송 실패: {str(e)}</p>
        </div>
        """
