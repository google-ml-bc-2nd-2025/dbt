"""
파일 처리와 관련된 유틸리티 함수들
"""

import os
import shutil
import uuid
import numpy as np
from pathlib import Path
import tempfile
from render.smpl_animation import apply_to_glb

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
    print(f"파일 저장 경로: {model_path}")
    print(f"파일 복사: {file_path} -> {model_path}")
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
    print(f"뷰어 URL: {viewer_url}")
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

def axis_angle_to_rotation_6d(pose_aa):  # (T, 72) or (T, 24, 3)
    """
    축-각도(axis-angle) 회전 표현을 6D 회전 표현으로 변환합니다.
    
    Args:
        pose_aa: 축-각도 포즈 배열 (T, 72) 또는 (T, 24, 3)
    
    Returns:
        6D 회전 표현 배열 (T, 22*6=132)
    """
    if len(pose_aa.shape) == 3:  # (T, 24, 3)
        T, joints, _ = pose_aa.shape
        pose_aa_reshaped = pose_aa.reshape(T, joints * 3)
    else:  # (T, 72)
        T = pose_aa.shape[0]
        pose_aa_reshaped = pose_aa
    
    pose_6d = np.zeros((T, 22 * 6), dtype=np.float32)  # 첫 22개 관절만 사용

    for t in range(T):
        frame_aa = pose_aa_reshaped[t].reshape(24, 3)  # (24, 3)
        frame_6d = []
        for joint_idx in range(22):  # 첫 22개 관절만 사용
            rotmat = R.from_rotvec(frame_aa[joint_idx]).as_matrix()  # (3, 3)
            rot_6d = rotmat[:, :2].reshape(6)  # 앞 두 열 → (6,)
            frame_6d.append(rot_6d)
        pose_6d[t] = np.concatenate(frame_6d)

    return pose_6d  # (T, 132)

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
    import datetime
    import numpy as np
    import gradio as gr
    from pathlib import Path
    
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
            # {'smpl_data': {'joint_map': [...], 'thetas': [...], 'root_translation': [...]}

            smpl_format = {}
            smpl_format["pose"] = result_data["smpl_data"]["joint_map"]
            smpl_format["betas"] = result_data["smpl_data"]["thetas"]
            smpl_format["trans"] = result_data["smpl_data"]["root_translation"]

            # static 에 저장
            # 출력 파일 경로 생성
            STATIC_DIR = Path(__file__).parent / "static" / "generated"
            STATIC_DIR.mkdir(exist_ok=True)
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"generated_{current_time}.npy"
            output_dir = STATIC_DIR
            output_path = output_dir / output_name
            
            try:
                # 여기서 실제 GLB/FBX 변환 코드 구현 필요
                # 임시 구현: 더미 데이터 생성
                frame_count = 60  # MotionCLIP 기본 프레임 수
                
                # MotionCLIP 포맷에 맞게 데이터 생성
                pose = np.array(smpl_format["pose"], dtype=np.float32)  # [60, 72] 형태의 포즈
                trans = np.array(smpl_format["trans"], dtype=np.float32)  # [60, 3] 형태의 트랜스폼
                betas = np.array(smpl_format["betas"], dtype=np.float32)  # [60, 10] 형태의 베타스
                # (프레임 수, 24, 3) -> (프레임 수, 72) 형태로 변환
                pose_flat = pose.reshape(frame_count, -1)
                
                # 6D 회전 표현으로 변환
                pose_6d = axis_angle_to_rotation_6d(pose)

                # NPY 파일 저장
                data = {
                    'poses': pose_flat,
                    'fps': 30
                }
                if trans is not None:
                    data['trans'] = trans
                if betas is not None:
                    data['betas'] = betas
                if pose_6d is not None:
                    data['poses_6d'] = pose_6d

                np.save(output_path, data)
                print(f"파일 저장 완료: {output_path}")

                return process_animation_in_generated(output_path)

                # return output_path
            except Exception as e:
                print(f"파일 저장 오류: {str(e)}")
                return None
                # return f"""
                # <div id="prompt-result">
                #     <p style="color: red;">프롬프트 전송 실패: 파일 저장 오류</p>
                #     <p>{str(e)}</p>
                # </div>
                # """

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

# 버튼 클릭 이벤트 처리를 위한 함수 정의
def process_animation_in_generated(anim_path):
    """애니메이션 파일 형식에 따라 적절한 처리 함수를 호출"""
    # 스킨이 없으면 오류 메시지 표시
    if anim_path is None:
        return """
        <div style="width: 100%; height: 500px; background-color: #333; border-radius: 8px; 
                    display: flex; justify-content: center; align-items: center; color: #ccc;">
            <div style="text-align: center;">
                <h3>오류</h3>
                <p>모델이 정상적이지 않습니다.</p>
            </div>
        </div>
        """
    
    # Constants needed for further processing
    MODELS_DIR = Path(__file__).parent / "static" / "models"
    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    VIEWER_PATH = Path(__file__).parent / "static" / "viewer" / "index.html"

    # Create a temporary file object from the npy file
    with tempfile.NamedTemporaryFile(suffix=Path(anim_path).suffix, delete=False) as tmp:
        # Copy the content of the original file
        tmp.write(open(anim_path, 'rb').read())
        tmp_path = tmp.name

    # Create a MockFile object that mimics gr.File
    class MockFile:
        def __init__(self, path):
            self.name = path
            
    # Create anim object similar to what gr.File would return
    anim = MockFile(tmp_path)

    # Also need to have a skin model for the viewer
    # Use default skin.glb from static directory if available
    skin_path = Path(__file__).parent / "static" / "models" / "tpose.glb"
    if skin_path.exists():
        skin = MockFile(str(skin_path))
    else:
        return f"""
        <div style="width: 100%; height: 500px; background-color: #333; border-radius: 8px; 
                    display: flex; justify-content: center; align-items: center; color: #ccc;">
            <div style="text-align: center;">
                <h3>기본 포즈 skin 파일없음</h3>
                <p>{str(e)}</p>
            </div>
        </div>
        """

    file_ext = Path(anim.name).suffix.lower()
    
    # NPY 또는 NPZ 파일 처리 (SMPL 애니메이션)
    if file_ext in ['.npy', '.npz']:
        try:
            anim = apply_to_glb(skin, anim, VIEWER_PATH, MODELS_DIR, return_type='glb')
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
    # 기존 GLB/BVH 파일 처리
    print(f"skin = {skin}")
    print(f"anim = {anim}")
    return apply_animation(skin, anim, VIEWER_PATH, MODELS_DIR)