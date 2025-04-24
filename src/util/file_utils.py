"""
파일 처리와 관련된 유틸리티 함수들
"""

import os
import numpy as np
from pathlib import Path
import tempfile
from render.smpl_animation import apply_to_glb
from render.humanml3d_renderer import render_humanml3d
from util.model_utils import save_model
import requests
import json
import time
import gradio as gr

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

def send_prompt(prompt_text, progress=gr.Progress(track_tqdm=True)):
    """
    프롬프트를 전송하고 결과를 처리하는 함수
    
    Args:
        prompt_text: 사용자 입력 프롬프트
        progress: Gradio Progress 컴포넌트
    """
    
    if not prompt_text.strip():
        return "프롬프트를 입력해주세요."
    
    try:
        # 프롬프트 전송
        progress(0, desc="프롬프트 전송 중...")
        response = requests.post(
            'http://localhost:8000/api/prompt',
            headers={'Content-Type': 'application/json'},
            json={'prompt': prompt_text},
        )
        
        if response.status_code == 200:
            result_data = response.json()
            task_id = result_data.get('task_id')
            
            if not task_id:
                return f"""
                <div id="prompt-result">
                    <p style="color: red;">작업 ID가 없습니다</p>
                    <p style="color: #666; font-size: 0.9em;">서버 응답에 작업 ID가 포함되어 있지 않습니다.</p>
                </div>
                """
            
            # 상태 추적 시작
            max_retries = 60  # 180초 (3초 간격)
            retry_count = 0
            last_state = None
            
            for i in progress.tqdm(range(max_retries), desc="작업 진행 중..."):
                try:
                    status_response = requests.get(
                        f'http://localhost:8000/api/tasks/{task_id}',
                    )
                    
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        current_status = status_data.get('status')
                        
                        # 상태가 변경되었을 때만 로그 출력
                        if current_status != last_state:
                            print(f"작업 상태 변경: {last_state} -> {current_status}")
                            last_state = current_status

                        # None 상태나 processing 상태일 때는 계속 대기
                        if current_status is None or current_status == 'processing':
                            time.sleep(3)  # 3초 대기
                            continue

                        if current_status == 'completed':
                            # 모션 데이터 조회
                            motion_data = status_data.get('data', {}).get('motion', {})
                            print(f"motion_data = {motion_data}")
                            print(f"status_data = {status_data}")

                            try:
                                # motion_data가 문자열인 경우 JSON으로 파싱
                                if isinstance(motion_data, str):
                                    motion_data = json.loads(motion_data)
                                
                                # base64로 인코딩된 데이터 디코딩
                                if 'data' in motion_data:
                                    import base64
                                    import numpy as np
                                    from scipy.spatial.transform import Rotation as R
                                    
                                    # base64 디코딩
                                    decoded_data = base64.b64decode(motion_data['data'])
                                    
                                    # numpy 배열로 변환
                                    motion_array = np.frombuffer(decoded_data, dtype=motion_data['dtype'])
                                    
                                    # shape에 맞게 reshape
                                    motion_array = motion_array.reshape(motion_data['shape'])
                                    
                                    # 메타데이터에서 pose와 trans의 shape 가져오기
                                    pose_shape = status_data.get('data', {}).get('animation_result', {}).get('metadata', {}).get('pose_shape', (1, 120, 24, 3))
                                    trans_shape = status_data.get('data', {}).get('animation_result', {}).get('metadata', {}).get('trans_shape', (1, 120, 3))
                                    
                                    # pose와 trans 데이터 분리
                                    pose_size = np.prod(pose_shape)
                                    trans_size = np.prod(trans_shape)
                                    
                                    pose_data = motion_array[:pose_size].reshape(pose_shape)
                                    trans_data = motion_array[-trans_size:].reshape(trans_shape)
                                    
                                    smpl_format = {
                                        "pose": pose_data.tolist(),
                                        "betas": [],
                                        "trans": trans_data.tolist()
                                    }
                                    if pose_data.shape[2] == 24:
                                        print(f"수정 전 pose 데이터 형태: {pose_data.shape}")
                                        pose_data = pose_data[..., :22, :] 
                                        pose_data = pose_data[0]
                                        # Convert SMPL format to HumanML3D format
                                        # In SMPL, rotations are in axis-angle format
                                        # We need to convert to match HumanML3D joint orientation

                                        # Convert axis-angle to rotation matrices
                                        rot_matrices = R.from_rotvec(pose_data.reshape(-1, 3)).as_matrix()
                                        rot_matrices = rot_matrices.reshape(pose_data.shape[0], pose_data.shape[1], 3, 3)

                                        # Apply rotation offsets to match HumanML3D convention
                                        # X rotation offset of -π/2 on specific joints (could be adjusted based on testing)
                                        for joint_idx in [7, 8, 10, 11]:  # ankles and feet
                                            offset_rot = R.from_euler('x', -np.pi/2).as_matrix()
                                            rot_matrices[:, joint_idx] = rot_matrices[:, joint_idx] @ offset_rot

                                        # Convert back to axis-angle representation
                                        pose_data = R.from_matrix(rot_matrices.reshape(-1, 3, 3)).as_rotvec()
                                        pose_data = pose_data.reshape(-1, 22, 3)

                                        print(f"수정 후 pose 데이터 형태: {pose_data.shape}")

                                    # Update the SMPL format with modified pose data
                                    smpl_format["pose"] = pose_data.tolist()
                                    # 임시 파일로 numpy 데이터 저장
                                    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp_file:
                                        tmp_path = tmp_file.name
                                        # 변환된 pose 데이터를 numpy 파일로 저장
                                        np.save(tmp_path, pose_data)
                                        print(f"애니메이션 데이터를 임시 파일에 저장: {tmp_path}")

                                    # 임시 파일을 이용하여 애니메이션 처리
                                    try:
                                        print(f"애니메이션 변환 성공: {tmp_path}")
                                        return render_humanml3d(tmp_file)
                                    except Exception as e:
                                        print(f"애니메이션 변환 실패: {str(e)}")
                                        return f"""
                                        <div id="prompt-result">
                                            <p style="color: red;">애니메이션 처리 오류</p>
                                            <p style="color: #666; font-size: 0.9em;">{str(e)}</p>
                                        </div>
                                        """
                                else:
                                    return f"""
                                    <div id="prompt-result">
                                        <p style="color: green;">모션 생성 실패!</p>
                                        <pre>생성에 실패했습니다.</pre>
                                    </div>
                                    """
                                
                            except Exception as e:
                                return f"""
                                <div id="prompt-result">
                                    <p style="color: red;">데이터 변환 오류</p>
                                    <p style="color: #666; font-size: 0.9em;">{str(e)}</p>
                                </div>
                                """
                        
                        elif current_status == 'failed':
                            error_message = status_data.get('error', {}).get('message', '알 수 없는 오류가 발생했습니다.')
                            return f"""
                            <div id="prompt-result">
                                <p style="color: red;">모션 생성 실패</p>
                                <p style="color: #666; font-size: 0.9em;">{error_message}</p>
                            </div>
                            """
                    
                except requests.exceptions.RequestException as e:
                    print(f"상태 확인 중 네트워크 오류: {str(e)}")
                    retry_count += 1
                    continue
                
                retry_count += 1
            
            # 최대 재시도 횟수 도달
            return f"""
            <div id="prompt-result">
                <p style="color: orange;">모션 생성 시간 초과</p>
                <p style="color: #666; font-size: 0.9em;">서버가 바쁠 수 있습니다. 잠시 후 다시 시도해주세요.</p>
            </div>
            """
        else:
            return f"""
            <div id="prompt-result">
                <p style="color: red;">서버 오류 발생 (상태 코드: {response.status_code})</p>
                <p style="color: #666; font-size: 0.9em;">잠시 후 다시 시도해주세요.</p>
            </div>
            """
            
    except requests.exceptions.RequestException as e:
        return f"""
        <div id="prompt-result">
            <p style="color: red;">네트워크 오류 발생</p>
            <p style="color: #666; font-size: 0.9em;">{str(e)}</p>
        </div>
        """
    except Exception as e:
        return f"""
        <div id="prompt-result">
            <p style="color: red;">알 수 없는 오류 발생</p>
            <p style="color: #666; font-size: 0.9em;">{str(e)}</p>
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
    skin_path = Path(__file__).parent / "static" / "tpose.glb"

    if skin_path.exists():
        skin = MockFile(str(skin_path))
    else:
        return f"""
        <div style="width: 100%; height: 500px; background-color: #333; border-radius: 8px; 
                    display: flex; justify-content: center; align-items: center; color: #ccc;">
            <div style="text-align: center;">
                <h3>기본 포즈 skin 파일없음 in file_utils</h3>
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