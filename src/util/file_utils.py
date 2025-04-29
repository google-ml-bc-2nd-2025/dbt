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
from model.smpl import joint_positions as smpl_default_positions
import base64
import numpy as np
from scipy.spatial.transform import Rotation as R

def apply_animation(skin_model, anim_model, viewer_path, models_dir, file_ext="glb"):
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
    print(f"file_ext = {file_ext}")
    if file_ext == "glb":
        if skin_model is None:
            return "스킨 모델을 먼저 업로드해주세요."
        else:
            skin_url = save_model(skin_model, "skin", models_dir)
    else:
        skin_url = None

    # 모델 저장 및 URL 생성
    
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
    viewer_url = f"/file={viewer_path}?"
    if skin_url:
        viewer_url += f"skin={skin_url}"
    else:
        viewer_url += f"skin="
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
            'http://47.186.55.156:57179/predict',
            headers={'Content-Type': 'application/json'},
            json={
                'prompt': prompt_text,
                "num_repetitions": 1,
                "output_format": "json_file"
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            try:
                json_file = result.get('json_file', {})
                for key in json_file.keys():
                    print(f"Key: {key}")

                # with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as json_tmp_file:
                #     json.dump(json_file, json_tmp_file, ensure_ascii=False, indent=4)
                #     json_tmp_path = json_tmp_file.name
                #     print(f"JSON 데이터를 임시 파일에 저장: {json_tmp_path}")
            except Exception as e:
                print(f"JSON 파일 저장 오류: {str(e)}")
                raise ValueError(str(e))

            # thetas를 numpy 배열로 변환
            thetas = json_file.get('motions', [])
            pose_array = np.array(thetas, dtype=np.float32)
            # root_translation을 numpy 배열로 변환
            root_translation = json_file.get('root_translation', [])
            trans_array = np.array(root_translation, dtype=np.float32)
            # betas는 빈 배열로 설정
            betas_array = np.array([], dtype=np.float32)
            
            # 전체 모션 데이터를 하나의 numpy 배열로 결합
            motion_array = np.concatenate([
                pose_array.reshape(-1),
                betas_array,
                trans_array.reshape(-1)
            ])

            print(f'result_data = {response.status_code}, {motion_array.shape}')
            animation_data = pose_array
            print(f'motion_data is {type(animation_data)} {pose_array.shape}')
            try:
                motion_data_array = animation_data.reshape(120, 22, 3)
                print(f'motion_data_array is {motion_data_array.shape}')
            except ValueError as e:
                print(f"Reshape error: {e}. Ensure animation_data has the correct size.")
                return f"애니메이션 데이터 크기가 올바르지 않습니다. 오류: {e}"
            
            try:
                # 딕셔너리로 래핑하여 #terminalSelection과 같은 형식으로 만들기
                motion_dict = {
                    'motion': motion_data_array,
                    'text': ['good job'],
                    'lengths': np.array([120]),
                    'num_samples': 1,
                    'num_repetitions': 1
                }
                pose_data = motion_dict
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
            except Exception as e:
                print(f"요청 오류: {str(e)}")
            
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