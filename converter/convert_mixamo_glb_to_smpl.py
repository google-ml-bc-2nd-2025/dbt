import os
import numpy as np
import trimesh
import pygltflib
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import re
import glob


'''
    mixamo glb 파일에서 SMPL 포즈 데이터 추출
    
'''


# SMPL 본 순서
SMPL_JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 'right_hand'
]

# Mixamo 이름 → SMPL 이름
MIXAMO_TO_SMPL = {
    'Hips': 'pelvis',
    'LeftUpLeg': 'left_hip',
    'RightUpLeg': 'right_hip',
    'Spine': 'spine1',
    'LeftLeg': 'left_knee',
    'RightLeg': 'right_knee',
    'Spine1': 'spine2',
    'LeftFoot': 'left_ankle',
    'RightFoot': 'right_ankle',
    'Spine2': 'spine3',
    'LeftToeBase': 'left_foot',
    'RightToeBase': 'right_foot',
    'Neck': 'neck',
    'LeftShoulder': 'left_collar',
    'RightShoulder': 'right_collar',
    'Head': 'head',
    'LeftArm': 'left_shoulder',
    'RightArm': 'right_shoulder',
    'LeftForeArm': 'left_elbow',
    'RightForeArm': 'right_elbow',
    'LeftHand': 'left_wrist',
    'RightHand': 'right_wrist',
    'LeftHandIndex1': 'left_hand',
    'RightHandIndex1': 'right_hand',
}


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

def rotation_6d_to_axis_angle(pose_6d):
    """
    6D 회전 표현을 축-각도 표현으로 변환합니다.
    
    Args:
        pose_6d: 6D 포즈 배열 (T, 132)
    
    Returns:
        축-각도 회전 표현 배열 (T, 72)
    """
    T = pose_6d.shape[0]
    pose_aa = np.zeros((T, 24, 3), dtype=np.float32)  # 24개 관절 모두 (SMPL 호환성)
    
    for t in range(T):
        for joint_idx in range(22):  # 6D 표현은 22개 관절만 있음
            # 6D 회전 표현 추출 (각 관절당 6차원)
            rot_6d = pose_6d[t, joint_idx*6:(joint_idx+1)*6].reshape(3, 2)
            
            # 첫 두 열 가져오기
            col1 = rot_6d[:, 0]
            col2 = rot_6d[:, 1]
            
            # 정규화
            col1_normalized = col1 / (np.linalg.norm(col1) + 1e-8)
            col2_normalized = col2 / (np.linalg.norm(col2) + 1e-8)
            
            # Gram-Schmidt 과정으로 정규직교 벡터 생성
            col2_orthogonal = col2_normalized - np.dot(col1_normalized, col2_normalized) * col1_normalized
            col2_orthogonal = col2_orthogonal / (np.linalg.norm(col2_orthogonal) + 1e-8)
            
            # 세 번째 열은 외적으로 계산
            col3 = np.cross(col1_normalized, col2_orthogonal)
            
            # 회전 행렬 구성
            rotmat = np.column_stack([col1_normalized, col2_orthogonal, col3])
            
            # 회전 행렬을 축-각도로 변환
            rot_aa = R.from_matrix(rotmat).as_rotvec()
            pose_aa[t, joint_idx] = rot_aa
    
    # 추가 관절은 기본값 유지 (현재 0으로 설정)
    return pose_aa.reshape(T, 72)  # (T, 72)

import numpy as np
from scipy.interpolate import interp1d

def normalize_motion_duration(pose, trans, original_duration_sec, target_duration_sec=2.0, target_len=60):
    """
    pose: (T, D)
    trans: (T, 3)
    original_duration_sec: 애니메이션 실제 길이 (초)
    target_duration_sec: 정규화할 기준 길이 (초)
    target_len: 정규화할 프레임 수 (기본 60프레임 @30fps)
    """

    T = pose.shape[0]
    original_times = np.linspace(0, original_duration_sec, T)
    target_times = np.linspace(0, target_duration_sec, target_len)

    if original_duration_sec > target_duration_sec:
        # 2초보다 긴 경우 → 시간 축 기준으로 리샘플링 (축소)
        interp_pose = interp1d(original_times, pose, axis=0, kind='linear')
        interp_trans = interp1d(original_times, trans, axis=0, kind='linear')
        pose_final = interp_pose(target_times)
        trans_final = interp_trans(target_times)
        mask = np.ones(target_len, dtype=np.uint8)  # 전부 유효
    else:
        # 2초보다 짧은 경우 → 시간 비율로 자르고 패딩
        use_len = int(target_len * (original_duration_sec / target_duration_sec))
        use_len = min(use_len, T)
        pose_used = pose[:use_len]
        trans_used = trans[:use_len]
        pad_len = target_len - use_len
        pad_pose = np.zeros((pad_len, pose.shape[1]), dtype=np.float32)
        pad_trans = np.zeros((pad_len, trans.shape[1]), dtype=np.float32)
        pose_final = np.concatenate([pose_used, pad_pose], axis=0)
        trans_final = np.concatenate([trans_used, pad_trans], axis=0)
        mask = np.array([1]*use_len + [0]*pad_len, dtype=np.uint8)

    return pose_final, trans_final, mask


def find_matching_bones(model_bone_names, mapping_dict):
    """
    모델의 본 이름과 매핑 딕셔너리를 기반으로 매칭되는 본을 찾습니다.
    
    Args:
        model_bone_names: 모델의 본 이름 리스트
        mapping_dict: 매핑 딕셔너리
    
    Returns:
        매칭된 본 딕셔너리 {모델_본_이름: 매핑_대상_본_이름}
    """
    matched = {}
    
    # 정확히 일치하는 본 찾기
    for bone_name in model_bone_names:
        for src_name, target_name in mapping_dict.items():
            if bone_name == src_name:
                matched[bone_name] = src_name
                break
    
    # 정확히 일치하는 본이 없으면 부분 일치 시도
    if not matched:
        for bone_name in model_bone_names:
            for src_name, target_name in mapping_dict.items():
                # 대소문자 무시하고 이름에 포함되어 있는지 확인
                if src_name.lower() in bone_name.lower() or bone_name.lower() in src_name.lower():
                    matched[bone_name] = src_name
                    break
    
    return matched

def get_smpl_joint_index(joint_name):
    """SMPL 관절 이름에 해당하는 인덱스를 반환합니다."""
    try:
        return SMPL_JOINT_NAMES.index(joint_name)
    except ValueError:
        return None

def extract_smpl_pose_from_mixamo(glb_path):
    """
    Mixamo GLB 파일에서 SMPL 포즈 데이터를 추출합니다.
    pygltflib를 사용하여 애니메이션 데이터를 정확히 추출합니다.
    
    Args:
        glb_path: GLB 파일 경로
    
    Returns:
        SMPL 포즈 배열 (T, 72), 6D 회전 표현 배열 (T, 132), 메타데이터 딕셔너리
    """
    try:
        # GLB 파일 로드
        gltf = pygltflib.GLTF2().load(glb_path)
        print(f"GLB 모델 로드 성공: {glb_path}")
        
        # 스켈레톤 분석
        if not gltf.nodes:
            print("GLB 모델에 노드가 없습니다.")
            return None, None, None
            
        # 모델의 본 이름 추출
        bone_names = []
        bone_indices = {}
        for i, node in enumerate(gltf.nodes):
            if hasattr(node, 'name') and node.name:
                bone_names.append(node.name)
                bone_indices[node.name] = i
        
        print(f"모델에서 {len(bone_names)}개의 본 발견")
        
        # SMPL <-> 모델 본 매핑 찾기
        matches = find_matching_bones(bone_names, MIXAMO_TO_SMPL)
        print(f"매핑된 본: {len(matches)}개")
        
        if not matches:
            print("매핑된 본이 없습니다.")
            return None, None, None
        
        # 애니메이션 데이터 확인
        if not gltf.animations:
            print("애니메이션이 없습니다.")
            return None, None, None
        
        # 애니메이션 정보 추출
        animation = gltf.animations[0]  # 첫 번째 애니메이션 사용
        print(f"애니메이션 이름: {getattr(animation, 'name', 'Unnamed')}")
        print(f"애니메이션 채널 수: {len(animation.channels)}")
        print(f"애니메이션 샘플러 수: {len(animation.samplers)}")
        
        # 시간 추출 (타임라인)
        times_array = None
        max_time = 0.0
        
        for channel in animation.channels:
            sampler = animation.samplers[channel.sampler]
            time_accessor = gltf.accessors[sampler.input]
            time_buffer_view = gltf.bufferViews[time_accessor.bufferView]
            time_buffer_data = gltf.get_data_from_buffer_uri(gltf.buffers[time_buffer_view.buffer].uri)
            
            # 시간 데이터 읽기
            time_offset = time_buffer_view.byteOffset if hasattr(time_buffer_view, 'byteOffset') and time_buffer_view.byteOffset is not None else 0
            time_data = np.frombuffer(
                time_buffer_data, 
                dtype=np.float32, 
                count=time_accessor.count,
                offset=time_offset
            )
            
            if time_data.size > 0:
                max_time = max(max_time, np.max(time_data))
                # 가장 긴 시간 배열 저장
                if times_array is None or time_data.size > times_array.size:
                    times_array = time_data
        
        if times_array is None or times_array.size == 0:
            print("시간 데이터를 추출할 수 없습니다.")
            return None, None, None
        
        frame_count = times_array.size
        print(f"프레임 수: {frame_count}, 총 시간: {max_time:.2f}초")
        
        # FPS 계산
        fps = frame_count / max_time if max_time > 0 else 30
        print(f"계산된 FPS: {fps:.2f}")
        
        # 각 채널에서 회전 데이터 추출
        # 각 관절에 대한 회전 데이터 저장 딕셔너리
        joint_rotations = {joint_name: None for joint_name in SMPL_JOINT_NAMES}
        node_to_rotations = {}  # 노드 인덱스 -> 회전 값 매핑
        
        for channel in animation.channels:
            if channel.target.path != "rotation":
                continue  # 회전 데이터만 처리
                
            node_idx = channel.target.node
            if node_idx >= len(gltf.nodes) or not hasattr(gltf.nodes[node_idx], 'name'):
                continue
                
            node_name = gltf.nodes[node_idx].name
            if not node_name:
                continue
                
            # 샘플러 정보 가져오기
            sampler = animation.samplers[channel.sampler]
            output_accessor = gltf.accessors[sampler.output]
            output_buffer_view = gltf.bufferViews[output_accessor.bufferView]
            output_buffer_data = gltf.get_data_from_buffer_uri(gltf.buffers[output_buffer_view.buffer].uri)
            
            # 회전 데이터 읽기 (쿼터니언)
            output_offset = output_buffer_view.byteOffset if hasattr(output_buffer_view, 'byteOffset') and output_buffer_view.byteOffset is not None else 0
            rotation_data = np.frombuffer(
                output_buffer_data, 
                dtype=np.float32, 
                count=output_accessor.count * 4,  # 쿼터니언은 4 성분
                offset=output_offset
            ).reshape(-1, 4)  # (프레임 수, 4) 형태
            
            # 노드 -> 회전 데이터 연결
            node_to_rotations[node_idx] = rotation_data
            
            # SMPL 본에 매핑
            for model_bone, smpl_bone in matches.items():
                smpl_joint_name = MIXAMO_TO_SMPL.get(smpl_bone)
                bone_idx = bone_indices.get(model_bone)
                
                if smpl_joint_name and bone_idx == node_idx:
                    # 쿼터니언 -> 회전 벡터 변환
                    rotvecs = R.from_quat(rotation_data).as_rotvec()
                    joint_rotations[smpl_joint_name] = rotvecs
                    print(f"관절 {model_bone} -> {smpl_joint_name} 회전 데이터 추출 성공: {len(rotvecs)} 프레임")
        
        # 24개 관절 각각에 대한 회전 데이터 확보
        # 누락된 관절은 기본값(영벡터)으로 채움
        pose = np.zeros((frame_count, 24, 3), dtype=np.float32)
        
        for i, joint_name in enumerate(SMPL_JOINT_NAMES):
            if joint_rotations[joint_name] is not None:
                rot_data = joint_rotations[joint_name]
                # 프레임 수 맞추기
                if len(rot_data) >= frame_count:
                    pose[:, i, :] = rot_data[:frame_count]
                else:
                    pose[:frame_count, i, :] = np.pad(
                        rot_data, 
                        ((0, frame_count - len(rot_data)), (0, 0)),
                        mode='edge'
                    )
        
        # (프레임 수, 24, 3) -> (프레임 수, 72) 형태로 변환
        pose_flat = pose.reshape(frame_count, -1)
        
        # 6D 회전 표현으로 변환
        pose_6d = axis_angle_to_rotation_6d(pose)
        
        # 메타데이터 반환
        metadata = {
            'fps': fps,
            'frame_count': frame_count,
            'duration': max_time,
            'original_file': os.path.basename(glb_path)
        }
        
        return pose_flat, pose_6d, metadata
        
    except Exception as e:
        print(f"GLB 파일 처리 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def save_smpl_npz(pose, out_path, fps=30, trans=None, betas=None, pose_6d=None):
    """
    SMPL 포즈 데이터를 NPZ 파일로 저장합니다.
    
    Args:
        pose: 포즈 데이터 (T, 72) 형태
        out_path: 출력 파일 경로
        fps: 프레임 레이트
        trans: 위치 데이터 (없으면 기본값 사용)
        betas: 형상 파라미터 (없으면 기본값 사용)
        pose_6d: 6D 회전 표현 데이터 (T, 132) 형태
    """
    T = pose.shape[0]
    
    # 기본값 설정
    if trans is None:
        trans = np.zeros((T, 3), dtype=np.float32)  # Mixamo는 위치 정보 없음
    
    if betas is None:
        betas = np.zeros((10,), dtype=np.float32)
    
    # 저장할 데이터 구성
    save_data = {
        'poses': pose,
        'trans': trans,
        'betas': betas,
        'fps': fps,
        'frame_count': T
    }
    
    # 6D 회전 표현 추가 (있는 경우)
    if pose_6d is not None:
        save_data['poses_6d'] = pose_6d
    
    # NPZ 저장
    np.savez(out_path, **save_data)
    print(f"SMPL 데이터 저장 완료: {out_path}")

def save_smpl_npy(pose, out_path, fps=30, trans=None, betas=None, pose_6d=None):
    """
    SMPL 포즈 데이터를 NPY 파일로 저장합니다.
    
    Args:
        pose: 포즈 데이터 (T, 72) 형태
        out_path: 출력 파일 경로
        fps: 프레임 레이트
        trans: 위치 데이터 (없으면 기본값 사용)
        betas: 형상 파라미터 (없으면 기본값 사용)
        pose_6d: 6D 회전 표현 데이터 (T, 132) 형태
    """
    data = {
        'poses': pose,
        'fps': fps
    }
    
    if trans is not None:
        data['trans'] = trans
    
    if betas is not None:
        data['betas'] = betas
    
    if pose_6d is not None:
        data['poses_6d'] = pose_6d
    
    np.save(out_path, data)
    print(f"SMPL 데이터 저장 완료: {out_path}")

def convert_glb_to_motionclip(glb_path, output_dir=None):
    """
    GLB 파일을 MotionClip 형식으로 변환합니다.
    
    Args:
        glb_path: GLB 파일 경로
        output_dir: 출력 디렉토리 (None이면 GLB 파일과 같은 디렉토리 사용)
    
    Returns:
        성공 여부(bool)와 결과 파일 경로들(dict)
    """
    try:
        # GLB에서 SMPL 포즈 데이터 추출
        pose, pose_6d, metadata = extract_smpl_pose_from_mixamo(glb_path)
        
        if pose is None:
            print(f"'{glb_path}' 파일의 포즈 데이터 추출 실패")
            return False, {}
        
        # 파일 경로 및 이름 설정
        base_dir = os.path.dirname(glb_path)
        base_name = os.path.splitext(os.path.basename(glb_path))[0]
        
        # 출력 디렉토리 설정
        if output_dir is None:
            output_dir = base_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 파일 경로 생성
        npy_path = os.path.join(output_dir, f"{base_name}_smpl.npy")
        npz_path = os.path.join(output_dir, f"{base_name}_smpl.npz")
        motionclip_path = os.path.join(output_dir, f"{base_name}_motionclip.npz")
        
        # 메타데이터 및 프레임 정보
        frame_count = pose.shape[0]
        fps = metadata.get('fps', 30) if metadata else 30
        
        # 중간 결과물 저장 (NPY 단일 배열)
        # save_smpl_npy(pose, npy_path, fps=fps, pose_6d=pose_6d)
        
        # NPZ 형식으로도 저장 (여러 배열)
        # save_smpl_npz(pose, npz_path, fps=fps, pose_6d=pose_6d)
        
        # 기본 위치 정보 및 체형 파라미터 생성
        trans = np.zeros((frame_count, 3), dtype=np.float32)
        betas = np.zeros(10, dtype=np.float32)
        
        # 텍스트 설명 파일 찾기 (파일명과 같은 이름의 txt 파일)
        txt_filename = f"{base_name}.txt"
        txt_path = os.path.join(base_dir, txt_filename)
        
        # 설명 텍스트 설정
        description = base_name  # 기본값
        if os.path.exists(txt_path):
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    description = f.read().strip()
                    if not description:  # 빈 파일인 경우
                        description = base_name
            except Exception as e:
                print(f"텍스트 파일 읽기 오류: {e}")
        
        # 설명 텍스트 정리 (공백, 대시, 밑줄 등을 단일 공백으로 변경)
        description = ' '.join([word.strip() for word in re.split(r'[,\s\-_]', description) if word.strip()])
        
        # 30fps, 2초, 60 프레임으로 데이터 보정. 
        # pose_6d,trans, padding_mask = normalize_motion_duration(pose_6d, trans, 
        #     metadata['duration'], target_duration_sec=2.0, target_len=60)
        # 학습 시 padding_mastk를 사용하여 패딩된 프레임을 무시하도록 설정

        # MotionClip 형식으로 저장
        np.savez(
            motionclip_path,
            pose=pose_6d,       # 6D 회전 데이터를 'pose'로 저장 (MotionClip 형식)
            trans=trans,        # 위치 정보
            betas=betas,        # 체형 파라미터
            text=np.array([description], dtype=object)  # 텍스트 설명
        )
        
        print(f"변환 완료: {os.path.basename(glb_path)}")
        print(f"  - 프레임: {frame_count}, FPS: {fps:.2f}")
        print(f"  - 설명: '{description}'")
        
        return True, {
            'npy': npy_path,
            'npz': npz_path,
            'motionclip': motionclip_path
        }
    
    except Exception as e:
        print(f"'{os.path.basename(glb_path)}' 변환 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False, {}

# 사용 예시
if __name__ == "__main__":
    import sys
    import time
    
    # 명령줄 인자 처리
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        # 기본 경로
        input_path = "/Users/jihyunlee/projects/ml_google_2nd_project/samples/glb/"
    
    # 출력 디렉토리 설정 (옵션)
    output_dir = None
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    # 입력이 디렉토리인지 파일인지 확인
    if os.path.isdir(input_path):
        # 디렉토리 내의 모든 GLB 파일 처리
        glb_files = glob.glob(os.path.join(input_path, "*.glb"))
        print(f"'{input_path}' 디렉토리에서 {len(glb_files)}개의 GLB 파일을 찾았습니다.")
        
        success_count = 0
        start_time = time.time()
        
        for i, glb_file in enumerate(glb_files):
            print(f"\n[{i+1}/{len(glb_files)}] 처리 중: {os.path.basename(glb_file)}")
            success, _ = convert_glb_to_motionclip(glb_file, output_dir)
            if success:
                success_count += 1
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n처리 완료: 총 {len(glb_files)}개 중 {success_count}개 변환 성공 ({duration:.1f}초 소요)")
        if success_count < len(glb_files):
            print(f"  - {len(glb_files) - success_count}개 파일 변환 실패")
    
    elif os.path.isfile(input_path) and input_path.lower().endswith('.glb'):
        # 단일 GLB 파일 처리
        print(f"단일 GLB 파일 변환 시작: {input_path}")
        success, results = convert_glb_to_motionclip(input_path, output_dir)
        
        if success:
            print(f"\n변환 성공!")
            print(f"생성된 파일:")
            print(f"  - NPY: {results['npy']}")
            print(f"  - NPZ: {results['npz']}")
            print(f"  - MotionClip: {results['motionclip']}")
        else:
            print("\n변환 실패!")
    
    else:
        print(f"오류: '{input_path}'는 유효한 GLB 파일 또는 디렉토리가 아닙니다.")
        print("사용법: python convert_mixamo_glb_to_smpl.py [GLB파일 또는 디렉토리] [출력디렉토리(선택)]")
        sys.exit(1)
