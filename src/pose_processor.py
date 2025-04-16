"""
포즈 데이터 처리 및 변환 모듈
"""

import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import re

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

def axis_angle_to_rotation_6d(pose_aa):
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

def normalize_motion_duration(pose, trans, original_duration_sec, target_duration_sec=2.0, target_len=60):
    """
    모션 데이터의 기간을 정규화합니다.
    
    Args:
        pose: (T, D) 형태의 포즈 데이터
        trans: (T, 3) 형태의 위치 데이터
        original_duration_sec: 애니메이션 실제 길이 (초)
        target_duration_sec: 정규화할 기준 길이 (초)
        target_len: 정규화할 프레임 수 (기본 60프레임 @30fps)
    
    Returns:
        정규화된 포즈, 위치, 마스크
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

def save_motion_clip(pose_6d, out_path, description=None, trans=None, betas=None):
    """
    MotionClip 형식으로 포즈 데이터를 저장합니다.
    
    Args:
        pose_6d: 6D 회전 표현 포즈 데이터 (T, 132)
        out_path: 출력 파일 경로
        description: 모션 설명 텍스트
        trans: 위치 데이터 (T, 3)
        betas: 체형 파라미터 (10,)
    """
    T = pose_6d.shape[0]
    
    # 기본값 설정
    if trans is None:
        trans = np.zeros((T, 3), dtype=np.float32)
    
    if betas is None:
        betas = np.zeros((10,), dtype=np.float32)
    
    if description is None:
        # 파일명에서 설명 추출
        base_name = os.path.splitext(os.path.basename(out_path))[0]
        description = ' '.join([word.strip() for word in re.split(r'[,\s\-_]', base_name) if word.strip()])
    
    # MotionClip 형식으로 저장
    np.savez(
        out_path,
        pose=pose_6d,       # 6D 회전 데이터를 'pose'로 저장 (MotionClip 형식)
        trans=trans,        # 위치 정보
        betas=betas,        # 체형 파라미터
        text=np.array([description], dtype=object)  # 텍스트 설명
    )
    
    print(f"MotionClip 데이터 저장 완료: {out_path}")
    print(f"  - 프레임: {T}")
    print(f"  - 설명: '{description}'")

def mediapipe_to_smpl(mediapipe_pose):
    """
    MediaPipe 포즈 형식을 SMPL 포즈 형식으로 변환합니다.
    현재는 단순 스켈레톤 매핑을 사용하며, 향후 기계학습 기반 변환으로 대체될 예정입니다.
    
    Args:
        mediapipe_pose: MediaPipe 포즈 데이터 [T, 33, 3]
    
    Returns:
        SMPL 포맷 포즈 데이터 (축-각도 표현, [T, 72]) 및 6D 표현 ([T, 132])
    """
    T = mediapipe_pose.shape[0]
    
    # 현재는 단순 변환으로 구현
    # 추후 ML 모델로 보다 정확한 변환 가능
    
    # 간단한 스켈레톤 매핑 (인덱스 기반)
    # MediaPipe는 33개 랜드마크를 사용, SMPL은 24개 관절 사용
    # 대략적인 매핑을 정의
    mp_to_smpl_mapping = {
        0: 0,     # nose -> pelvis (중심점)
        11: 13,   # left_shoulder -> left_collar
        12: 14,   # right_shoulder -> right_collar
        13: 16,   # left_elbow -> left_shoulder
        14: 17,   # right_elbow -> right_shoulder
        15: 18,   # left_wrist -> left_elbow
        16: 19,   # right_wrist -> right_elbow
        23: 1,    # left_hip -> left_hip
        24: 2,    # right_hip -> right_hip
        25: 4,    # left_knee -> left_knee
        26: 5,    # right_knee -> right_knee
        27: 7,    # left_ankle -> left_ankle
        28: 8,    # right_ankle -> right_ankle
    }
    
    # SMPL 포즈 초기화 (기본 포즈: T프레임, 24관절, 3차원)
    smpl_pose = np.zeros((T, 24, 3), dtype=np.float32)
    
    # MediaPipe 포즈를 SMPL 포즈로 매핑
    for mp_idx, smpl_idx in mp_to_smpl_mapping.items():
        if mp_idx < mediapipe_pose.shape[1]:
            # 단순히 위치 복사 (실제로는 각도 계산 필요)
            # 여기서는 근사치로 표현
            smpl_pose[:, smpl_idx] = mediapipe_pose[:, mp_idx] * 0.1
    
    # 포즈 데이터 형태 변환
    smpl_pose_aa = smpl_pose.reshape(T, 72)  # (T, 72)
    
    # 6D 회전 표현으로 변환
    smpl_pose_6d = axis_angle_to_rotation_6d(smpl_pose)  # (T, 132)
    
    return smpl_pose_aa, smpl_pose_6d

def mediapipe_to_motion_clip(mediapipe_pose, output_path, description=None):
    """
    MediaPipe 포즈 데이터를 MotionClip 형식으로 변환하여 저장합니다.
    
    Args:
        mediapipe_pose: MediaPipe 포즈 데이터 [T, 33, 3]
        output_path: 출력 파일 경로
        description: 모션 설명 텍스트
    
    Returns:
        성공 여부 (bool)
    """
    try:
        # MediaPipe 포즈를 SMPL 포즈로 변환
        _, smpl_pose_6d = mediapipe_to_smpl(mediapipe_pose)
        
        # 기본 위치 정보 및 체형 파라미터 생성
        T = mediapipe_pose.shape[0]
        trans = np.zeros((T, 3), dtype=np.float32)
        betas = np.zeros(10, dtype=np.float32)
        
        # 설명 텍스트 설정
        if description is None:
            # 파일명에서 설명 추출
            base_name = os.path.splitext(os.path.basename(output_path))[0]
            description = ' '.join([word.strip() for word in re.split(r'[,\s\-_]', base_name) if word.strip()])
        
        # MotionClip 형식으로 저장
        np.savez(
            output_path,
            pose=smpl_pose_6d,  # 6D 회전 데이터
            trans=trans,        # 위치 정보
            betas=betas,        # 체형 파라미터
            text=np.array([description], dtype=object)  # 텍스트 설명
        )
        
        print(f"MediaPipe 포즈에서 MotionClip 변환 완료: {output_path}")
        print(f"  - 프레임: {T}")
        print(f"  - 설명: '{description}'")
        
        return True
    
    except Exception as e:
        print(f"MediaPipe → MotionClip 변환 오류: {e}")
        import traceback
        traceback.print_exc()
        return False