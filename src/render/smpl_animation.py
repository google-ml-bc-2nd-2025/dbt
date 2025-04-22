"""
SMPL(Skinned Multi-Person Linear model) 형식의 애니메이션을 GLB 모델에 적용하는 모듈입니다.
NPY 파일로 저장된 SMPL 포즈 및 형상 파라미터를 로드하고 GLB 모델에 적용합니다.
"""

import os
import numpy as np
import uuid
import shutil
import json
import tempfile
import subprocess
from pathlib import Path
import trimesh
import pygltflib
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import io
from io import BytesIO
import base64

from model.bone_mappings import SMPL_JOINT_NAMES, find_matching_bones, get_smpl_joint_index
from model.smpl import mixamo_to_smpl_map as MIXAMO_TO_SMPL
from util.viewer_template import create_viewer_html
from converter.convert_mdm_to_glb import create_improved_glb_animation

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

def glb_to_6d_rotation(glb_file_path, output_path=None):
    """
    GLB 파일에서 애니메이션을 추출하고 6D 회전 표현으로 변환하여 NPZ 파일로 저장합니다.
    
    Args:
        glb_file_path: GLB 파일 경로
        output_path: 출력 NPZ 파일 경로 (None인 경우 입력 파일과 동일한 경로에 저장)
    
    Returns:
        저장된 NPZ 파일 경로
    """
    import pygltflib
    from pathlib import Path
    
    # 출력 경로가 지정되지 않은 경우 입력 파일과 같은 경로에 저장
    if (output_path is None):
        output_path = Path(glb_file_path).with_suffix('.npz')

    # GLB 파일 로드
    gltf = pygltflib.GLTF2().load(glb_file_path)
    
    # 애니메이션 데이터 추출
    animations = []
    poses_aa = []  # axis-angle 형태로 포즈 저장
    
    for anim in gltf.animations:
        # 각 애니메이션 채널에서 데이터 추출
        for channel in anim.channels:
            target = channel.target
            if (target.path == "rotation"):  # 회전 데이터만 처리
                sampler = anim.samplers[channel.sampler]
                
                # 키프레임 시간 데이터
                time_accessor = gltf.accessors[sampler.input]
                time_buffer_view = gltf.bufferViews[time_accessor.bufferView]
                time_buffer = gltf.buffers[time_buffer_view.buffer]
                time_data = np.frombuffer(
                    time_buffer.data[time_buffer_view.byteOffset:time_buffer_view.byteOffset + time_buffer_view.byteLength],
                    dtype=np.float32
                )
                
                # 회전 데이터 (쿼터니언)
                rotation_accessor = gltf.accessors[sampler.output]
                rotation_buffer_view = gltf.bufferViews[rotation_accessor.bufferView]
                rotation_buffer = gltf.buffers[rotation_buffer_view.buffer]
                rotation_data = np.frombuffer(
                    rotation_buffer.data[rotation_buffer_view.byteOffset:rotation_buffer_view.byteOffset + rotation_buffer_view.byteLength],
                    dtype=np.float32
                ).reshape(-1, 4)  # 쿼터니언 [x, y, z, w]
                
                # 쿼터니언을 axis-angle로 변환
                rotations = R.from_quat(rotation_data)
                axis_angles = rotations.as_rotvec()  # (frames, 3)
                
                # joint_id와 프레임별 회전 정보 저장
                joint_id = target.node
                poses_aa.append({
                    'joint_id': joint_id,
                    'times': time_data,
                    'rotations': axis_angles
                })

    

    # 모든 관절의 회전 데이터를 프레임별로 통합
    if poses_aa:

        # 프레임 수 결정
        max_frames = max(len(pose['times']) for pose in poses_aa)
        
        # 모든 관절의 회전을 저장할 배열 (frames, joints, 3)
        all_poses_aa = np.zeros((max_frames, 24, 3), dtype=np.float32)
        
        # 각 관절의 회전 데이터 할당
        for pose in poses_aa:
            joint_id = pose['joint_id']
            if (joint_id < 24):  # SMPL 모델은 일반적으로 24개의 관절을 가짐
                all_poses_aa[:len(pose['times']), joint_id] = pose['rotations']
        
        # axis-angle에서 6D 회전 표현으로 변환
        all_poses_6d = axis_angle_to_rotation_6d(all_poses_aa)
        
        # NPZ 파일로 저장
        np.savez(
            output_path,
            poses_6d=all_poses_6d,  # 6D 회전 표현 (T, 132)
            poses_aa=all_poses_aa.reshape(max_frames, -1),  # 원본 axis-angle (T, 72) (호환성)
            frame_count=max_frames
        )
        
        return output_path
    
    return None

class SMPLAnimationLoader:
    """SMPL 애니메이션 데이터를 로드하고 GLB 모델에 적용하는 클래스"""
    
    def __init__(self, smpl_params_file=None):
        """
        SMPL 애니메이션 로더 초기화
        
        Args:
            smpl_params_file: SMPL 애니메이션 파라미터가 저장된 JSON 또는 NPZ 파일 경로
        """
        self.fps = 30  # 기본 프레임 레이트
        self.animation_data = None
        self.use_6d_rotation = False  # 6D 회전 표현 사용 여부
        self.poses_6d = None  # 6D 회전 표현 데이터
        
        if smpl_params_file:
            self.load_animation(smpl_params_file)

    def load_from_mdm_data(self, mdm_motion_data):
        """
        MDM에서 생성된 모션 데이터를 메모리에서 직접 로드
        
        Args:
            mdm_motion_data: MDM에서 생성된 모션 데이터 (frame_count, 263) 형태 또는
                            다른 호환 가능한 모션 데이터 형태
        
        Returns:
            성공 여부 (bool)
        """
        try:
            # MDM 데이터 형식 확인 및 처리
            if isinstance(mdm_motion_data, np.ndarray):
                if mdm_motion_data.shape[1] == 263:  # HumanML3D 형식
                    # 첫 3개 값은 루트 위치, 나머지는 관절 회전 (축-각도)
                    joint_rotations = mdm_motion_data[:, 3:]
                    # 필요시 형식 변환 (MDM 출력 형식에 따라 조정 필요)
                    poses = joint_rotations.reshape(joint_rotations.shape[0], -1)
                else:
                    # 다른 형태의 모션 데이터는 있는 그대로 사용
                    poses = mdm_motion_data
                    
                # 프레임 수 및 기본 FPS 설정
                frame_count = len(poses)
                fps = 30  # MDM 기본 FPS 값
                
                # animation_data 딕셔너리 구성
                self.animation_data = {
                    'poses': poses.tolist() if isinstance(poses, np.ndarray) else poses,
                    'shape': [0.0] * 10,
                    'trans': [[0.0, 0.0, 0.0] for _ in range(frame_count)],
                    'fps': fps
                }
                
                self.fps = fps
                print(f"MDM 모션 데이터 로드 성공: {frame_count} 프레임, {fps} FPS")
                return True
            else:
                print(f"지원되지 않는 MDM 데이터 형식: {type(mdm_motion_data)}")
                return False
                
        except Exception as e:
            print(f"MDM 데이터 로드 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_animation(self, filepath):
        """
        SMPL 애니메이션 파일 로드
        
        Args:
            filepath: JSON, NPY 또는 NPZ 파일 경로
        
        Returns:
            성공 여부 (bool)
        """
        try:
            ext = os.path.splitext(filepath)[1].lower()
            
            if (ext == '.json'):
                with open(filepath, 'r') as f:
                    self.animation_data = json.load(f)
                    
                    # FPS 정보 확인
                    if ('fps' in self.animation_data):
                        self.fps = self.animation_data['fps']
                    
                    # axis-angle 형식으로 로드
                    self.use_6d_rotation = False
                    print(f"JSON 애니메이션 로드 성공: {len(self.animation_data['poses'])} 프레임")
                    return True
                    
            elif (ext == '.npy'):
                try:
                    poses = np.load(filepath, allow_pickle=True) # 예) (18, 3, 3, 120) - 120 : 프레임수
                    
                    # motion 키가 있는지 확인하고 있으면 해당 데이터 사용
                    if isinstance(poses, np.ndarray) and poses.dtype == np.dtype('O') and isinstance(poses.item(), dict) and 'motion' in poses.item():
                        print("motion 키 발견: dict 내부에서 추출")
                        poses = poses.item()['motion']
                    # 기존 방식 (딕셔너리가 직접 반환되는 경우)
                    elif isinstance(poses, dict) and 'motion' in poses:
                        print("motion 키 발견: 직접 추출")
                        poses = poses['motion']

                    print(f"NPY 파일 불러옴: {type(poses)} , {poses.shape}, {poses.ndim}")  # 디버그 정보 추가

                    # 배열 변환 및 길이 확인
                    if not isinstance(poses, np.ndarray):
                        poses = np.array(poses)
                        print("포즈 데이터를 numpy 배열로 변환했습니다.")
                    
                    if len(poses.shape) == 0:
                        # 크기가 없는 객체
                        print("오류: 포즈 데이터가 올바른 형태가 아닙니다.")
                        return False
                    
                    # 프레임 수 기반 FPS 추정 (17초 영상 기준)
                    frame_count = len(poses)
                    estimated_fps = max(24, min(60, frame_count / 17))  # 17초 영상 가정, 범위는 24-60
                    
                    # NPY 파일은 포즈 데이터만 포함하므로 나머지 필드는 기본값으로 설정
                    self.animation_data = {
                        'poses': poses.tolist() if isinstance(poses, np.ndarray) else poses,
                        'shape': [0.0] * 10,
                        'trans': [[0.0, 0.0, 0.0] for _ in range(frame_count)],
                        'fps': estimated_fps  # 추정된 FPS 사용
                    }
                    
                    self.fps = estimated_fps  # 클래스 변수 fps도 업데이트
                    print(f"NPY 애니메이션 로드 성공: {frame_count} 프레임, 추정 FPS: {estimated_fps:.1f}")
                    return True
                except Exception as e:
                    print(f"NPY 파일 로드 중 오류 발생: {e}")
                    import traceback
                    traceback.print_exc()
                    return False
            
            elif (ext == '.npz'):
                npz_data = np.load(filepath, allow_pickle=True)
                
                if ('poses_6d' in npz_data):
                    # 6D 회전 표현이 있는 경우
                    self.poses_6d = npz_data['poses_6d']
                    self.use_6d_rotation = True
                    frame_count = npz_data.get('frame_count', len(self.poses_6d))
                    estimated_fps = max(24, min(60, frame_count / 17))  # 17초 영상 가정, 범위는 24-60
                    
                    # axis-angle 형식이 있으면 함께 로드 (역호환성)
                    if ('poses_aa' in npz_data):
                        poses_aa = npz_data['poses_aa']
                        self.animation_data = {
                            'poses': poses_aa.tolist() if isinstance(poses_aa, np.ndarray) else poses_aa,
                            'shape': [0.0] * 10,
                            'trans': [[0.0, 0.0, 0.0] for _ in range(len(poses_aa))],
                            'fps': estimated_fps
                        }
                    else:
                        # 6D 표현을 axis-angle로 변환하여 호환성 유지
                        poses_aa = rotation_6d_to_axis_angle(self.poses_6d)
                        self.animation_data = {
                            'poses': poses_aa.tolist(),
                            'shape': [0.0] * 10,
                            'trans': [[0.0, 0.0, 0.0] for _ in range(len(poses_aa))],
                            'fps': estimated_fps
                        }
                    
                    print(f"NPZ 애니메이션 로드 성공 (6D 회전 표현): {frame_count} 프레임")
                    return True
                
                elif ('poses' in npz_data or 'poses_aa' in npz_data):
                    # 기존 형식의 NPZ 파일 (축-각도 표현)
                    poses = npz_data.get('poses', npz_data.get('poses_aa', None))
                    
                    if (poses is not None):
                        self.animation_data = {
                            'poses': poses.tolist() if isinstance(poses, np.ndarray) else poses,
                            'shape': npz_data.get('shape', [0.0] * 10),
                            'trans': npz_data.get('trans', [[0.0, 0.0, 0.0] for _ in range(len(poses))]),
                            'fps': npz_data.get('fps', self.fps)
                        }
                        
                        # 로드된 포즈를 6D 표현으로 변환
                        poses_array = np.array(poses)
                        self.poses_6d = axis_angle_to_rotation_6d(poses_array)
                        self.use_6d_rotation = True
                        
                        print(f"NPZ 애니메이션 로드 성공 (축-각도에서 변환): {len(poses)} 프레임")
                        return True
                
                else:
                    print(f"NPZ 파일에 필요한 포즈 데이터가 없습니다.")
                    return False
                
            else:
                print(f"지원되지 않는 파일 형식: {ext}")
                return False
                
        except Exception as e:
            print(f"애니메이션 로드 오류: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_joint_rotations(self):
        """
        SMPL 관절 회전 데이터 반환
        
        Returns:
            관절별 회전 데이터 딕셔너리 {joint_name: [frame1_rotation, frame2_rotation, ...]}
        """
        if (not self.animation_data or 'poses' not in self.animation_data):
            print("로드된 애니메이션 데이터가 없습니다.")
            return {}
        
        joint_rotations = {joint_name: [] for joint_name in SMPL_JOINT_NAMES}
        
        # 6D 회전 표현을 사용하는 경우
        if (self.use_6d_rotation and self.poses_6d is not None):
            # 6D 회전 표현을 축-각도로 변환
            poses_aa = rotation_6d_to_axis_angle(self.poses_6d)
            
            # 각 프레임마다 관절 회전 데이터 추출
            for frame_idx in range(poses_aa.shape[0]):
                frame_pose = poses_aa[frame_idx]  # (72,) 형태
                
                # 프레임의 모든 관절에 대해 처리
                for joint_idx, joint_name in enumerate(SMPL_JOINT_NAMES):
                    if (joint_idx < 22):  # 6D 회전은 22개 관절까지만 사용
                        # 관절별 3차원 회전 벡터 추출
                        rot_start = joint_idx * 3
                        rot_end = rot_start + 3
                        
                        if (rot_end <= len(frame_pose)):
                            rotation = frame_pose[rot_start:rot_end]
                            joint_rotations[joint_name].append(rotation.tolist())
        else:
            # 기존 코드 유지 (axis-angle 형식 처리)
            for frame_pose in self.animation_data['poses']:
                # 프레임의 모든 관절에 대해 처리
                for joint_idx, joint_name in enumerate(SMPL_JOINT_NAMES):
                    # 관절별 3차원 회전 벡터 추출
                    rot_start = joint_idx * 3
                    rot_end = rot_start + 3
                    
                    if (rot_end <= len(frame_pose)):
                        rotation = frame_pose[rot_start:rot_end]
                        joint_rotations[joint_name].append(rotation)
        
        return joint_rotations
    
    # 회전 행렬 수정을 위한 더 강력한 안정화 함수를 추가
    def stabilize_rotation_matrix(self, rot_mat):
        try:
            # 행렬식이 너무 작으면 바로 단위 행렬 반환
            if abs(np.linalg.det(rot_mat)) < 1e-6:
                return np.eye(3)
                
            # SVD 분해
            U, s, Vt = np.linalg.svd(rot_mat, full_matrices=False)
            
            # 특이값 정규화 및 확인
            if np.any(s < 1e-6):
                return np.eye(3)
            
            # 특이값을 1로 설정하여 정규직교성 보장
            s = np.ones_like(s)
            
            # 회전 행렬 재구성
            R = U @ np.diag(s) @ Vt
            
            # 행렬식 확인
            det = np.linalg.det(R)
            
            # 행렬식이 음수면 마지막 열 뒤집기
            if det < 0:
                U[:, -1] = -U[:, -1]
                R = U @ np.diag(s) @ Vt
                
            return R
        except Exception:
            # 오류 발생 시 단위 행렬 반환
            return np.eye(3)
        
    def create_keyframes(self, target_fps=30):
        """
        애니메이션 키프레임 데이터 생성
        
        Args:
            target_fps: 대상 프레임 레이트
        
        Returns:
            키프레임 딕셔너리 {joint_name: {'rotation': [...], 'times': [...]}}
        """
        if len(self.animation_data) <= 0:
            print(f"애니메이션 데이터가 없습니다. self.animation_data={self.animation_data}")
            return {}
        
        joint_rotations = self.get_joint_rotations()
        keyframes = {}
        
        # 프레임 수 확인
        frame_count = len(self.animation_data['poses'])
        if (frame_count == 0):
            return {}
        
        # 프레임 간격 계산 (초 단위)
        src_fps = self.animation_data.get('fps', self.fps)
        
        # 원본 영상 길이 계산 (초 단위)
        duration_sec = frame_count / src_fps
        print(f"원본 애니메이션 길이: {duration_sec}초 ({frame_count} 프레임, {src_fps} FPS)")
        
        # 시간값 생성 - 전체 프레임에 대해 정확한 타임스탬프 생성
        times = [i * (1.0 / src_fps) for i in range(frame_count)]
        
        # 목표 FPS로 보간할 시간값 생성
        if (target_fps != src_fps):
            # 전체 애니메이션 길이를 유지하도록 타임스탬프 생성
            target_frame_time = 1.0 / target_fps
            target_frame_count = int(duration_sec * target_fps) + 1  # 마지막 프레임 포함
            target_times = [i * target_frame_time for i in range(target_frame_count)]
            print(f"변환 후 애니메이션: {len(target_times)} 프레임, {target_fps} FPS, {duration_sec}초")
        else:
            target_times = times
            print(f"FPS 변환 없음: {len(times)} 프레임 유지")
        
        # 각 관절에 대한 키프레임 생성 (나머지 코드는 동일)
        for joint_name, rotations in joint_rotations.items():
            if (not rotations):
                continue
                
            try:
                # 회전 벡터를 쿼터니언으로 변환
                rotvecs = np.array(rotations)
                print(f'관절 {joint_name} 회전 벡터 형태: {rotvecs.shape}')
                # Extract dimensions and shape information
                if rotvecs.ndim == 4:  # Check if shape is like (18, 3, 3, 120)
                    rotvecs = rotvecs[0]  # 첫 번째 애니메이션만 선택하여 (3, 3, 120) 형태로 변경
                    dim1, dim2, frame_count = rotvecs.shape
                    print(f"Shape: ({dim1}, {dim2}, {frame_count})")
                    rotvecs = np.transpose(rotvecs,(2, 0,1))  # (120, 3, 3)
                    rotvecs = rotvecs.reshape(frame_count, dim1, dim2)  # (120*18, 3,3)  # 회전 행렬을 9D로 flatten
                    print(f"수정 후 : {type(rotvecs)} , {rotvecs.shape}, {rotvecs.ndim}")  # 디버그 정보 추가

                # 입력 형태 검사 및 수정
                if len(rotvecs.shape) == 3:  # (N, 3, 3) 형태인 경우 (회전 행렬)
                    quats = []
                    for rot_mat in rotvecs:
                        try:
                            # 회전 행렬 안정화: 개선된 함수 사용
                            valid_rot_mat = self.stabilize_rotation_matrix(rot_mat)
                            
                            # 안정화 후에도 행렬식 검사
                            if np.linalg.det(valid_rot_mat) <= 0:
                                print(f"  - 회전 행렬 안정화 실패, 기본값 사용: 행렬식 = {np.linalg.det(valid_rot_mat)}")
                                quat = np.array([0, 0, 0, 1])
                            else:
                                quat = R.from_matrix(valid_rot_mat).as_quat()
                        except Exception as e:
                            print(f"  - 회전 행렬 수정 실패, 기본값 사용: {e}")
                            quat = np.array([0, 0, 0, 1])
                            
                        quats.append(quat)
                    quats = np.array(quats)
                else:  # (N, 3) 형태인 경우 (회전 벡터)
                    # 매우 작은 값들 0으로 처리 (수치적 안정성)
                    rotvecs = np.where(np.abs(rotvecs) < 1e-10, 0, rotvecs)

                # rotvecs = rotation_6d_to_axis_angle(rotvecs)
                # 크기가 너무 큰 회전 벡터 조정 (pi를 넘어가는 경우)
                norms = np.linalg.norm(rotvecs, axis=1)
                scaled_rotvecs = rotvecs.copy()

                for i, norm in enumerate(norms):
                    # 배열인지 확인하고 적절히 처리
                    if np.isscalar(norm):
                        if norm > np.pi:
                            scaled_rotvecs[i] = rotvecs[i] * (np.pi / norm)
                    else:
                        # 배열인 경우 각 요소별 처리
                        mask = norm > np.pi
                        if np.any(mask):
                            scale_factor = np.where(mask, np.pi / norm, 1.0)
                            scaled_rotvecs[i] = rotvecs[i] * scale_factor
                    try:
                        if rotvecs.shape[-2:] == (3, 3):
                            quats = R.from_matrix(rotvecs).as_quat()
                        elif rotvecs.shape[-1] == 3:
                            quats = R.from_rotvec(rotvecs).as_quat()
                        else:
                            raise ValueError(f"지원하지 않는 회전 데이터 shape: {rotvecs.shape}")
                    except Exception as e:
                        # print(f"  - 회전 벡터 변환 실패, 기본값 사용: {e}")
                        # 기본 단위 쿼터니언으로 대체
                        quats = np.tile(np.array([0, 0, 0, 1]), (len(rotvecs), 1))                
                # 목표 FPS로 보간이 필요한 경우
                if (target_fps != src_fps):
                    # 각 쿼터니언 성분별 보간 함수 생성
                    interp_funcs = [
                        interp1d(times, quats[:, i], kind='linear', bounds_error=False, fill_value="extrapolate")
                        for i in range(4)
                    ]
                    
                    # 목표 시간에서의 쿼터니언 값 계산
                    interp_quats = np.column_stack([
                        interp_func(target_times) for interp_func in interp_funcs
                    ])
                    
                    # 보간된 쿼터니언 정규화
                    quat_lengths = np.sqrt(np.sum(interp_quats**2, axis=1))
                    normalized_quats = interp_quats / quat_lengths[:, np.newaxis]
                    
                    # 키프레임 데이터 저장
                    keyframes[joint_name] = {
                        'rotation': normalized_quats.tolist(),
                        'times': target_times
                    }
                else:
                    # 보간 없이 원본 데이터 사용
                    keyframes[joint_name] = {
                        'rotation': quats.tolist(),
                        'times': times
                    }
                    
            except Exception as e:
                print(f"관절 {joint_name}의 키프레임 생성 중 오류: {e}")
                # 오류가 발생한 관절은 건너뜀
                continue
        
        return keyframes
    
    def apply_to_glb(self, glb_path, output_path=None):
        """
        SMPL 애니메이션을 GLB 모델에 적용
        
        Args:
            glb_path: 대상 GLB 파일 경로
            output_path: 출력 GLB 파일 경로 (None이면 원본 파일명에 _animated 추가)
            
        Returns:
            성공 여부 (bool)
        """
        if (not self.animation_data):
            print("애니메이션 데이터가 없습니다.")
            return False
        
        try:
            # GLB 파일 로드
            gltf = pygltflib.GLTF2().load(glb_path)
            print(f"GLB 모델 로드 성공: {glb_path}")
            
            # 최종 GLB 파일 저장 후, 다시 로드하여 검증
            print(f"저장된 GLB 검증: {len(gltf.animations)}개 애니메이션")
            if gltf.animations:
                print(f"  - 애니메이션 0: {gltf.animations[0].name}, {len(gltf.animations[0].channels)}개 채널")
                print(f"  - 시간 액세서: {gltf.accessors[gltf.animations[0].samplers[0].input].count}개 키프레임")

            # 스켈레톤 분석
            if (not gltf.nodes):
                print("GLB 모델에 노드가 없습니다.")
                return False
                
            # 모델의 본 이름 추출
            bone_names = []
            bone_indices = {}
            for i, node in enumerate(gltf.nodes):
                if (hasattr(node, 'name') and node.name):
                    bone_names.append(node.name)
                    bone_indices[node.name] = i
            
            # apply_to_glb 함수 내에서 매핑 정보 상세 출력
            
            # SMPL <-> 모델 본 매핑 찾기 (개선된 매핑 전달)
            matches = find_matching_bones(bone_names, MIXAMO_TO_SMPL, nodes=gltf.nodes)
            print(f"모델에서 {len(bone_names)}개의 본 발견: {bone_names[:10]}...")
            print(f"매핑된 본: {len(matches)}개, 매핑 목록: {matches}")
            
            if (not matches):
                print("매핑된 본이 없습니다.")
                return False
            
            # 애니메이션 길이 계산 (초)
            frame_count = len(self.animation_data['poses'])
            src_fps = self.animation_data.get('fps', self.fps)
            duration = frame_count / src_fps
            print(f"애니메이션 길이: {duration}초 ({frame_count}프레임, {src_fps}FPS)")
            
            # 키프레임 데이터 생성 - 원본 길이를 보존하기 위해 target_fps를 동일하게 설정
            # 또는 target_fps를 30으로 고정하되 전체 길이를 보존하는 방식으로 수정
            keyframes = self.create_keyframes(target_fps=src_fps)  # 원본 FPS 유지

            # keyframes이 비어 있는지 확인
            if not keyframes:
                print("생성된 키프레임이 없습니다. 애니메이션을 적용할 수 없습니다.")
                return False
                                    
            # 애니메이션 채널 및 샘플러 생성
            all_times = []
            samplers = []
            channels = []
            
            # GLB에 이미 애니메이션이 있으면 제거
            gltf.animations = []
            
            # 액세서 인덱스 추적
            accessor_index = len(gltf.accessors)
            buffer_view_index = len(gltf.bufferViews)
            
            # 애니메이션 데이터를 담을 바이너리 데이터
            animation_buffer_data = bytearray()
            
            # 시간 데이터 생성 (모든 본이 공유)
            sample_joint = next(iter(keyframes.values()))
            times = sample_joint['times']
            # 시간 범위 확인 - 마지막 시간이 예상 길이와 일치하는지 확인
            expected_duration = frame_count / src_fps
            if len(times) > 0:
                # 최소/최대 시간 명시적 설정 (애니메이션 길이 보존)
                min_time = 0.0
                max_time = expected_duration                

                print(f"시간 범위: {min(times)} ~ {max(times)}")
                print(f"예상 길이: {expected_duration}")
                # 시간 데이터 생성 후 검증
                print(f"시간 데이터: 처음 5개 값={times[:5]}, 마지막 5개 값={times[-5:]}")
                print(f"키프레임 수: {len(times)}, 예상 개수: {frame_count}")
                
                if abs(times[-1] - expected_duration) > 0.01:
                    scale = expected_duration / times[-1] if times[-1] > 0 else 1.0
                    times = [t * scale for t in times]
                    print(f"시간 범위 조정: 스케일링 적용 (×{scale:.2f})")

            time_data = np.array(times, dtype=np.float32).tobytes()
            
            # 시간 데이터 버퍼뷰 및 액세서 추가
            time_buffer_view = pygltflib.BufferView(
                buffer=0,
                byteOffset=len(animation_buffer_data),
                byteLength=len(time_data),
                target=pygltflib.ARRAY_BUFFER
            )
            gltf.bufferViews.append(time_buffer_view)
            
            time_accessor = pygltflib.Accessor(
                bufferView=buffer_view_index,
                componentType=pygltflib.FLOAT,
                count=len(times),
                type=pygltflib.SCALAR,
                max=[float(expected_duration)],
                min=[0.0]
            )
            gltf.accessors.append(time_accessor)
            
            animation_buffer_data.extend(time_data)
            time_accessor_index = accessor_index
            
            accessor_index += 1
            buffer_view_index += 1
            
            # 매핑된 본에 애니메이션 적용
            for model_bone, smpl_bone in matches.items():
                if (model_bone not in bone_indices):
                    continue
                    
                node_index = bone_indices[model_bone]
                
                # SMPL 본 이름 찾기
                smpl_joint_name = MIXAMO_TO_SMPL.get(smpl_bone)
                if (not smpl_joint_name or smpl_joint_name not in keyframes):
                    continue
                
                # 회전 데이터 가져오기
                rotation_keyframes = keyframes[smpl_joint_name]['rotation']
                
                # 회전 데이터를 바이너리로 변환
                rotation_data = np.array(rotation_keyframes, dtype=np.float32).tobytes()
                
                # 회전 데이터 버퍼뷰 및 액세서 추가
                rot_buffer_view = pygltflib.BufferView(
                    buffer=0,
                    byteOffset=len(animation_buffer_data),
                    byteLength=len(rotation_data),
                    target=pygltflib.ARRAY_BUFFER
                )
                gltf.bufferViews.append(rot_buffer_view)
                
                quat_min = np.min(rotation_keyframes, axis=0).tolist()
                quat_max = np.max(rotation_keyframes, axis=0).tolist()
                
                rot_accessor = pygltflib.Accessor(
                    bufferView=buffer_view_index,
                    componentType=pygltflib.FLOAT,
                    count=len(rotation_keyframes),
                    type=pygltflib.VEC4,
                    max=quat_max,
                    min=quat_min
                )
                gltf.accessors.append(rot_accessor)
                
                animation_buffer_data.extend(rotation_data)
                
                # 샘플러 및 채널 생성
                sampler = pygltflib.AnimationSampler(
                    input=time_accessor_index,
                    output=accessor_index,
                    interpolation="STEP"  # "LINElet duration = clip.durationAR" 대신 "STEP" 사용
                )
                samplers.append(sampler)
                
                channel = pygltflib.AnimationChannel(
                    sampler=len(samplers)-1,
                    target=pygltflib.AnimationChannelTarget(
                        node=node_index,
                        path="rotation"
                    )
                )
                channels.append(channel)
                
                accessor_index += 1
                buffer_view_index += 1
            
            # 애니메이션 정보 출력
            print(f"애니메이션 정보: {len(channels)}개 채널, {len(samplers)}개 샘플러")
            print(f"애니메이션 시간 범위: {min(times)}초 ~ {max(times)}초")
            print(f"키프레임 수: {len(times)}")

            # rotation_keyframes 변수가 범위 밖에서 사용되는 문제 해결
            # 가장 마지막으로 처리된 회전 데이터를 저장할 변수 추가
            last_processed_rotation = None

            # 첫 번째와 마지막 회전 값 확인 (사용 전에 변수가 정의되어 있는지 확인)
            if 'rotation_keyframes' in locals() and rotation_keyframes and len(rotation_keyframes) > 1:
                print(f"첫 번째 프레임 회전값: {rotation_keyframes[0]}")
                print(f"마지막 프레임 회전값: {rotation_keyframes[-1]}")
                # 나중에 참조할 수 있도록 회전 데이터 저장
                last_processed_rotation = rotation_keyframes[-1]
            
            # 나머지 본들을 위한 보간 로직
            # 매핑되지 않은 본들에 대한 더 적극적인 보간 처리
            child_to_parent = {}
            parent_to_children = {}

            # 부모-자식 관계 분석
            for i, node in enumerate(gltf.nodes):
                if hasattr(node, 'children') and node.children:
                    parent_to_children[i] = node.children
                    for child in node.children:
                        child_to_parent[child] = i
            
            # 매핑되지 않은 본들을 위한 처리
            for i, node in enumerate(gltf.nodes):
                if not hasattr(node, 'name') or not node.name or node.name in matches:
                    continue
                
                # 계층 기반 참조 본 찾기 (상향 검색: 자식->부모)
                reference_node_idx = None
                reference_rotations = None
                parent_idx = i
                depth = 0
                max_depth = 3  # 3단계까지만 부모 방향 탐색
                
                while depth < max_depth:
                    # 부모 노드가 매핑된 본인지 확인
                    parent_idx = child_to_parent.get(parent_idx)
                    if parent_idx is None:
                        break
                    
                    if hasattr(gltf.nodes[parent_idx], 'name') and gltf.nodes[parent_idx].name in matches:
                        smpl_joint = matches[gltf.nodes[parent_idx].name]
                        if smpl_joint in keyframes:
                            reference_rotations = keyframes[smpl_joint]['rotation']
                            reference_node_idx = parent_idx
                            print(f"상향 참조: {node.name} -> {gltf.nodes[parent_idx].name} ({smpl_joint})")
                            break
                    
                    depth += 1
                
                # 하향 검색 (부모->자식)
                if reference_rotations is None:
                    children = parent_to_children.get(i, [])
                    for child_idx in children:
                        if hasattr(gltf.nodes[child_idx], 'name') and gltf.nodes[child_idx].name in matches:
                            smpl_joint = matches[gltf.nodes[child_idx].name]
                            if smpl_joint in keyframes:
                                reference_rotations = keyframes[smpl_joint]['rotation']
                                reference_node_idx = child_idx
                                print(f"하향 참조: {node.name} -> {gltf.nodes[child_idx].name} ({smpl_joint})")
                                break
                
                # 참조 본을 찾았으면 애니메이션 적용
                if reference_rotations:
                    # 참조 본과의 관계(부모/자식)에 따라 감쇠 계수 조절
                    if reference_node_idx == child_to_parent.get(i):
                        # 참조 본이 부모인 경우
                        damping = 0.85  # 부모의 움직임을 강하게 반영
                    elif i == child_to_parent.get(reference_node_idx):
                        # 참조 본이 자식인 경우  
                        damping = 0.7   # 자식의 움직임을 적당히 반영
                    else:
                        # 그 외 관계
                        damping = 0.6   # 약하게 반영
                
                # 부모 본의 회전 확인
                if (parent_idx is not None and hasattr(gltf.nodes[parent_idx], 'name')):
                    parent_name = gltf.nodes[parent_idx].name
                    if (parent_name in matches.values()):
                        smpl_joint = MIXAMO_TO_SMPL.get(parent_name)
                        if (smpl_joint in keyframes):
                            reference_rotations = keyframes[smpl_joint]['rotation']
                
                # 자식 본의 회전 확인 (부모 본에 애니메이션이 없는 경우)
                if (reference_rotations is None and children):
                    for child_idx in children:
                        if (hasattr(gltf.nodes[child_idx], 'name')):
                            child_name = gltf.nodes[child_idx].name
                            if (child_name in matches.values()):
                                smpl_joint = MIXAMO_TO_SMPL.get(child_name)
                                if (smpl_joint in keyframes):
                                    reference_rotations = keyframes[smpl_joint]['rotation']
                                    break
                
                # 참조할 회전 데이터가 있으면 약간의 감쇠를 적용하여 사용
                if (reference_rotations):
                    # 감쇠 계수 (부모/자식 본의 움직임을 감소시키기 위함)
                    damping = 0.7
                    
                    # 기본 회전 가져오기 (있는 경우)
                    default_rotation = node.rotation if (hasattr(node, 'rotation') and node.rotation) else [0, 0, 0, 1]
                    
                    # 참조 회전에서 일부만 적용
                    damped_rotations = []
                    for ref_rot in reference_rotations:
                        # 기본 회전과 참조 회전 사이 보간
                        damped_rot = [
                            default_rotation[i] * (1 - damping) + ref_rot[i] * damping
                            for i in range(4)
                        ]
                        damped_rotations.append(damped_rot)
                    
                    # 회전 데이터를 바이너리로 변환
                    rotation_data = np.array(damped_rotations, dtype=np.float32).tobytes()
                    
                    # 회전 데이터 버퍼뷰 및 액세서 추가
                    rot_buffer_view = pygltflib.BufferView(
                        buffer=0,
                        byteOffset=len(animation_buffer_data),
                        byteLength=len(rotation_data),
                        target=pygltflib.ARRAY_BUFFER
                    )
                    gltf.bufferViews.append(rot_buffer_view)
                    
                    quat_min = np.min(damped_rotations, axis=0).tolist()
                    quat_max = np.max(damped_rotations, axis=0).tolist()
                    
                    rot_accessor = pygltflib.Accessor(
                        bufferView=buffer_view_index,
                        componentType=pygltflib.FLOAT,
                        count=len(damped_rotations),
                        type=pygltflib.VEC4,
                        max=quat_max,
                        min=quat_min
                    )
                    gltf.accessors.append(rot_accessor)
                    
                    animation_buffer_data.extend(rotation_data)
                    
                    # 샘플러 및 채널 생성
                    sampler = pygltflib.AnimationSampler(
                        input=time_accessor_index,
                        output=accessor_index,
                        interpolation="STEP"  # "LINElet duration = clip.durationAR" 대신 "STEP" 사용
                    )
                    samplers.append(sampler)
                    
                    channel = pygltflib.AnimationChannel(
                        sampler=len(samplers)-1,
                        target=pygltflib.AnimationChannelTarget(
                            node=i,
                            path="rotation"
                        )
                    )
                    channels.append(channel)
                    
                    accessor_index += 1
                    buffer_view_index += 1
            
            # 애니메이션 객체 생성 전, 채널 정보 상세 출력
            for i, channel in enumerate(channels):
                target_node = channel.target.node
                node_name = gltf.nodes[target_node].name if hasattr(gltf.nodes[target_node], 'name') else f"노드 {target_node}"
                print(f"채널 {i}: 노드 '{node_name}', 경로 '{channel.target.path}'")            
                
            # 애니메이션 객체 생성
            animation = pygltflib.Animation(
                name="SMPLAnimation",
                channels=channels,
                samplers=samplers,
                # extension 필드 추가
                extensions={
                    "KHR_animation_duration": {
                        "duration": duration,
                        "fps": self.fps,
                        "auto_play": True
                    }
                },
                # 메타데이터 보강
                extras={
                    'fps': self.fps,
                    'duration': duration,
                    'frame_count': frame_count,
                    'auto_play': True,
                    'loop': True
                }
            )
            gltf.animations.append(animation)
            
            # 바이너리 버퍼 업데이트
            if (not gltf.buffers):
                gltf.buffers = [pygltflib.Buffer()]
            
            # 이슈 해결: binary_blob 처리 방식 완전히 수정
            # 기존 바이너리 데이터 가져오기 대신 새로 생성
            
            # 1. 메모리에 GLB 파일 저장
            temp_output = f"{glb_path}_temp.glb"
            gltf.save(temp_output)
            
            # 2. 임시 파일 다시 로드하여 애니메이션 데이터 추가
            with open(temp_output, 'rb') as f:
                glb_data = f.read()
                
            # 애니메이션 바이너리 데이터 추가 (GLB 형식에 맞게)
            # GLB 형식: magic(4) + version(4) + length(4) + json chunk + bin chunk
            # 바이너리 청크 찾기
            chunk_header_size = 12
            chunk_type_size = 4
            
            # 첫 번째 청크 크기 (JSON)
            json_chunk_length = int.from_bytes(glb_data[12:16], byteorder='little')
            json_chunk_end = 20 + json_chunk_length  # 20 = (magic+version+length) + (chunk length+type)
            
            # 두 번째 청크 (BIN)가 있는지 확인
            if (len(glb_data) > json_chunk_end + 8):
                bin_chunk_length = int.from_bytes(glb_data[json_chunk_end:json_chunk_end+4], byteorder='little')
                bin_chunk_type = glb_data[json_chunk_end+4:json_chunk_end+8]
                
                if (bin_chunk_type == b'BIN\0'):
                    # 기존 BIN 청크가 있는 경우
                    bin_chunk_data = glb_data[json_chunk_end+8:json_chunk_end+8+bin_chunk_length]
                    
                    # 새 바이너리 데이터 생성
                    new_bin_chunk = bin_chunk_data + animation_buffer_data
                    new_bin_chunk_length = len(new_bin_chunk)
                    
                    # 새 바이너리 청크로 GLB 재구성
                    new_glb_data = (
                        glb_data[:json_chunk_end] +  # 헤더와 JSON 청크
                        new_bin_chunk_length.to_bytes(4, byteorder='little') +  # 새 BIN 청크 크기
                        b'BIN\0' +  # BIN 청크 타입
                        new_bin_chunk  # 새 BIN 청크 데이터
                    )
                    
                    # 전체 파일 크기도 업데이트
                    new_length = len(new_glb_data)
                    new_glb_data = (
                        new_glb_data[:8] +  # magic + version
                        new_length.to_bytes(4, byteorder='little') +  # 새 파일 크기
                        new_glb_data[12:]  # 나머지 데이터
                    )
                else:
                    # 기존 BIN 청크가 없는 경우, 추가
                    new_bin_chunk_length = len(animation_buffer_data)
                    
                    # 새 바이너리 청크로 GLB 재구성
                    new_glb_data = (
                        glb_data[:json_chunk_end] +  # 헤더와 JSON 청크
                        new_bin_chunk_length.to_bytes(4, byteorder='little') +  # 새 BIN 청크 크기
                        b'BIN\0' +  # BIN 청크 타입
                        animation_buffer_data  # 새 BIN 청크 데이터
                    )
                    
                    # 전체 파일 크기도 업데이트
                    new_length = len(new_glb_data)
                    new_glb_data = (
                        new_glb_data[:8] +  # magic + version
                        new_length.to_bytes(4, byteorder='little') +  # 새 파일 크기
                        new_glb_data[12:]  # 나머지 데이터
                    )
            else:
                # 기존 BIN 청크가 없는 경우, 추가
                new_bin_chunk_length = len(animation_buffer_data)
                
                # 새 바이너리 청크로 GLB 재구성
                new_glb_data = (
                    glb_data[:json_chunk_end] +  # 헤더와 JSON 청크
                    new_bin_chunk_length.to_bytes(4, byteorder='little') +  # 새 BIN 청크 크기
                    b'BIN\0' +  # BIN 청크 타입
                    animation_buffer_data  # 새 BIN 청크 데이터
                )
                
                # 전체 파일 크기도 업데이트
                new_length = len(new_glb_data)
                new_glb_data = (
                    new_glb_data[:8] +  # magic + version
                    new_length.to_bytes(4, byteorder='little') +  # 새 파일 크기
                    new_glb_data[12:]  # 나머지 데이터
                )
            
            # 업데이트된 버퍼 길이 설정
            gltf.buffers[0].byteLength = new_bin_chunk_length
            
            # 출력 경로 설정
            if (not output_path):
                base_name = os.path.splitext(glb_path)[0]
                output_path = f"{base_name}_animated.glb"
            
            # 직접 바이너리를 파일에 쓰기
            with open(output_path, 'wb') as f:
                f.write(new_glb_data)
            print(f"애니메이션이 적용된 GLB 파일 저장 완료: {output_path}")
            
            # 임시 파일 제거
            if (os.path.exists(temp_output)):
                os.remove(temp_output)
            
            return output_path
            
        except Exception as e:
            print(f"GLB 파일에 애니메이션 적용 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return False

def apply_to_glb(skin_model, anim_data, viewer_path, models_dir, return_type='html'):
    """
    SMPL 포즈 기반 애니메이션 데이터(NPY/NPZ/JSON 등)를 GLB(3D 메시) 스킨 모델에 적용하여, 
    애니메이션이 포함된 GLB 파일을 생성하거나, 웹 뷰어에서 볼 수 있는 HTML(iframe) 코드를 반환하는 함수입니다.

    Args:
        skin_model: 스킨 모델 파일 객체 (GLB 형식)
        anim_data: SMPL 애니메이션 파일 객체 (NPY/NPZ/JSON 형식)
        viewer_path: 뷰어 HTML 파일의 경로
        models_dir: 모델 파일이 저장될 디렉토리
        return_type: 'html' 또는 'glb' (glb로 지정하면 파일 객체 반환)

    Returns:
        HTML 문자열(iframe) 또는 파일 객체(TemporaryFile)
    """

    # 필요한 모듈 로컬로 import
    import os
    import uuid
    import shutil
    from pathlib import Path
    import traceback
    from types import SimpleNamespace
    
    if skin_model is None:
        return "스킨 모델을 먼저 업로드해주세요."

    try:
        unique_id = str(uuid.uuid4())[:8]

        # 애니메이션이 없는 경우 스킨 파일만 처리
        if anim_data is None or not hasattr(anim_data, 'name') or not anim_data.name.lower().endswith(('.npy', '.npz', '.json')):
            # 기존 코드 유지 (HTML 반환)
            print("애니메이션 파일이 없거나 유효하지 않습니다. 스킨 모델만 표시합니다.")
            
            # 파일명 생성
            skin_ext = Path(skin_model.name).suffix.lower()
            result_filename = f"anim_{unique_id}{skin_ext}"
            result_path = os.path.join(models_dir, result_filename)
            
            # 스킨 모델 파일 복사
            try:
                shutil.copy2(skin_model.name, result_path)
                print(f"스킨 모델 복사 성공: {result_path}")
            except Exception as e:
                print(f"스킨 모델 복사 실패: {e}")
                # 파일 복사가 실패한 경우 대체 방법 시도
                with open(result_path, 'wb') as f_out:
                    with open(skin_model.name, 'rb') as f_in:
                        f_out.write(f_in.read())
                print(f"파일 읽기/쓰기로 복사 시도: {result_path}")
            
            # 파일 생성 확인
            if os.path.exists(result_path):
                print(f"파일 확인: {result_path}, 크기: {os.path.getsize(result_path)} 바이트")
            else:
                print(f"오류: 파일이 생성되지 않음: {result_path}")
            
            if return_type == 'glb':
                # GLB만 요청된 경우 파일 객체 형태로 반환
                file_obj = SimpleNamespace()
                file_obj.name = result_path
                return file_obj
            
            # URL 생성 (고정 템플릿 사용)
            glb_url = f"static/models/{result_filename}"
            print(f"GLB URL: {glb_url}, 여기로??")
            loader = SMPLAnimationLoader()
            loader.load_animation(skin_model.name)
            fps = loader.fps
            frame_count = len(loader.animation_data['poses']) if loader.animation_data and 'poses' in loader.animation_data else 0
            duration = frame_count / fps if fps > 0 else 0
            print(f"GLB URL: {glb_url}")
            
            # 고정 템플릿 사용 (viewer_template.html 파일을 직접 참조)
            viewer_url = f"{viewer_path}?model={glb_url}&duration={duration}&fps={fps}&frames={frame_count}"
            print(f"뷰어 URL: {viewer_url}")
            return None            

        # 파일명 생성
        skin_ext = Path(skin_model.name).suffix.lower()
        result_filename = f"anim_{unique_id}{skin_ext}"
        result_path = os.path.join(models_dir, result_filename)

        # 애니메이션 로더로 파일 처리
        # 파일 로드
        motion_data = np.load(anim_data.name, allow_pickle=True).item()
        
        # motion 키가 있는 경우 추출
        if isinstance(motion_data, np.ndarray) and motion_data.dtype == np.dtype('O') and isinstance(motion_data.item(), dict):
            if 'motion' in motion_data.item():
                motion_data = motion_data.item()['motion']
        elif isinstance(motion_data, dict) and 'motion' in motion_data:
            motion_data = motion_data['motion']
        
        print(f"Motion 데이터 형태: {type(motion_data)}")
        if isinstance(motion_data, np.ndarray):
            print(f"Shape: {motion_data.shape}, 차원: {motion_data.ndim}")

        # convert_mdm_to_glb 함수 존재 여부 확인
        try:
            # 함수 호출 전 디버그 정보 출력
            print(f"Convert 함수 호출: motion_data={type(motion_data)}, skin_model={skin_model.name}")
            output_files = create_improved_glb_animation(motion_data, result_path)
            
            if output_files and len(output_files) > 0:
                print(f"생성된 애니메이션 파일들: {output_files}")
                file_obj = SimpleNamespace()
                file_obj.name = output_files
                return file_obj
            else:
                print("convert_mdm_to_glb 함수에서 빈 결과 반환됨, 기본 로더로 처리")
        except Exception as convert_error:
            print(f"convert_mdm_to_glb 함수 호출 실패: {convert_error}")
            traceback.print_exc()
            print("기본 SMPLAnimationLoader로 처리를 계속합니다.")

    except Exception as e:
        print(f"SMPL 애니메이션 적용 오류: {e}")
        print(f"[DEBUG] 예외 발생 시점의 데이터: 프레임 수={frame_count if 'frame_count' in locals() else 'N/A'}, FPS={fps if 'fps' in locals() else 'N/A'}")
        traceback.print_exc()
        
        # # GLB 요청인 경우 오류 발생시 원본 스킨 반환
        # if return_type == 'glb':
        #     file_obj = SimpleNamespace()
        #     file_obj.name = skin_model.name
        #     return file_obj
        
        # 오류가 발생해도 최대한 결과를 보여주기 위해 시도
        try:
            # 파일명 생성
            skin_ext = Path(skin_model.name).suffix.lower()
            result_filename = f"error_{unique_id}{skin_ext}"
            result_path = os.path.join(models_dir, result_filename)
            
            # 스킨 모델만 복사
            shutil.copy2(skin_model.name, result_path)
            
            # URL 생성 (고정 템플릿 사용)
            glb_url = f"static/models/{result_filename}"
            # 애니메이션 메타데이터를 URL에 추가
            fps = loader.fps
            frame_count = len(loader.animation_data['poses'])
            duration = frame_count / fps
            viewer_url = f"{viewer_path}?model={glb_url}&duration={duration}&fps={fps}&frames={frame_count}"
            print(f"[DEBUG] 애니메이션 적용 - 뷰어 URL: {viewer_url}")
            print(f"[DEBUG] 파라미터: duration={duration}, fps={fps}, frames={frame_count}")

            return f"""
            <div style="width: 100%; height: 500px; border-radius: 8px; overflow: hidden; position: relative;">
                <div id="loading-overlay-{unique_id}" style="position: absolute; width: 100%; height: 100%; 
                      background-color: rgba(0,0,0,0.7); color: white; display: flex; justify-content: center; 
                      align-items: center; z-index: 10;">
                    <div style="text-align: center;">
                        <h3>오류 복구 모드</h3>
                        <p>애니메이션 적용 중 오류가 발생했습니다. 스킨 모델만 표시합니다.</p>
                    </div>
                </div>
                <iframe id="model-viewer-{unique_id}" src="{viewer_url}" 
                        style="width: 100%; height: 100%; border: none;"
                        onload="document.getElementById('loading-overlay-{unique_id}').style.display='none';">
                </iframe>
            </div>
            
            <div style="margin-top: 10px;">
                <p style="margin-bottom: 5px; color: #ff6b6b;">⚠️ 오류 발생</p>
                <p>SMPL 애니메이션을 GLB에 적용하는 중 오류가 발생했습니다.</p>
                <p><strong>오류 메시지:</strong> {str(e)}</p>
            </div>
            """
        except:
            # 완전히 실패한 경우
            return f"""
            <div style="width: 100%; height: 500px; background-color: #333; border-radius: 8px; 
                     display: flex; justify-content: center; align-items: center; color: #ccc;">
                <div style="text-align: center;">
                    <h3>SMPL 애니메이션 적용 오류</h3>
                    <p>{str(e)}</p>
                    <p>NPY 파일이 올바른 SMPL 형식인지 확인하세요.</p>
                </div>
            </div>
            """
