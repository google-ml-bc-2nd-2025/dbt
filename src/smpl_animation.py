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

from bone_mappings import SMPL_JOINT_NAMES, MIXAMO_TO_SMPL, find_matching_bones, get_smpl_joint_index

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
    if output_path is None:
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
            if target.path == "rotation":  # 회전 데이터만 처리
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
            if joint_id < 24:  # SMPL 모델은 일반적으로 24개의 관절을 가짐
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
            
            if ext == '.json':
                with open(filepath, 'r') as f:
                    self.animation_data = json.load(f)
                    
                    # FPS 정보 확인
                    if 'fps' in self.animation_data:
                        self.fps = self.animation_data['fps']
                    
                    # axis-angle 형식으로 로드
                    self.use_6d_rotation = False
                    print(f"JSON 애니메이션 로드 성공: {len(self.animation_data['poses'])} 프레임")
                    return True
                    
            elif ext == '.npy':
                poses = np.load(filepath, allow_pickle=True)
                
                # NPY 파일은 포즈 데이터만 포함하므로 나머지 필드는 기본값으로 설정
                self.animation_data = {
                    'poses': poses.tolist() if isinstance(poses, np.ndarray) else poses,
                    'shape': [0.0] * 10,
                    'trans': [[0.0, 0.0, 0.0] for _ in range(len(poses))],
                    'fps': self.fps
                }
                
                # 기본적으로 axis-angle 형식으로 간주
                self.use_6d_rotation = False
                print(f"NPY 애니메이션 로드 성공: {len(self.animation_data['poses'])} 프레임")
                return True
            
            elif ext == '.npz':
                npz_data = np.load(filepath, allow_pickle=True)
                
                if 'poses_6d' in npz_data:
                    # 6D 회전 표현이 있는 경우
                    self.poses_6d = npz_data['poses_6d']
                    self.use_6d_rotation = True
                    
                    # axis-angle 형식이 있으면 함께 로드 (역호환성)
                    if 'poses_aa' in npz_data:
                        poses_aa = npz_data['poses_aa']
                        self.animation_data = {
                            'poses': poses_aa.tolist() if isinstance(poses_aa, np.ndarray) else poses_aa,
                            'shape': [0.0] * 10,
                            'trans': [[0.0, 0.0, 0.0] for _ in range(len(poses_aa))],
                            'fps': npz_data.get('fps', self.fps)
                        }
                    else:
                        # 6D 표현을 axis-angle로 변환하여 호환성 유지
                        poses_aa = rotation_6d_to_axis_angle(self.poses_6d)
                        self.animation_data = {
                            'poses': poses_aa.tolist(),
                            'shape': [0.0] * 10,
                            'trans': [[0.0, 0.0, 0.0] for _ in range(len(poses_aa))],
                            'fps': npz_data.get('fps', self.fps)
                        }
                    
                    frame_count = npz_data.get('frame_count', len(self.poses_6d))
                    print(f"NPZ 애니메이션 로드 성공 (6D 회전 표현): {frame_count} 프레임")
                    return True
                
                elif 'poses' in npz_data or 'poses_aa' in npz_data:
                    # 기존 형식의 NPZ 파일 (축-각도 표현)
                    poses = npz_data.get('poses', npz_data.get('poses_aa', None))
                    
                    if poses is not None:
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
        if not self.animation_data or 'poses' not in self.animation_data:
            print("로드된 애니메이션 데이터가 없습니다.")
            return {}
        
        joint_rotations = {joint_name: [] for joint_name in SMPL_JOINT_NAMES}
        
        # 6D 회전 표현을 사용하는 경우
        if self.use_6d_rotation and self.poses_6d is not None:
            # 6D 회전 표현을 축-각도로 변환
            poses_aa = rotation_6d_to_axis_angle(self.poses_6d)
            
            # 각 프레임마다 관절 회전 데이터 추출
            for frame_idx in range(poses_aa.shape[0]):
                frame_pose = poses_aa[frame_idx]  # (72,) 형태
                
                # 프레임의 모든 관절에 대해 처리
                for joint_idx, joint_name in enumerate(SMPL_JOINT_NAMES):
                    if joint_idx < 22:  # 6D 회전은 22개 관절까지만 사용
                        # 관절별 3차원 회전 벡터 추출
                        rot_start = joint_idx * 3
                        rot_end = rot_start + 3
                        
                        if rot_end <= len(frame_pose):
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
                    
                    if rot_end <= len(frame_pose):
                        rotation = frame_pose[rot_start:rot_end]
                        joint_rotations[joint_name].append(rotation)
        
        return joint_rotations
    
    def create_keyframes(self, target_fps=30):
        """
        애니메이션 키프레임 데이터 생성
        
        Args:
            target_fps: 대상 프레임 레이트
        
        Returns:
            키프레임 딕셔너리 {joint_name: {'rotation': [...], 'times': [...]}}
        """
        if not self.animation_data:
            return {}
        
        joint_rotations = self.get_joint_rotations()
        keyframes = {}
        
        # 프레임 수 확인
        frame_count = len(self.animation_data['poses'])
        if frame_count == 0:
            return {}
        
        # 프레임 간격 계산 (초 단위)
        src_fps = self.animation_data.get('fps', self.fps)
        frame_time = 1.0 / src_fps
        
        # 시간값 생성
        times = [i * frame_time for i in range(frame_count)]
        
        # 목표 FPS로 보간할 시간값 생성
        if target_fps != src_fps:
            target_frame_time = 1.0 / target_fps
            target_duration = times[-1] + frame_time
            target_frame_count = int(target_duration * target_fps)
            target_times = [i * target_frame_time for i in range(target_frame_count)]
        else:
            target_times = times
        
        # 각 관절에 대한 키프레임 생성
        for joint_name, rotations in joint_rotations.items():
            if not rotations:
                continue
                
            # 쿼터니언 변환 및 보간을 위한 준비
            rotvecs = np.array(rotations)
            
            # 회전 벡터를 쿼터니언으로 변환
            quats = R.from_rotvec(rotvecs).as_quat()  # (frame_count, 4) 형태
            
            # 목표 FPS로 보간이 필요한 경우
            if target_fps != src_fps:
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
        if not self.animation_data:
            print("애니메이션 데이터가 없습니다.")
            return False
        
        try:
            # GLB 파일 로드
            gltf = pygltflib.GLTF2().load(glb_path)
            print(f"GLB 모델 로드 성공: {glb_path}")
            
            # 스켈레톤 분석
            if not gltf.nodes:
                print("GLB 모델에 노드가 없습니다.")
                return False
                
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
                return False
            
            # 키프레임 데이터 생성
            keyframes = self.create_keyframes(target_fps=30)  # 30fps로 통일
            
            # 애니메이션 길이 계산 (초)
            frame_count = len(self.animation_data['poses'])
            duration = frame_count / self.fps
            
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
                max=[float(max(times))],
                min=[float(min(times))]
            )
            gltf.accessors.append(time_accessor)
            
            animation_buffer_data.extend(time_data)
            time_accessor_index = accessor_index
            
            accessor_index += 1
            buffer_view_index += 1
            
            # 매핑된 본에 애니메이션 적용
            for model_bone, smpl_bone in matches.items():
                if model_bone not in bone_indices:
                    continue
                    
                node_index = bone_indices[model_bone]
                
                # SMPL 본 이름 찾기
                smpl_joint_name = MIXAMO_TO_SMPL.get(smpl_bone)
                if not smpl_joint_name or smpl_joint_name not in keyframes:
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
                    interpolation="LINEAR"
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
            
            # 나머지 본들을 위한 보간 로직
            # 각 본의 부모-자식 관계 분석
            child_to_parent = {}
            for i, node in enumerate(gltf.nodes):
                if hasattr(node, 'children') and node.children:
                    for child in node.children:
                        child_to_parent[child] = i
            
            # 매핑되지 않은 본에 대해서는 부모 또는 자식 본의 애니메이션을 보간
            for i, node in enumerate(gltf.nodes):
                if not hasattr(node, 'name') or not node.name or node.name in matches.values():
                    continue
                
                # 부모 또는 자식 본 중 애니메이션이 있는 본 찾기
                parent_idx = child_to_parent.get(i)
                children = [j for j, parent in child_to_parent.items() if parent == i]
                
                reference_rotations = None
                
                # 부모 본의 회전 확인
                if parent_idx is not None and hasattr(gltf.nodes[parent_idx], 'name'):
                    parent_name = gltf.nodes[parent_idx].name
                    if parent_name in matches.values():
                        smpl_joint = MIXAMO_TO_SMPL.get(parent_name)
                        if smpl_joint in keyframes:
                            reference_rotations = keyframes[smpl_joint]['rotation']
                
                # 자식 본의 회전 확인 (부모 본에 애니메이션이 없는 경우)
                if reference_rotations is None and children:
                    for child_idx in children:
                        if hasattr(gltf.nodes[child_idx], 'name'):
                            child_name = gltf.nodes[child_idx].name
                            if child_name in matches.values():
                                smpl_joint = MIXAMO_TO_SMPL.get(child_name)
                                if smpl_joint in keyframes:
                                    reference_rotations = keyframes[smpl_joint]['rotation']
                                    break
                
                # 참조할 회전 데이터가 있으면 약간의 감쇠를 적용하여 사용
                if reference_rotations:
                    # 감쇠 계수 (부모/자식 본의 움직임을 감소시키기 위함)
                    damping = 0.7
                    
                    # 기본 회전 가져오기 (있는 경우)
                    default_rotation = node.rotation if hasattr(node, 'rotation') and node.rotation else [0, 0, 0, 1]
                    
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
                        interpolation="LINEAR"
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
            
            # 애니메이션 객체 생성
            animation = pygltflib.Animation(
                name="SMPLAnimation",
                channels=channels,
                samplers=samplers
            )
            gltf.animations.append(animation)
            
            # 바이너리 버퍼 업데이트
            if not gltf.buffers:
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
            if len(glb_data) > json_chunk_end + 8:
                bin_chunk_length = int.from_bytes(glb_data[json_chunk_end:json_chunk_end+4], byteorder='little')
                bin_chunk_type = glb_data[json_chunk_end+4:json_chunk_end+8]
                
                if bin_chunk_type == b'BIN\0':
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
            if not output_path:
                base_name = os.path.splitext(glb_path)[0]
                output_path = f"{base_name}_animated.glb"
            
            # 직접 바이너리를 파일에 쓰기
            with open(output_path, 'wb') as f:
                f.write(new_glb_data)
            print(f"애니메이션이 적용된 GLB 파일 저장 완료: {output_path}")
            
            # 임시 파일 제거
            if os.path.exists(temp_output):
                os.remove(temp_output)
            
            return output_path
            
        except Exception as e:
            print(f"GLB 파일에 애니메이션 적용 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return False

def apply_to_glb(skin_model, anim_data, viewer_path, models_dir):
    """
    SMPL 애니메이션 데이터를 GLB 모델에 적용합니다.
    
    Args:
        skin_model: 스킨 모델 파일 객체 (GLB 형식)
        anim_data: SMPL 애니메이션 파일 객체 (NPY 형식)
        viewer_path: 뷰어 HTML 파일의 경로
        models_dir: 모델 파일이 저장될 디렉토리
    
    Returns:
        HTML 문자열 (iframe으로 뷰어 출력)
    """
    if skin_model is None:
        return "스킨 모델을 먼저 업로드해주세요."
    
    try:
        # 고유 ID 생성
        unique_id = str(uuid.uuid4())[:8]
        
        # 애니메이션이 없는 경우 스킨 파일만 처리
        if anim_data is None or not hasattr(anim_data, 'name') or not anim_data.name.lower().endswith('.npy'):
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
            
            # URL 생성
            glb_url = f"static/models/{result_filename}"
            print(f"GLB URL: {glb_url}")
            viewer_url = f"{viewer_path}?model={glb_url}"
            print(f"뷰어 URL: {viewer_url}")
            
            # iframe으로 뷰어 표시
            return f"""
            <div style="width: 100%; height: 500px; border-radius: 8px; overflow: hidden; position: relative;">
                <div id="loading-overlay-{unique_id}" style="position: absolute; width: 100%; height: 100%; 
                      background-color: rgba(0,0,0,0.7); color: white; display: flex; justify-content: center; 
                      align-items: center; z-index: 10;">
                    <div style="text-align: center;">
                        <h3>모델 로딩 중...</h3>
                        <p>잠시만 기다려주세요.</p>
                    </div>
                </div>
                <iframe id="model-viewer-{unique_id}" src="{viewer_url}" 
                        style="width: 100%; height: 100%; border: none;"
                        onload="document.getElementById('loading-overlay-{unique_id}').style.display='none';">
                </iframe>
            </div>
            
            <div style="margin-top: 10px;">
                <p style="margin-bottom: 5px;">모델 정보:</p>
                <ul style="margin-top: 0;">
                    <li>파일: {os.path.basename(skin_model.name)}</li>
                    <li>애니메이션: 없음 (정적 모델)</li>
                </ul>
            </div>
            """
        
        # 파일명 생성
        skin_ext = Path(skin_model.name).suffix.lower()
        result_filename = f"anim_{unique_id}{skin_ext}"
        result_path = os.path.join(models_dir, result_filename)
        
        print(f"출력 파일 경로: {result_path}")
        print(f"models_dir: {models_dir}")
        
        # 기존 파일 동작 확인
        print(f"스킨 모델 확인: {skin_model.name}, 존재함: {os.path.exists(skin_model.name)}")
        print(f"애니메이션 파일 확인: {anim_data.name}, 존재함: {os.path.exists(anim_data.name)}")
        
        # SMPLAnimationLoader를 직접 사용하여 애니메이션 적용
        loader = SMPLAnimationLoader()
        
        # 애니메이션 파일 로드
        loader.load_animation(anim_data.name)
        
        if not loader.animation_data:
            print("애니메이션 데이터 로드 실패")
            # 스킨 모델만 복사
            shutil.copy2(skin_model.name, result_path)
            
            # URL 생성 및 결과 반환
            glb_url = f"static/models/{result_filename}"
            viewer_url = f"{viewer_path}?model={glb_url}"
            
            return f"""
            <div style="width: 100%; height: 500px; border-radius: 8px; overflow: hidden; position: relative;">
                <div id="loading-overlay-{unique_id}" style="position: absolute; width: 100%; height: 100%; 
                      background-color: rgba(0,0,0,0.7); color: white; display: flex; justify-content: center; 
                      align-items: center; z-index: 10;">
                    <div style="text-align: center;">
                        <h3>모델 로딩 중...</h3>
                        <p>애니메이션 데이터 로드에 실패했습니다. 스킨 모델만 표시합니다.</p>
                    </div>
                </div>
                <iframe id="model-viewer-{unique_id}" src="{viewer_url}" 
                        style="width: 100%; height: 100%; border: none;"
                        onload="document.getElementById('loading-overlay-{unique_id}').style.display='none';">
                </iframe>
            </div>
            
            <div style="margin-top: 10px;">
                <p style="margin-bottom: 5px; color: #ff6b6b;">⚠️ 애니메이션 데이터 로드 실패</p>
                <p>NPY 파일에서 유효한 애니메이션 데이터를 찾을 수 없습니다.</p>
            </div>
            """
        
        print(f"애니메이션 데이터 로드 성공: {len(loader.animation_data['poses'])} 프레임")
        
        # SMPLAnimationLoader의 apply_to_glb 메서드 사용
        output_path = loader.apply_to_glb(skin_model.name, result_path)
        
        if not output_path or not os.path.exists(output_path):
            print(f"애니메이션 적용 실패: 출력 파일 {output_path}가 존재하지 않습니다.")
            # 스킨 모델만 복사
            shutil.copy2(skin_model.name, result_path)
            
            # URL 생성 및 결과 반환
            glb_url = f"static/models/{result_filename}"
            viewer_url = f"{viewer_path}?model={glb_url}"
            
            return f"""
            <div style="width: 100%; height: 500px; border-radius: 8px; overflow: hidden; position: relative;">
                <div id="loading-overlay-{unique_id}" style="position: absolute; width: 100%; height: 100%; 
                      background-color: rgba(0,0,0,0.7); color: white; display: flex; justify-content: center; 
                      align-items: center; z-index: 10;">
                    <div style="text-align: center;">
                        <h3>모델 로딩 중...</h3>
                        <p>애니메이션 적용에 실패하여 스킨 모델만 표시합니다.</p>
                    </div>
                </div>
                <iframe id="model-viewer-{unique_id}" src="{viewer_url}" 
                        style="width: 100%; height: 100%; border: none;"
                        onload="document.getElementById('loading-overlay-{unique_id}').style.display='none';">
                </iframe>
            </div>
            
            <div style="margin-top: 10px;">
                <p style="margin-bottom: 5px; color: #ff6b6b;">⚠️ 애니메이션 적용 실패</p>
                <p>SMPL 애니메이션을 GLB에 적용하는 중 오류가 발생했습니다. 스킨 모델만 표시합니다.</p>
            </div>
            """
        
        print(f"애니메이션 적용 성공: {output_path}")
        print(f"출력 파일 확인: {output_path}, 크기: {os.path.getsize(output_path)} 바이트")
        
        # 출력 파일이 result_path와 다른 경우 복사
        if output_path != result_path:
            shutil.copy2(output_path, result_path)
            print(f"결과 파일 복사: {output_path} -> {result_path}")
        
        # 파일 생성 확인
        if os.path.exists(result_path):
            print(f"최종 파일 확인: {result_path}, 크기: {os.path.getsize(result_path)} 바이트")
        else:
            print(f"오류: 최종 파일이 생성되지 않음: {result_path}")
            return "파일 생성 실패: 결과 파일을 찾을 수 없습니다."
        
        # URL 생성
        glb_url = f"static/models/{result_filename}"
        print(f"최종 GLB URL: {glb_url}")
        viewer_url = f"{viewer_path}?model={glb_url}"
        print(f"최종 뷰어 URL: {viewer_url}")
        
        # 애니메이션 메타데이터 (가능한 경우)
        fps = loader.fps
        frame_count = len(loader.animation_data['poses']) if loader.animation_data and 'poses' in loader.animation_data else 0
        
        # iframe으로 뷰어 표시
        return f"""
        <div style="width: 100%; height: 500px; border-radius: 8px; overflow: hidden; position: relative;">
            <div id="loading-overlay-{unique_id}" style="position: absolute; width: 100%; height: 100%; 
                  background-color: rgba(0,0,0,0.7); color: white; display: flex; justify-content: center; 
                  align-items: center; z-index: 10;">
                <div style="text-align: center;">
                    <h3>애니메이션 모델 로딩 중...</h3>
                    <p>잠시만 기다려주세요.</p>
                </div>
            </div>
            <iframe id="model-viewer-{unique_id}" src="{viewer_url}" 
                    style="width: 100%; height: 100%; border: none;"
                    onload="document.getElementById('loading-overlay-{unique_id}').style.display='none';">
            </iframe>
        </div>
        
        <div style="margin-top: 10px;">
            <p style="margin-bottom: 5px;">모델 정보:</p>
            <ul style="margin-top: 0;">
                <li>프레임 수: {frame_count}</li>
                <li>FPS: {fps}</li>
                <li>애니메이션 길이: {frame_count / fps:.2f}초</li>
            </ul>
        </div>
        
        <div style="margin-top: 10px;">
            <p style="margin-bottom: 5px; font-weight: bold;">애니메이션 제어:</p>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <button onclick="document.getElementById('model-viewer-{unique_id}').contentWindow.postMessage({{'action': 'play'}}, '*')">
                    재생
                </button>
                <button onclick="document.getElementById('model-viewer-{unique_id}').contentWindow.postMessage({{'action': 'pause'}}, '*')">
                    일시정지
                </button>
                <button onclick="document.getElementById('model-viewer-{unique_id}').contentWindow.postMessage({{'action': 'reset'}}, '*')">
                    리셋
                </button>
            </div>
        </div>
        """
            
    except Exception as e:
        print(f"SMPL 애니메이션 적용 오류: {e}")
        import traceback
        traceback.print_exc()
        
        # 오류가 발생해도 최대한 결과를 보여주기 위해 시도
        try:
            # 파일명 생성
            skin_ext = Path(skin_model.name).suffix.lower()
            result_filename = f"error_{unique_id}{skin_ext}"
            result_path = os.path.join(models_dir, result_filename)
            
            # 스킨 모델만 복사
            shutil.copy2(skin_model.name, result_path)
            
            # URL 생성
            glb_url = f"static/models/{result_filename}"
            viewer_url = f"{viewer_path}?model={glb_url}"
            
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
