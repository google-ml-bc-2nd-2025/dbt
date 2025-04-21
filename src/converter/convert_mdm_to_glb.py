import pygltflib
import numpy as np
import os
import uuid
import struct
from scipy.spatial.transform import Rotation as R
from model.smpl import joint_positions, smpl_humanml3d_to_mixamo_index, joint_connections # Ensure this import is correct
import numpy as np
import pickle
from pathlib import Path

# 뼈대 구조 정의 (이름과 계층 구조)
bone_names = [
    "mixamorig:Hips",              # 0
    "mixamorig:LeftUpLeg",         # 1
    "mixamorig:RightUpLeg",        # 2
    "mixamorig:Spine",             # 3
    "mixamorig:LeftLeg",           # 4
    "mixamorig:RightLeg",          # 5
    "mixamorig:Spine1",            # 6
    "mixamorig:Spine2",            # 9
    "mixamorig:LeftFoot",          # 7
    "mixamorig:RightFoot",         # 8
    "mixamorig:LeftToeBase",       # 10
    "mixamorig:RightToeBase",      # 11
    "mixamorig:Neck",              # 12
    "mixamorig:Head",              # 15
    "mixamorig:LeftShoulder",      # 13
    "mixamorig:RightShoulder",     # 14
    "mixamorig:LeftArm",           # 16
    "mixamorig:RightArm",          # 17
    "mixamorig:LeftForeArm",       # 18
    "mixamorig:RightForeArm",      # 19
    "mixamorig:LeftHand",          # 20
    "mixamorig:RightHand"          # 21
]


# 부모-자식 관계 정의
hierarchy = {
    1: 0,   # LeftUpLeg -> Hips
    2: 0,   # RightUpLeg -> Hips
    3: 0,   # Spine -> Hips
    4: 1,   # LeftLeg -> LeftUpLeg
    5: 2,   # RightLeg -> RightUpLeg
    6: 3,   # Spine1 -> Spine
    7: 4,   # LeftFoot -> LeftLeg
    8: 5,   # RightFoot -> RightLeg
    9: 6,   # Spine2 -> Spine1
    10: 7,  # LeftToeBase -> LeftFoot
    11: 8,  # RightToeBase -> RightFoot
    12: 9,  # Neck -> Spine2
    13: 12, # LeftShoulder -> Neck
    14: 12, # RightShoulder -> Neck
    15: 12, # Head -> Neck
    16: 13, # LeftArm -> LeftShoulder
    17: 14, # RightArm -> RightShoulder
    18: 16, # LeftForeArm -> LeftArm
    19: 17, # RightForeArm -> RightArm
    20: 18, # LeftHand -> LeftForeArm
    21: 19  # RightHand -> RightForeArm
}

# 관절 위치 (T-포즈)
positions = {
    0: [0.0, 0.0, 0.0],              # Hips
    1: [-0.1, -0.05, 0.0],           # LeftUpLeg
    2: [0.1, -0.05, 0.0],            # RightUpLeg
    3: [0.0, 0.1, 0.0],              # Spine
    4: [-0.1, -0.5, 0.0],            # LeftLeg
    5: [0.1, -0.5, 0.0],             # RightLeg
    6: [0.0, 0.2, 0.0],              # Spine1
    7: [-0.1, -1.0, 0.0],            # LeftFoot
    8: [0.1, -1.0, 0.0],             # RightFoot
    9: [0.0, 0.3, 0.0],              # Spine2
    10: [-0.1, -1.1, 0.1],           # LeftToeBase
    11: [0.1, -1.1, 0.1],            # RightToeBase
    12: [0.0, 0.4, 0.0],             # Neck
    13: [-0.1, 0.4, 0.0],            # LeftShoulder
    14: [0.1, 0.4, 0.0],             # RightShoulder
    15: [0.0, 0.5, 0.0],             # Head
    16: [-0.3, 0.4, 0.0],            # LeftArm
    17: [0.3, 0.4, 0.0],             # RightArm
    18: [-0.6, 0.4, 0.0],            # LeftForeArm
    19: [0.6, 0.4, 0.0],             # RightForeArm
    20: [-0.9, 0.4, 0.0],            # LeftHand
    21: [0.9, 0.4, 0.0]              # RightHand
}


def fix_rotation_for_joint_type(rotation_vectors, joint_idx):
    """
    특정 관절 유형에 따라 회전 벡터를 보정
    
    Args:
        rotation_vectors: 회전 벡터 배열 (F, 3) 형태
        joint_idx: 관절 인덱스
        
    Returns:
        보정된 회전 벡터 배열 (F, 3) 형태
    """
    fixed = fix_rotation_axes(rotation_vectors)
    
    # 머리/목 관절 (12, 15)에 대한 특별 처리
    if joint_idx in [12, 15]:  # Neck, Head
        # Y축 회전 보정 (머리 방향 조정)
        euler = R.from_rotvec(fixed).as_euler('xyz', degrees=True)
        euler[:, 1] = -euler[:, 1]  # Y축 방향 반전
        fixed = R.from_euler('xyz', euler, degrees=True).as_rotvec()
    
    # 다리 관절 (1, 2, 4, 5, 7, 8) - 다리가 아래쪽을 향하도록 조정
    elif joint_idx in [1, 2, 4, 5, 7, 8]:  # LeftUpLeg, RightUpLeg, 등
        # 다리는 X축 회전이 주 움직임
        # X축 회전 방향을 조정하여 올바른 방향으로 움직이게 함
        euler = R.from_rotvec(fixed).as_euler('xyz', degrees=True)
        euler[:, 0] = -euler[:, 0]  # X축 방향 추가 반전 (다리 특화)
        fixed = R.from_euler('xyz', euler, degrees=True).as_rotvec()
    
    # 팔 관절 (16-21)에 대한 보정
    elif 16 <= joint_idx <= 21:  # 팔 관절 (왼쪽/오른쪽 어깨, 팔꿈치, 손목)
        # 팔 방향 조정
        if joint_idx in [16, 18, 20]:  # 왼쪽 팔 관절
            # 왼쪽 팔의 Z축 회전 보정
            euler = R.from_rotvec(fixed).as_euler('xyz', degrees=True)
            euler[:, 2] = -euler[:, 2]  # Z축 방향 조정
            fixed = R.from_euler('xyz', euler, degrees=True).as_rotvec()
    
    return fixed

def compute_local_rotations(global_rotations, joint_hierarchy):
    """
    글로벌 회전 벡터를 로컬 회전 벡터로 변환
    
    Args:
        global_rotations: 글로벌 회전 벡터 배열 (N, 3) 형태
        joint_hierarchy: 관절 계층 구조 (자식->부모 인덱스 맵)
    
    Returns:
        로컬 회전 벡터 배열 (N, 3) 형태
    """
    local_rotations = np.zeros_like(global_rotations)
    
    # 처리 순서 결정 (부모부터 자식 순서로)
    processing_order = []
    remaining = set(range(len(global_rotations)))
    
    # 루트 노드 먼저 추가
    root_nodes = [i for i in remaining if i not in joint_hierarchy or joint_hierarchy[i] == -1]
    for root in root_nodes:
        processing_order.append(root)
        if root in remaining:
            remaining.remove(root)
    
    # 나머지 노드를 부모-자식 순서로 추가
    while remaining:
        for i in list(remaining):
            parent_idx = joint_hierarchy.get(i, -1)
            if parent_idx == -1 or parent_idx not in remaining:
                processing_order.append(i)
                remaining.remove(i)
    
    # 각 관절에 대해 로컬 회전 계산
    for i in processing_order:
        if i not in joint_hierarchy or joint_hierarchy[i] == -1:  # 루트 관절
            local_rotations[i] = global_rotations[i]
        else:
            parent_idx = joint_hierarchy[i]
            
            # 부모와 자신의 글로벌 회전 가져오기
            parent_global_rotation = R.from_rotvec(global_rotations[parent_idx])
            current_global_rotation = R.from_rotvec(global_rotations[i])
            
            # 부모 회전의 역을 적용하여 로컬 회전 계산
            local_rotation = parent_global_rotation.inv() * current_global_rotation
            local_rotations[i] = local_rotation.as_rotvec()
    
    return local_rotations

# 디버깅 정보 출력을 위한 함수
def print_debug_info(label, data, max_samples=3):
    print(f"\n===== {label} =====")
    if isinstance(data, np.ndarray):
        print(f"Shape: {data.shape}")
        if data.size > 0:
            print(f"Sample data (처음 {max_samples}개):")
            print(data[:min(max_samples, len(data))])
    else:
        print(data)
    print("=" * (len(label) + 12))

def fix_rotation_axes(rotation_vectors):
    fixed = np.copy(rotation_vectors)
    
    # 일반적인 회전 변환은 유지
    fixed[:, 0] = -fixed[:, 0]  # X축 회전 방향 반전
    
    # 다리와 발에 대한 특별한 변환이 필요하다면 여기서 처리
    # 예: 특정 축 회전에 오프셋 추가
    
    # 회전 벡터의 크기가 작은 경우 처리
    small_rotation_mask = np.sum(fixed**2, axis=1) < 1e-10
    fixed[small_rotation_mask] = np.zeros(3)
    
    return fixed

# 대안 변환 방법 시도
def rotvec_to_quat(rotvec):
    """회전 벡터에서 쿼터니언으로 직접 변환 (대체용)"""
    theta = np.linalg.norm(rotvec, axis=1, keepdims=True)
    axis = np.zeros_like(rotvec)
    non_zero = theta > 1e-10
    axis[non_zero] = rotvec[non_zero] / theta[non_zero]
    
    half_theta = theta * 0.5
    cos_half = np.cos(half_theta)
    sin_half = np.sin(half_theta)
    
    qx = axis[:, 0:1] * sin_half
    qy = axis[:, 1:2] * sin_half
    qz = axis[:, 2:3] * sin_half
    qw = cos_half
    
    return np.hstack([qx, qy, qz, qw])

def create_animation_only_glb(motion_data, output_path=None, fps=30, joint_map=None):
    """
    애니메이션 데이터만 포함하는 최소한의 GLB 파일 생성
    
    Args:
        motion_data: (N, 22, 3, F) 형태의 모션 데이터
        output_path: 출력 파일 경로
        fps: 애니메이션 프레임 레이트
        joint_map: 모델 관절 이름과 인덱스 매핑 (None이면 기본값 사용)
    """
    if output_path is None:
        base_name = os.path.join(os.getcwd(), "animation")
    else:
        base_name = os.path.splitext(output_path)[0]

    # 생성된 파일 경로 저장할 리스트
    output_glbs = []

    # 최소한의 GLTF 구조 생성
    gltf = pygltflib.GLTF2()
    
    # 필수 요소 초기화
    gltf.scene = 0
    gltf.scenes = [pygltflib.Scene(nodes=[0])]  # 루트 노드만 있는 씬
    
    # 노드 계층 구조 생성
    # 루트 노드와 함께 필요한 뼈대 노드들도 생성
    root_node = pygltflib.Node()
    gltf.nodes = [root_node]
    
    # 관절 매핑 설정 (없으면 생성)
    bone_indices = {}
    if joint_map is None:
        joint_map = {}
        print("기본 관절 매핑 사용...")
        # SMPL -> Mixamo 이름 기반 매핑
        for i, mixamo_name in enumerate(smpl_humanml3d_to_mixamo_index):
            if i >= 22: break  # SMPL 22개 관절 기준
            joint_map[i] = mixamo_name
            
            # 노드 추가 (index+1 위치에 추가, 0번은 루트)
            if i > 0:  # 0번은 이미 루트로 추가됨
                node = pygltflib.Node(
                    name=mixamo_name,
                    translation=[0.0, 0.0, 0.0],  # 기본 위치
                    rotation=[0.0, 0.0, 0.0, 1.0]  # 기본 회전 (쿼터니언)
                )
                
                # SMPL 모델의 본 길이 정보를 이용하여 위치 설정
                # if i in joint_positions:
                #     node.translation = [float(pos) for pos in joint_positions[i]]
                # 각 관절의 기본 회전 설정
           
                # 루트 노드에 전체 모델 회전 적용
                # Y축으로 180도 회전하여 앞/뒤 방향 조정
                root_correction = R.from_euler('y', 180, degrees=True).as_quat()
                root_node.rotation = [float(q) for q in root_correction]
                
                gltf.nodes.append(node)
                bone_indices[mixamo_name] = len(gltf.nodes) - 1
                print(f"자식 노드 추가: {mixamo_name}, 인덱스: {len(gltf.nodes) - 1}")  # 로그 추가

    # 계층 구조 설정
    for i in range(1, len(smpl_humanml3d_to_mixamo_index)):
        # 관절 이름 가져오기
        joint_name = smpl_humanml3d_to_mixamo_index[i]
        
        # 부모 관절 찾기 (예: SMPL 계층 구조 기반)
        for child_idx, parent_idx in joint_connections:
            if child_idx == i:
                # print(f"관절 {joint_name}의 부모 인덱스: {parent_idx}")  # 로그 추가
                if parent_idx < len(smpl_humanml3d_to_mixamo_index):
                    parent_name = smpl_humanml3d_to_mixamo_index[parent_idx]
                    
                    # 부모-자식 관계 설정
                    if parent_name in bone_indices and joint_name in bone_indices:
                        parent_node = gltf.nodes[bone_indices[parent_name]]
                        child_node_idx = bone_indices[joint_name]
                        
                        # 부모 노드에 자식 추가
                        if not hasattr(parent_node, 'children'):
                            parent_node.children = []
                        parent_node.children.append(child_node_idx)

    # 빈 버퍼 생성
    gltf.buffers = [pygltflib.Buffer(byteLength=0)]
    binary_blob = bytearray()
    
    # 애니메이션 처리 전에 글로벌 회전을 로컬 회전으로 변환
    hierarchy_map = {}
    for child_idx, parent_idx in joint_connections:
        if child_idx < 22:  # 22개 관절만 사용
            hierarchy_map[child_idx] = parent_idx

    # 애니메이션 처리
    for animation_idx in range(motion_data.shape[0]):
        current_motion = motion_data[animation_idx]  # (22, 3, F)
        frame_count = current_motion.shape[2]
        animation_data = np.transpose(current_motion, (2, 0, 1))  # (F, 22, 3)
        
        # 글로벌 -> 로컬 변환
        for frame in range(animation_data.shape[0]):
            frame_rotations = animation_data[frame, :, :]  # (22, 3)
            local_rotations = compute_local_rotations(frame_rotations, hierarchy_map)
            animation_data[frame, :, :] = local_rotations

        # 애니메이션 샘플러 및 채널 리스트
        samplers = []
        channels = []
        
        # 시간 데이터 추가
        times = np.array([i * (1.0 / fps) for i in range(frame_count)], dtype=np.float32)
        time_data_bytes = times.tobytes()
        time_byte_offset = len(binary_blob)
        binary_blob.extend(time_data_bytes)
        
        time_buffer_view = pygltflib.BufferView(
            buffer=0,
            byteOffset=time_byte_offset,
            byteLength=len(time_data_bytes)
        )
        gltf.bufferViews.append(time_buffer_view)
        time_buffer_view_index = len(gltf.bufferViews) - 1
        
        time_accessor = pygltflib.Accessor(
            bufferView=time_buffer_view_index,
            componentType=pygltflib.FLOAT,
            count=frame_count,
            type=pygltflib.SCALAR,
            max=[float(times.max())],
            min=[float(times.min())]
        )
        gltf.accessors.append(time_accessor)
        time_accessor_index = len(gltf.accessors) - 1
        
        # 각 관절별로 회전 데이터 추가
        for joint_idx in range(min(22, animation_data.shape[1])):
            # 관절 이름 가져오기
            joint_name = joint_map.get(joint_idx)
            if not joint_name:
                continue
                
            # 해당 관절의 노드 인덱스 결정
            if joint_idx == 0:
                node_index = 0  # 루트 노드는 항상 0번
                print(f"루트 노드 애니메이션 트랙: 노드 인덱스 {node_index}, 이름: {gltf.nodes[node_index].name}")  # 로그 추가
            else:
                # joint_name이 bone_indices에 없으면 새로 추가
                if joint_name not in bone_indices:
                    node = pygltflib.Node(name=joint_name)
                    gltf.nodes.append(node)
                    bone_indices[joint_name] = len(gltf.nodes) - 1
                    print(f"새로운 자식 노드 추가: {joint_name}, 인덱스: {len(gltf.nodes) - 1}")  # 로그 추가
                node_index = bone_indices[joint_name]
                print(f"자식 노드 애니메이션 트랙: 노드 인덱스 {node_index}, 이름: {gltf.nodes[node_index].name}")  # 로그 추가
                        
            # 관절의 회전 데이터 추출
            joint_rotations = animation_data[:, joint_idx, :]  # (F, 3)

            joint_rotations = fix_rotation_for_joint_type(joint_rotations, joint_idx)  # 좌표계 변환
            
            # 쿼터니언 변환
            try:
                # 쿼터니언 변환
                quaternions = R.from_rotvec(joint_rotations).as_quat()  # (F, 4) [x, y, z, w]
                
                # NaN 값 재확인
                if np.isnan(quaternions).any():
                    identity_quat = np.array([0.0, 0.0, 0.0, 1.0])
                    nan_indices = np.isnan(quaternions).any(axis=1)
                    quaternions[nan_indices] = identity_quat
                    
            except Exception as e:
                print(f"쿼터니언 변환 오류 (관절 {joint_idx}): {e}")
                # 오류 발생 시 항등 쿼터니언으로 설정
                quaternions = np.array([[0.0, 0.0, 0.0, 1.0]] * frame_count)
            
            rotation_data_bytes = quaternions.astype(np.float32).tobytes()
            rot_byte_offset = len(binary_blob)
            binary_blob.extend(rotation_data_bytes)
            
            rot_buffer_view = pygltflib.BufferView(
                buffer=0,
                byteOffset=rot_byte_offset,
                byteLength=len(rotation_data_bytes)
            )
            gltf.bufferViews.append(rot_buffer_view)
            rot_buffer_view_index = len(gltf.bufferViews) - 1
            
            quat_min = np.min(quaternions, axis=0).astype(float).tolist()
            quat_max = np.max(quaternions, axis=0).astype(float).tolist()
            
            rot_accessor = pygltflib.Accessor(
                bufferView=rot_buffer_view_index,
                componentType=pygltflib.FLOAT,
                count=frame_count,
                type=pygltflib.VEC4,
                max=quat_max,
                min=quat_min
            )
            gltf.accessors.append(rot_accessor)
            rot_accessor_index = len(gltf.accessors) - 1
            
            # 애니메이션 샘플러 및 채널 생성
            sampler = pygltflib.AnimationSampler(
                input=time_accessor_index,
                output=rot_accessor_index,
                interpolation="LINEAR"
            )
            samplers.append(sampler)
            sampler_index = len(samplers) - 1

            channel = pygltflib.AnimationChannel(
                sampler=sampler_index,
                target=pygltflib.AnimationChannelTarget(
                    node=node_index,
                    path="rotation"
                )
            )
            print(f"애니메이션 채널 추가: 대상 노드 인덱스 {node_index}, 경로: rotation")  # 로그 추가
            channels.append(channel)
        
        # 애니메이션 객체 생성
        if channels:
            animation = pygltflib.Animation(
                name=f"Animation_{animation_idx}",
                channels=channels,
                samplers=samplers
            )
            gltf.animations = [animation]  # 각 파일에 한 개의 애니메이션만 포함
            
            # 최종 버퍼 길이 설정
            gltf.buffers[0].byteLength = len(binary_blob)
            gltf.set_binary_blob(binary_blob)
            
            # 파일명 생성 및 저장
            final_path = f"{base_name}_animation_{animation_idx}.glb"
            gltf.save(final_path)
            output_glbs.append(final_path)
            
            print(f"애니메이션 {animation_idx} 저장 완료: {final_path} ({len(channels)} 채널)")
            
            # 다음 애니메이션을 위해 초기화
            gltf.animations = []
        
    return output_glbs

def create_improved_glb_animation(motion_data, output_path=None, fps=30, target_bones=None):
    """
    SMPL 모션 데이터를 기반으로 GLB 애니메이션 생성
    create_direct_glb_animation의 구조를 기본으로 하고 선택한 본만 애니메이션 데이터를 적용
    
    Args:
        motion_data: (N, 22, 3, F) 형태의 SMPL 모션 데이터
        output_path: 출력 파일 경로
        fps: 애니메이션 프레임 레이트
        target_bones: 교체할 본 인덱스 목록 (None이면 모든 본 교체)
    """
    # 출력 경로 설정
    if output_path is None:
        try:
            static_dir = Path(__file__).parent / "static"
            static_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(static_dir / "result_static.glb")
        except Exception as e:
            print(f"기본 경로 설정 실패: {e}")
            # 대체 경로 사용
            import tempfile
            output_path = os.path.join(tempfile.gettempdir(), "improved_animation.glb")
    
    # 첫 번째 애니메이션만 사용
    current_motion = motion_data[0]  # (22, 3, F)
    frame_count = current_motion.shape[2]
    animation_data = np.transpose(current_motion, (2, 0, 1))  # (F, 22, 3)
    
    # 시간 데이터 생성
    times = np.array([i * (1.0 / fps) for i in range(frame_count)], dtype=np.float32)
    
    # GLB 모델 생성
    gltf = pygltflib.GLTF2()
    
    # 씬 설정
    gltf.scene = 0
    gltf.scenes = [pygltflib.Scene(nodes=[0])]
    
    # 기본 본 설정은 그대로 유지
    # 부모-자식 관계 정의는 그대로 유지
    # 관절 위치 (T-포즈)는 그대로 유지
    # 노드 생성도 그대로 유지
    
    # 직접 GLB 생성 코드를 재사용하되, 애니메이션 부분만 수정
    base_gltf = create_direct_glb_animation()
    base_gltf = base_gltf["gltf"]
    # 노드 구조 복사
    gltf.nodes = base_gltf.nodes
    
    # 빈 버퍼 생성
    gltf.buffers = [pygltflib.Buffer(byteLength=0)]
    binary_blob = bytearray()
    
    # 시간 데이터 추가
    time_data_bytes = times.tobytes()
    time_byte_offset = len(binary_blob)
    binary_blob.extend(time_data_bytes)
    
    time_buffer_view = pygltflib.BufferView(
        buffer=0,
        byteOffset=time_byte_offset,
        byteLength=len(time_data_bytes)
    )
    gltf.bufferViews = [time_buffer_view]
    time_buffer_view_index = 0
    
    time_accessor = pygltflib.Accessor(
        bufferView=time_buffer_view_index,
        componentType=pygltflib.FLOAT,
        count=frame_count,
        type=pygltflib.SCALAR,
        max=[float(times.max())],
        min=[float(times.min())]
    )
    gltf.accessors = [time_accessor]
    time_accessor_index = 0
    
    # 애니메이션 데이터 생성
    samplers = []
    channels = []
    
    # 기존 애니메이션 정보 복사 (타겟 본이 아닌 경우)
    if base_gltf.animations:
        base_animation = base_gltf.animations[0]
        
        # 기존 시간 데이터 접근자 가져오기
        base_time_accessor = base_gltf.accessors[base_animation.samplers[0].input]
        base_frame_count = base_time_accessor.count
        
        # 변경할 본 목록이 없으면 모든 본 변경
        if target_bones is None:
            target_bones = list(range(min(22, animation_data.shape[1])))
        
        # 기존 채널과 샘플러 복사 (변경할 본이 아닌 경우)
        for i, channel in enumerate(base_animation.channels):
            node_idx = channel.target.node
            
            # 변경할 본이 아니면 기존 데이터 복사
            if node_idx not in target_bones:
                base_sampler = base_animation.samplers[channel.sampler]
                
                # 기존 회전 데이터 접근자
                base_rot_accessor = base_gltf.accessors[base_sampler.output]
                base_rot_buffer_view = base_gltf.bufferViews[base_rot_accessor.bufferView]
                
                # 기존 바이너리 데이터 가져오기
                base_binary = base_gltf.get_data_from_buffer_uri(base_gltf.buffers[0].uri)
                base_rot_data = base_binary[
                    base_rot_buffer_view.byteOffset:
                    base_rot_buffer_view.byteOffset + base_rot_buffer_view.byteLength
                ]
                
                # 복사하여 새 버퍼에 추가
                rot_byte_offset = len(binary_blob)
                binary_blob.extend(base_rot_data)
                
                # 버퍼 뷰 추가
                rot_buffer_view = pygltflib.BufferView(
                    buffer=0,
                    byteOffset=rot_byte_offset,
                    byteLength=len(base_rot_data)
                )
                gltf.bufferViews.append(rot_buffer_view)
                rot_buffer_view_index = len(gltf.bufferViews) - 1
                
                # 접근자 추가
                rot_accessor = pygltflib.Accessor(
                    bufferView=rot_buffer_view_index,
                    componentType=base_rot_accessor.componentType,
                    count=base_rot_accessor.count,
                    type=base_rot_accessor.type,
                    max=base_rot_accessor.max,
                    min=base_rot_accessor.min
                )
                gltf.accessors.append(rot_accessor)
                rot_accessor_index = len(gltf.accessors) - 1
                
                # 샘플러 추가
                sampler = pygltflib.AnimationSampler(
                    input=time_accessor_index,
                    output=rot_accessor_index,
                    interpolation=base_sampler.interpolation
                )
                samplers.append(sampler)
                sampler_index = len(samplers) - 1
                
                # 채널 추가
                new_channel = pygltflib.AnimationChannel(
                    sampler=sampler_index,
                    target=pygltflib.AnimationChannelTarget(
                        node=node_idx,
                        path=channel.target.path
                    )
                )
                channels.append(new_channel)
                print(f"기존 본 애니메이션 복사: {bone_names[node_idx]}")
    
    # 새 애니메이션 데이터 처리 (타겟 본만)
    for joint_idx in target_bones:
        if joint_idx >= len(bone_names):
            continue

        # 관절 이름
        joint_name = bone_names[joint_idx]
        
        # 관절의 회전 데이터 추출
        joint_rotations = animation_data[:, joint_idx, :]  # (F, 3)
        
        # 좌표계 변환 적용
        joint_rotations = fix_rotation_for_joint_type(joint_rotations, joint_idx)
        
        # NaN 값 처리
        if np.isnan(joint_rotations).any():
            joint_rotations = np.nan_to_num(joint_rotations)
        
        try:
            # 쿼터니언 변환
            quaternions = R.from_rotvec(joint_rotations).as_quat()  # (F, 4) [x, y, z, w]
            
            # NaN 값 재확인
            if np.isnan(quaternions).any():
                identity_quat = np.array([0.0, 0.0, 0.0, 1.0])
                nan_indices = np.isnan(quaternions).any(axis=1)
                quaternions[nan_indices] = identity_quat
                
        except Exception as e:
            print(f"쿼터니언 변환 오류 (관절 {joint_idx}): {e}")
            # 오류 발생 시 항등 쿼터니언으로 설정
            quaternions = np.array([[0.0, 0.0, 0.0, 1.0]] * frame_count)

        # 쿼터니언 데이터를 이진 형식으로 변환
        quat_data = quaternions.astype(np.float32)
        quat_bytes = quat_data.tobytes()
        quat_byte_offset = len(binary_blob)
        binary_blob.extend(quat_bytes)
        
        # 버퍼 뷰 추가
        quat_buffer_view = pygltflib.BufferView(
            buffer=0,
            byteOffset=quat_byte_offset,
            byteLength=len(quat_bytes)
        )
        gltf.bufferViews.append(quat_buffer_view)
        quat_buffer_view_index = len(gltf.bufferViews) - 1
        
        # 접근자 추가
        quat_min = np.min(quaternions, axis=0).astype(float).tolist()
        quat_max = np.max(quaternions, axis=0).astype(float).tolist()
        
        quat_accessor = pygltflib.Accessor(
            bufferView=quat_buffer_view_index,
            componentType=pygltflib.FLOAT,
            count=frame_count,
            type=pygltflib.VEC4,
            max=quat_max,
            min=quat_min
        )
        gltf.accessors.append(quat_accessor)
        quat_accessor_index = len(gltf.accessors) - 1
        
        # 샘플러 및 채널 생성
        sampler = pygltflib.AnimationSampler(
            input=time_accessor_index,
            output=quat_accessor_index,
            interpolation="LINEAR"
        )
        samplers.append(sampler)
        sampler_index = len(samplers) - 1
        
        channel = pygltflib.AnimationChannel(
            sampler=sampler_index,
            target=pygltflib.AnimationChannelTarget(
                node=joint_idx,
                path="rotation"
            )
        )
        channels.append(channel)
        
        print(f"새 애니메이션 추가: {joint_name} (인덱스: {joint_idx})")
    
    # 애니메이션 객체 생성
    animation = pygltflib.Animation(
        name="MixedAnimation",
        channels=channels,
        samplers=samplers
    )
    gltf.animations = [animation]

    # 최종 버퍼 길이 설정
    gltf.buffers[0].byteLength = len(binary_blob)
    gltf.set_binary_blob(binary_blob)

    # # 최종 버퍼 길이 설정
    # gltf.buffers[0].byteLength = len(binary_blob)
    # gltf.set_binary_blob(binary_blob)
    # # 저장
    # gltf.save(output_path)


    output_path = f'{output_path}/improved_animation.glb' if output_path else os.path.join(os.getcwd(), "improved_animation.glb")
    # 저장
    print(f"혼합 애니메이션 생성 완료: {output_path}")


   # 애니메이션 객체 생성 직전에 추가
    print("\n===== 노드 21 상세 정보 (1) =====")
    node21 = gltf.nodes[21]
    print(f"이름: {node21.name}")
    print(f"위치: {node21.translation}")
    print(f"회전: {node21.rotation}")
    print(f"부모: 노드 19 (RightForeArm)")

    # # 노드 21에 대한 명시적인 초기화 추가
    # node21.name = "mixamorig:RightHand"  # 이름 확인
    # if not node21.translation or any(np.isnan(node21.translation)):
    #     node21.translation = [0.9, 0.4, 0.0]  # 위치 재설정
    # if not node21.rotation or any(np.isnan(node21.rotation)):
    #     node21.rotation = [0.0, 0.0, 0.0, 1.0]  # 기본 회전

    # # 부모-자식 관계 명시적 확인
    # right_forearm = gltf.nodes[19]  # 오른쪽 팔뚝
    # if not hasattr(right_forearm, "children") or not right_forearm.children:
    #     right_forearm.children = [21]
    # elif 21 not in right_forearm.children:
    #     right_forearm.children.append(21)

    import json
    # 저장하기 직전에 추가
    nodes_json = json.dumps([{k: v for k, v in vars(node).items() 
                        if not k.startswith('_') and v is not None} 
                        for node in gltf.nodes], indent=2)
    print(f"저장 직전 노드 21 JSON:\n{nodes_json[21]}")


    # 노드 21에 matrix 속성이 있는지 확인하고 제거
    if hasattr(node21, 'matrix') and node21.matrix is not None:
        print(f"노드 21에 matrix 속성 발견: {node21.matrix}")
        node21.matrix = None

    print("노드 21 데이터 확인 및 수정 완료")
    print("============================\n")

    gltf.nodes = base_gltf.nodes
    gltf.save(output_path)
    base_gltf.save(f"{os.path.dirname(output_path)}/_base{os.path.basename(output_path)}") # 정상
    print(f"- 총 {len(channels)}개 채널 중 {len(target_bones)}개 새로 추가됨")

    # 저장 후 파일에서 다시 로드하여 확인
    saved_gltf = pygltflib.GLTF2().load(output_path)
    saved_nodes_json = json.dumps([{k: v for k, v in vars(node).items() 
                                if not k.startswith('_') and v is not None} 
                            for node in saved_gltf.nodes], indent=2)
    print(f"저장 후 노드 21 JSON:\n{saved_gltf.nodes[21]}")
    print(f'저장 후 전체 노드 수: {saved_gltf.nodes}')
    
    return output_path

def create_direct_glb_animation(save_file=False, output_path=None):
    """
    GLB 포맷에 맞게 직접 허리 숙임 애니메이션을 생성
    SMPL 변환 데이터와 비교를 위한 함수
    
    Args:
        save_file: 파일로 저장할지 여부 (기본값: False)
        output_path: 저장할 경로 (save_file이 True인 경우에만 사용)
    
    Returns:
        dict: 생성된 애니메이션 데이터 (gltf 객체, 바이너리 데이터, 채널 정보 등)
    """
    import pygltflib
    import numpy as np
    import os
    from scipy.spatial.transform import Rotation as R
    
    # 결과 파일 경로 생성 (저장하는 경우에만 사용)
    if save_file and output_path is None:
        static_dir = os.path.join(os.getcwd(), 'static')
        if not os.path.exists(static_dir):
            os.makedirs(static_dir)
        output_path = os.path.join(static_dir, "direct_glb_spine_bend.glb")
    
    # 애니메이션 프레임 수와 FPS
    frames = 60
    fps = 30
    
    # 시간 데이터 생성
    times = np.array([i * (1.0 / fps) for i in range(frames)], dtype=np.float32)
    
    # GLB 모델 생성
    gltf = pygltflib.GLTF2()
    
    # 씬 설정
    gltf.scene = 0
    gltf.scenes = [pygltflib.Scene(nodes=[0])]

    # 노드 생성 (관절)
    nodes = []
    for i, name in enumerate(bone_names):
        # 위치 설정
        translation = joint_positions[i]
        
        # 기본 회전
        rotation = [0.0, 0.0, 0.0, 1.0]  # 단위 쿼터니언
        
        # # 특수 관절 초기 회전 설정
        # if i in [16, 17]:  # 팔 관절
        #     side = -1 if i == 16 else 1
        #     # 팔이 옆으로 펴지도록 회전 (T자 포즈)
        #     rot = R.from_euler('z', side * 90, degrees=True).as_quat()
        #     rotation = [float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])]
            
        node = pygltflib.Node(
            name=name,
            translation=[float(t) for t in translation],
            rotation=rotation,
        )
        
        # 부모-자식 관계 설정
        if i > 0:
            parent_idx = hierarchy[i]
            if not hasattr(nodes[parent_idx], 'children'):
                nodes[parent_idx].children = []
            nodes[parent_idx].children.append(i)

        nodes.append(node)
    
    gltf.nodes = nodes

   
    # 빈 버퍼 생성
    gltf.buffers = [pygltflib.Buffer(byteLength=0)]
    binary_blob = bytearray()
    
    # 시간 데이터 추가
    time_data_bytes = times.tobytes()
    time_byte_offset = len(binary_blob)
    binary_blob.extend(time_data_bytes)
    
    time_buffer_view = pygltflib.BufferView(
        buffer=0,
        byteOffset=time_byte_offset,
        byteLength=len(time_data_bytes)
    )
    gltf.bufferViews = [time_buffer_view]
    time_buffer_view_index = 0
    
    time_accessor = pygltflib.Accessor(
        bufferView=time_buffer_view_index,
        componentType=pygltflib.FLOAT,
        count=frames,
        type=pygltflib.SCALAR,
        max=[float(times.max())],
        min=[float(times.min())]
    )
    gltf.accessors = [time_accessor]
    time_accessor_index = 0
    
    # 애니메이션 데이터 생성
    samplers = []
    channels = []
    
    # 척추 관절 애니메이션 (3, 6, 11번 관절)
    spine_indices = [1]  # Spine, Spine1, Spine2
    spine_weights = [0.3, 0.5, 0.7]  # 각 척추 관절의 회전 가중치
    
    # 각 관절별 애니메이션 데이터를 저장할 리스트
    joint_animation_data = {}
    
    for i, joint_idx in enumerate(spine_indices):
        # 관절별 쿼터니언 데이터 생성
        quaternions = np.zeros((frames, 4))
        
        for f in range(frames):
            # sin 곡선을 사용한 자연스러운 굽힘
            cycle_progress = f / frames
            bend_angle = np.sin(cycle_progress * 2 * np.pi) * (45 * spine_weights[i])  # 최대 가중치에 따른 각도
            
            # X축 회전 (앞으로 숙임)
            rot = R.from_euler('x', bend_angle, degrees=True).as_quat()
            quaternions[f] = rot
            
        # 애니메이션 데이터 저장
        joint_animation_data[joint_idx] = quaternions
        
        # 쿼터니언 데이터를 이진 형식으로 변환
        quat_data = quaternions.astype(np.float32)
        quat_bytes = quat_data.tobytes()
        quat_byte_offset = len(binary_blob)
        binary_blob.extend(quat_bytes)
        
        # 버퍼 뷰 추가
        quat_buffer_view = pygltflib.BufferView(
            buffer=0,
            byteOffset=quat_byte_offset,
            byteLength=len(quat_bytes)
        )
        gltf.bufferViews.append(quat_buffer_view)
        quat_buffer_view_index = len(gltf.bufferViews) - 1
        
        # 접근자 추가
        quat_min = np.min(quaternions, axis=0).astype(float).tolist()
        quat_max = np.max(quaternions, axis=0).astype(float).tolist()
        
        quat_accessor = pygltflib.Accessor(
            bufferView=quat_buffer_view_index,
            componentType=pygltflib.FLOAT,
            count=frames,
            type=pygltflib.VEC4,
            max=quat_max,
            min=quat_min
        )
        gltf.accessors.append(quat_accessor)
        quat_accessor_index = len(gltf.accessors) - 1
        
        # 샘플러 및 채널 생성
        sampler = pygltflib.AnimationSampler(
            input=time_accessor_index,
            output=quat_accessor_index,
            interpolation="LINEAR"
        )
        samplers.append(sampler)
        sampler_index = len(samplers) - 1
        
        channel = pygltflib.AnimationChannel(
            sampler=sampler_index,
            target=pygltflib.AnimationChannelTarget(
                node=joint_idx,
                path="rotation"
            )
        )
        channels.append(channel)
        
    # 애니메이션 객체 생성
    animation = pygltflib.Animation(
        name="SpineBendAnimation",
        channels=channels,
        samplers=samplers
    )
    gltf.animations = [animation]
    
    # 최종 버퍼 길이 설정
    gltf.buffers[0].byteLength = len(binary_blob)
    gltf.set_binary_blob(binary_blob)
    
    # 필요한 경우 파일로 저장
    if save_file:
        gltf.save(output_path)
        print(f"직접 GLB 애니메이션 생성 완료: {output_path}")
    
    # 애니메이션 데이터를 반환
    return {
        'gltf': gltf,                   # GLB 객체
        'binary_blob': binary_blob,     # 바이너리 데이터
        'times': times,                 # 시간 데이터
        'fps': fps,                     # 프레임 레이트
        'frames': frames,               # 프레임 수
        'bone_names': bone_names,       # 본 이름
        'hierarchy': hierarchy,         # 계층 구조
        'positions': joint_positions,   # 관절 위치
        'channels': channels,           # 애니메이션 채널
        'samplers': samplers,           # 애니메이션 샘플러
        'animation': animation,         # 애니메이션 객체
        'joint_animation_data': joint_animation_data  # 관절별 애니메이션 데이터
    }


