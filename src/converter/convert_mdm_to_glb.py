import pygltflib
import numpy as np
import os
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

# 절대 위치(global)를 상대 위치(local)로 변환하는 함수

# global_to_local_positions 함수 수정
def global_to_local_positions(global_positions, hierarchy, correct_axes=True):
    """
    전역(절대) 위치를 지역(상대) 위치로 변환하고 좌표계를 보정
    
    Args:
        global_positions: (F, J, 3) 형태의 전역 위치 데이터
        hierarchy: 관절 계층 구조 (자식 인덱스 -> 부모 인덱스)
        correct_axes: 좌표축 보정 여부
        
    Returns:
        local_positions: (F, J, 3) 형태의 지역 위치 데이터
    """
    frames, joints, _ = global_positions.shape
    
    # 좌표축 보정 없이 먼저 상대 위치 계산
    local_positions = np.zeros_like(global_positions)
    
    # 각 프레임에 대해
    for f in range(frames):
        # 각 관절에 대해
        for j in range(joints):
            if j == 0 or j not in hierarchy:  # 루트 노드는 절대 위치 그대로 사용
                local_positions[f, j] = global_positions[f, j]
            else:
                parent_idx = hierarchy[j]
                # 부모 기준 상대 위치 계산 (자식 위치 - 부모 위치)
                local_positions[f, j] = global_positions[f, j] - global_positions[f, parent_idx]
    
    return local_positions


def create_improved_glb_animation(motion_data, output_path=None, fps=30, target_bones=None):
    """
    모션 데이터(절대 위치 좌표)를 기반으로 GLB 애니메이션 생성
    추가로 모든 자식 본들에 대해서도 보간 애니메이션 생성
    
    Args:
        motion_data: (F, 22, 3) 또는 (22, 3, F) 형태의 본 위치 데이터
        output_path: 출력 파일 경로 (None이면 기본 경로 사용)
        fps: 애니메이션 프레임 레이트
        target_bones: 사용할 본 인덱스 목록 (None이면 모든 본 사용)
    
    Returns:
        output_path: 생성된 GLB 파일 경로
    """
    import os

    # 1. 데이터 형태 정규화 (F, 22, 3) 형태로 변환
    print(f"\n===== GLB 애니메이션 생성 시작 =====")
    print(f"입력 데이터 형태: {motion_data.shape}, 차원: {motion_data.ndim}")
    
    if motion_data.ndim == 4:  # (N, 22, 3, F) - 배치 형태
        print(f"배치 데이터에서 첫 번째 애니메이션만 사용")
        # 첫 번째 애니메이션 추출 후 (22, 3, F) 형태
        anim_data = motion_data[0]
        # (22, 3, F) -> (F, 22, 3)으로 변환
        anim_data = np.transpose(anim_data, (2, 0, 1))
    elif motion_data.ndim == 3:
        if motion_data.shape[0] == 22 and motion_data.shape[1] == 3:
            # (22, 3, F) -> (F, 22, 3)으로 변환
            print(f"(22, 3, F) 형태에서 (F, 22, 3) 형태로 변환")
            anim_data = np.transpose(motion_data, (2, 0, 1))
        elif motion_data.shape[1] == 22 and motion_data.shape[2] == 3:
            # 이미 (F, 22, 3) 형태
            print(f"이미 (F, 22, 3) 형태")
            anim_data = motion_data
        else:
            raise ValueError(f"지원되지 않는 데이터 형태: {motion_data.shape}")
    else:
        raise ValueError(f"지원되지 않는 데이터 차원: {motion_data.ndim}")

    # 2. 스케일 조정 - 필요한 경우 조정
    scale_factor = 1  # 스케일 인자 조정
    anim_data = anim_data * scale_factor
    
    # 3. 절대 위치를 상대 위치로 변환
    print("절대 위치를 상대 위치로 변환 중...")
    local_anim_data = global_to_local_positions(anim_data, hierarchy, correct_axes=True)
    
    # 4. 출력 경로 설정
    if output_path is None:
        output_path = str(Path(__file__).parent / '../static/improved_animation.glb')
        print(f"기본 출력 경로 사용: {output_path}")
    
    # 5. 프레임 수와 본 수 확인
    num_frames = local_anim_data.shape[0]
    num_joints = min(local_anim_data.shape[1], 22)  # 최대 22개 본만 사용
    print(f"프레임 수: {num_frames}, 본 수: {num_joints}")
    
    # 6. 대상 본 설정
    if target_bones is None:
        target_bones = list(range(num_joints))
    else:
        # 범위 확인
        target_bones = [i for i in target_bones if 0 <= i < num_joints]
    
    print(f"애니메이션 적용 본: {len(target_bones)}개")
    
    try:
        # 7. 베이스 GLB 템플릿 로드 - 기존 골격 구조 유지를 위해
        base_glb_path = Path(__file__).parent / '../static/models/_baseimproved_animation.glb'
        print(f"기본 GLB 템플릿 로드 중: {base_glb_path}")
        if os.path.exists(base_glb_path):
            base_gltf = pygltflib.GLTF2().load(str(base_glb_path))
            print(f"기본 GLB 템플릿 로드 성공: {len(base_gltf.nodes)}개 노드")
            
            # 본 매핑 구성
            bone_name_to_idx = {}
            for i, node in enumerate(base_gltf.nodes):
                if hasattr(node, 'name') and node.name:
                    bone_name_to_idx[node.name] = i
                    
            # 기본 본 22개에 대한 인덱스 매핑
            bone_idx_mapping = {}
            for i, name in enumerate(bone_names):
                if i < num_joints and name in bone_name_to_idx:
                    bone_idx_mapping[i] = bone_name_to_idx[name]
            
            # 초기 GLB 구조 복사            
            gltf = pygltflib.GLTF2()
            gltf.scene = base_gltf.scene
            gltf.scenes = base_gltf.scenes
            gltf.nodes = base_gltf.nodes
            
            # 애니메이션이 없는 경우 기본 구조 생성
            if not hasattr(gltf, 'animations') or not gltf.animations:
                gltf.animations = []
            
            print(f"기본 GLB 템플릿에서 {len(bone_idx_mapping)}개 본 매핑됨")
            
            # 부모-자식 관계 분석
            node_hierarchy = {}  # 자식 -> 부모 인덱스
            for i, node in enumerate(gltf.nodes):
                if hasattr(node, 'children'):
                    for child_idx in node.children:
                        node_hierarchy[child_idx] = i
            
            # 본 그룹 분석 (각 기본 본과 그 자식들)
            bone_groups = {}  # 기본 본 인덱스 -> [자식 본 인덱스 리스트]
            for i in range(num_joints):
                if i in bone_idx_mapping:
                    base_bone_idx = bone_idx_mapping[i]
                    # 기본 본과 그 모든 자식 본들 찾기
                    bone_groups[i] = find_all_children(gltf, base_bone_idx)
                    print(f"본 {i} ({bone_names[i]}): {len(bone_groups[i])}개 자식 본 포함")
        else:
            print(f"기본 GLB 템플릿 파일이 없습니다. 기본 구조로 생성합니다.")
            # 기본 구조 생성 - 파일이 없는 경우
            gltf = pygltflib.GLTF2()
            gltf.scene = 0
            gltf.scenes = [pygltflib.Scene(nodes=[0], extensions={}, extras={})]
            
            # 노드(본) 생성
            nodes = []
            for i in range(num_joints):
                translation = local_anim_data[0, i].tolist()
                node = pygltflib.Node(
                    name=bone_names[i] if i < len(bone_names) else f"joint_{i}",
                    translation=[float(t) for t in translation],
                    rotation=[0.0, 0.0, 0.0, 1.0],
                    extensions={},
                    extras={}
                )
                nodes.append(node)
            
            # 계층 구조 설정
            for i in range(1, num_joints):
                if i in hierarchy:
                    parent_idx = hierarchy[i]
                    if 0 <= parent_idx < len(nodes):
                        if not hasattr(nodes[parent_idx], 'children'):
                            nodes[parent_idx].children = []
                        nodes[parent_idx].children.append(i)
            
            gltf.nodes = nodes
            bone_idx_mapping = {i: i for i in range(num_joints)}
            bone_groups = {i: [i] for i in range(num_joints)}  # 기본 본만 포함
            
        # 8. 애니메이션 데이터 설정
        binary_blob = bytearray()
        
        # 9. 시간 데이터 생성 및 추가
        times = np.arange(num_frames, dtype=np.float32) / fps
        time_bytes = times.tobytes()
        time_offset = 0
        binary_blob.extend(time_bytes)
        
        # 10. bufferView 및 accessor 생성 (시간)
        gltf.bufferViews = [
            pygltflib.BufferView(
                buffer=0,
                byteOffset=time_offset,
                byteLength=len(time_bytes),
                extensions={},
                extras={}
            )
        ]
        
        gltf.accessors = [
            pygltflib.Accessor(
                bufferView=0,
                componentType=pygltflib.FLOAT,
                count=num_frames,
                type=pygltflib.SCALAR,
                max=[float(times.max())],
                min=[float(times.min())],
                extensions={},
                extras={}
            )
        ]
        
        # 11. 애니메이션 객체 생성
        animation = pygltflib.Animation(
            name="InterpolatedAnimation",
            channels=[],
            samplers=[],
            extensions={},
            extras={}
        )
        
        # 12. 각 본에 대한 translation 채널 생성 (기본 22개 본 + 보간된 자식 본들)
        time_accessor_idx = 0
        processed_nodes = set()  # 이미 처리된 노드 추적
        
        print(f"애니메이션 채널 생성 중...")
        for joint_idx in target_bones:
            if joint_idx not in bone_idx_mapping:
                print(f"경고: 본 인덱스 {joint_idx}에 대한 매핑이 없습니다.")
                continue
                
            # 기본 본의 위치 데이터 추출 (F, 3) - 상대 위치
            positions = local_anim_data[:, joint_idx, :].astype(np.float32)
            
            # 기본 본 노드 인덱스
            base_node_idx = bone_idx_mapping[joint_idx]
            processed_nodes.add(base_node_idx)
            
            # 기본 본의 애니메이션 채널 생성
            pos_acc_idx = create_translation_channel(
                gltf, animation, positions, base_node_idx, time_accessor_idx, binary_blob
            )
            
            # 관련 자식 본 처리 (보간)
            if joint_idx in bone_groups:
                children = bone_groups[joint_idx]
                for child_idx in children:
                    if child_idx != base_node_idx and child_idx not in processed_nodes:
                        processed_nodes.add(child_idx)
                        
                        # 부모 본의 움직임을 자식에도 적용
                        child_node = gltf.nodes[child_idx]
                        
                        # 애니메이션 변위 보간 계산을 수정
                        child_base_pos = np.array(child_node.translation) if hasattr(child_node, 'translation') else np.zeros(3)
                        parent_base_pos = np.array(gltf.nodes[base_node_idx].translation) if hasattr(gltf.nodes[base_node_idx], 'translation') else np.zeros(3)
                        
                        # 각 프레임별로 순차적인 변위 적용 (프레임 간 차이를 누적)
                        child_positions = np.zeros((num_frames, 3), dtype=np.float32)
                        child_positions[0] = child_base_pos
                        
                        # 프레임별 변위 계산
                        for f in range(1, num_frames):
                            # 부모 본의 이전 프레임과 현재 프레임의 변위 차이
                            parent_movement = positions[f] - positions[f-1]
                            # 이전 프레임의 자식 위치에 부모 본의 변위를 더함
                            child_positions[f] = child_positions[f-1] + parent_movement
                        
                        # 자식 본 애니메이션 채널 생성
                        create_translation_channel(
                            gltf, animation, child_positions, child_idx, time_accessor_idx, binary_blob
                        )
        
        # 채널 개수 확인 및 출력
        print(f"생성된 애니메이션 채널: {len(animation.channels)}개")
        
        # 13. 애니메이션 추가
        gltf.animations = [animation]
        
        # 14. 버퍼 설정
        buffer = pygltflib.Buffer(
            byteLength=len(binary_blob),
            extensions={},
            extras={}
        )
        gltf.buffers = [buffer]
        
        # 15. Extensions 필드 초기화 (오류 방지)
        fix_extensions_extras(gltf)
        
        # 16. 바이너리 데이터 설정
        gltf.set_binary_blob(bytes(binary_blob))
        
        # 17. 파일 저장
        print(f"GLB 파일 저장: {output_path}")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        gltf.save(output_path)
        
        print(f"===== GLB 애니메이션 생성 완료 =====")
        print(f"애니메이션: {len(animation.channels)}개 채널, {num_frames}프레임, {fps}fps")
        
        return output_path
    except Exception as e:
        import traceback
        print(f"GLB 애니메이션 생성 중 오류 발생: {str(e)}")
        traceback.print_exc()
        return None

def find_all_children(gltf, node_idx, visited=None):
    """
    주어진 노드의 모든 자식 노드(재귀적)를 찾아 리스트로 반환
    
    Args:
        gltf: GLTF2 객체
        node_idx: 시작 노드 인덱스
        visited: 이미 방문한 노드 집합 (순환 참조 방지)
        
    Returns:
        list: 모든 자식 노드 인덱스 리스트 (자기 자신 포함)
    """
    if visited is None:
        visited = set()
        
    if node_idx in visited:
        return []
        
    visited.add(node_idx)
    result = [node_idx]
    
    node = gltf.nodes[node_idx]
    if hasattr(node, 'children'):
        for child_idx in node.children:
            result.extend(find_all_children(gltf, child_idx, visited))
            
    return result

def create_translation_channel(gltf, animation, positions, node_idx, time_accessor_idx, binary_blob):
    """
    위치 애니메이션 채널 생성
    
    Args:
        gltf: GLTF2 객체
        animation: Animation 객체
        positions: (F, 3) 형태의 위치 데이터
        node_idx: 대상 노드 인덱스
        time_accessor_idx: 시간 accessor 인덱스
        binary_blob: 바이너리 데이터
        
    Returns:
        int: 생성된 위치 데이터 accessor 인덱스
    """
    # 이진 데이터로 변환 및 추가
    pos_bytes = positions.tobytes()
    pos_offset = len(binary_blob)
    binary_blob.extend(pos_bytes)
    
    # bufferView 생성
    pos_bv = pygltflib.BufferView(
        buffer=0,
        byteOffset=pos_offset,
        byteLength=len(pos_bytes),
        extensions={},
        extras={}
    )
    gltf.bufferViews.append(pos_bv)
    pos_bv_idx = len(gltf.bufferViews) - 1
    
    # 위치 데이터의 최소/최대값
    pos_min = np.min(positions, axis=0).tolist()
    pos_max = np.max(positions, axis=0).tolist()
    
    # accessor 생성
    pos_acc = pygltflib.Accessor(
        bufferView=pos_bv_idx,
        componentType=pygltflib.FLOAT,
        count=len(positions),
        type=pygltflib.VEC3,
        max=pos_max,
        min=pos_min,
        extensions={},
        extras={}
    )
    gltf.accessors.append(pos_acc)
    pos_acc_idx = len(gltf.accessors) - 1
    
    # 샘플러 생성
    sampler = pygltflib.AnimationSampler(
        input=time_accessor_idx,
        output=pos_acc_idx,
        interpolation="LINEAR",
        extensions={},
        extras={}
    )
    animation.samplers.append(sampler)
    sampler_idx = len(animation.samplers) - 1
    
    # 채널 생성
    channel = pygltflib.AnimationChannel(
        sampler=sampler_idx,
        target=pygltflib.AnimationChannelTarget(
            node=node_idx,
            path="translation",
            extensions={},
            extras={}
        ),
        extensions={},
        extras={}
    )
    animation.channels.append(channel)
    
    return pos_acc_idx

def fix_extensions_extras(obj):
    """
    None 값을 가진 extensions와 extras를 빈 딕셔너리로 변경
    
    Args:
        obj: GLTF 객체 또는 그 속성
    """
    if isinstance(obj, list):
        for item in obj:
            fix_extensions_extras(item)
    elif hasattr(obj, "__dict__"):
        for k, v in obj.__dict__.items():
            if k in ("extensions", "extras") and v is None:
                setattr(obj, k, {})
            elif isinstance(v, (list, tuple)) and v:
                for item in v:
                    fix_extensions_extras(item)
            elif hasattr(v, "__dict__"):
                fix_extensions_extras(v)

def apply_animation_to_skin(skin_model_path, motion_data, output_path=None, fps=30):
    """
    미리 로딩된 스킨 모델(GLB)에 모션 데이터를 적용하여 애니메이션 GLB 생성
    
    Args:
        skin_model_path: 스킨 모델 GLB 파일 경로
        motion_data: (F, 22, 3) 또는 (22, 3, F) 형태의 본 위치 데이터
        output_path: 출력 파일 경로 (None이면 임시 파일 생성)
        fps: 애니메이션 프레임 레이트
    
    Returns:
        output_path: 생성된 GLB 파일 경로
    """
    import os
    import shutil
    import traceback
    import numpy as np
    import uuid
    from pathlib import Path
    
    print(f"\n===== 스킨 모델에 애니메이션 적용 시작 =====")
    print(f"스킨 모델: {skin_model_path}")
    
    try:
        # 1. 데이터 형태 정규화
        if motion_data.ndim == 4:  # (N, 22, 3, F) 배치 형태
            anim_data = motion_data[0]
            anim_data = np.transpose(anim_data, (2, 0, 1))
        elif motion_data.ndim == 3:
            if motion_data.shape[0] == 22 and motion_data.shape[1] == 3:
                anim_data = np.transpose(motion_data, (2, 0, 1))
            elif motion_data.shape[1] == 22 and motion_data.shape[2] == 3:
                anim_data = motion_data
            else:
                raise ValueError(f"지원되지 않는 데이터 형태: {motion_data.shape}")
        else:
            raise ValueError(f"지원되지 않는 데이터 차원: {motion_data.ndim}")
        
        print(f"정규화된 데이터 형태: {anim_data.shape}")
        
        # 2. 출력 경로 설정
        if output_path is None:
            unique_id = str(uuid.uuid4())[:8]
            output_path = str(Path(__file__).parent / f'../static/models/anim_{unique_id}.glb')
        
        # 3. 임시 작업 디렉토리 설정
        temp_dir = Path(output_path).parent / f"temp_{uuid.uuid4().hex[:8]}"
        os.makedirs(temp_dir, exist_ok=True)
        
        # 4. 원본 스킨 모델 파일을 복사 (기존 구조 보존)
        temp_model = temp_dir / "skin_model.glb"
        shutil.copy2(skin_model_path, temp_model)
        
        # 5. 원본 스킨 모델 로드
        gltf = pygltflib.GLTF2().load(str(temp_model))
        print(f"스킨 모델 로드 완료: {len(gltf.nodes)}개 노드")
        
        # 6. 본 매핑 설정
        bone_name_to_idx = {}
        for i, node in enumerate(gltf.nodes):
            if hasattr(node, 'name') and node.name:
                bone_name_to_idx[node.name] = i
        
        bone_idx_mapping = {}
        for i, name in enumerate(bone_names):
            if i < len(bone_names) and name in bone_name_to_idx:
                bone_idx_mapping[i] = bone_name_to_idx[name]
        
        print(f"매핑된 본: {len(bone_idx_mapping)}개")
        if len(bone_idx_mapping) == 0:
            print("경고: 매핑된 본이 없습니다. 스킨 모델과 본 이름이 일치하지 않을 수 있습니다.")
            # 본 이름 목록 표시
            print("스킨 모델의 본 이름:")
            for i, node in enumerate(gltf.nodes):
                if hasattr(node, 'name') and node.name:
                    print(f"  {i}: {node.name}")
        
        # 7. 절대 위치를 상대 위치로 변환
        local_anim_data = global_to_local_positions(anim_data, hierarchy)
        
        # 8. 프레임 수와 본 수 확인
        num_frames = local_anim_data.shape[0]
        num_joints = min(local_anim_data.shape[1], 22)
        print(f"프레임 수: {num_frames}, 본 수: {num_joints}")
        
        # 9. 시간 데이터 생성
        times = np.arange(num_frames, dtype=np.float32) / fps
        
        # 10. 애니메이션 객체 생성 (기존 애니메이션 대체)
        if not hasattr(gltf, 'animations') or not gltf.animations:
            gltf.animations = []
        else:
            # 기존 애니메이션 정보 확인 (디버깅)
            print(f"기존 애니메이션 정보: {len(gltf.animations)}개 애니메이션")
        
        # 기존 애니메이션 제거
        gltf.animations = []
        
        # 새 애니메이션 객체 생성
        animation = pygltflib.Animation(
            name="MDMAnimation",
            channels=[],
            samplers=[],
            extensions={},
            extras={"fps": fps}
        )
        
        # 11. 애니메이션 바이너리 데이터 생성
        # 바이너리 블롭 초기화 (기존 바이너리 데이터 유지)
        binary_blob = bytearray()
        
        # 12. 타임라인 데이터 추가
        time_bytes = times.tobytes()
        time_offset = len(binary_blob)
        binary_blob.extend(time_bytes)
        
        # 13. BufferView 및 Accessor 초기화
        # 기존 bufferViews와 accessors 백업 (기존 모델 데이터 유지를 위함)
        original_buffer_views = gltf.bufferViews.copy() if hasattr(gltf, 'bufferViews') else []
        original_accessors = gltf.accessors.copy() if hasattr(gltf, 'accessors') else []
        
        # 애니메이션용 새 버퍼뷰 및 액세서 컬렉션
        new_buffer_views = []
        new_accessors = []
        
        # 시간 버퍼뷰 및 액세서 생성
        time_buffer_view = pygltflib.BufferView(
            buffer=0,
            byteOffset=time_offset,
            byteLength=len(time_bytes),
            extensions={},
            extras={}
        )
        new_buffer_views.append(time_buffer_view)
        
        time_accessor = pygltflib.Accessor(
            bufferView=0,  # 새 버퍼뷰 인덱스
            componentType=pygltflib.FLOAT,
            count=num_frames,
            type=pygltflib.SCALAR,
            max=[float(times.max())],
            min=[float(times.min())],
            extensions={},
            extras={}
        )
        new_accessors.append(time_accessor)
        
        # 14. 각 매핑된 본에 대한 애니메이션 채널 생성
        processed_nodes = set()
        
        # 기본 본만 처리 (자식 본은 이후에 보간)
        for joint_idx, node_idx in bone_idx_mapping.items():
            if joint_idx >= num_joints:
                continue
            
            # 위치 데이터 추출 (F, 3)
            positions = local_anim_data[:, joint_idx, :].astype(np.float32)
            
            # 필요한 경우 위치 데이터 스케일 조정
            # positions *= 0.01  # 스케일 조정이 필요한 경우
            
            # 위치 데이터를 바이너리에 추가
            pos_bytes = positions.tobytes()
            pos_offset = len(binary_blob)
            binary_blob.extend(pos_bytes)
            
            # 위치 데이터용 버퍼뷰 생성
            pos_buffer_view = pygltflib.BufferView(
                buffer=0,
                byteOffset=pos_offset,
                byteLength=len(pos_bytes),
                extensions={},
                extras={}
            )
            new_buffer_views.append(pos_buffer_view)
            pos_buffer_view_idx = len(new_buffer_views) - 1
            
            # 위치 데이터용 액세서 생성
            pos_accessor = pygltflib.Accessor(
                bufferView=pos_buffer_view_idx,
                componentType=pygltflib.FLOAT,
                count=num_frames,
                type=pygltflib.VEC3,
                max=np.max(positions, axis=0).tolist(),
                min=np.min(positions, axis=0).tolist(),
                extensions={},
                extras={}
            )
            new_accessors.append(pos_accessor)
            pos_accessor_idx = len(new_accessors) - 1
            
            # 애니메이션 샘플러 생성
            sampler = pygltflib.AnimationSampler(
                input=0,  # 시간 액세서 인덱스
                output=pos_accessor_idx,
                interpolation="LINEAR",
                extensions={},
                extras={}
            )
            animation.samplers.append(sampler)
            sampler_idx = len(animation.samplers) - 1
            
            # 애니메이션 채널 생성
            channel = pygltflib.AnimationChannel(
                sampler=sampler_idx,
                target=pygltflib.AnimationChannelTarget(
                    node=node_idx,
                    path="translation",
                    extensions={},
                    extras={}
                ),
                extensions={},
                extras={}
            )
            animation.channels.append(channel)
            processed_nodes.add(node_idx)
            
        # 15. 애니메이션 추가
        gltf.animations.append(animation)
        
        # 16. 버퍼뷰 및 액세서 설정
        # 두 가지 방법 중 하나 선택:
        
        # 방법 1: 기존 데이터를 유지하고 애니메이션 데이터만 추가
        # gltf.bufferViews = original_buffer_views + new_buffer_views
        # gltf.accessors = original_accessors + new_accessors
        
        # 방법 2: 애니메이션 데이터만 사용 (스킨 메시 정보 손실 가능성)
        gltf.bufferViews = new_buffer_views
        gltf.accessors = new_accessors
        
        # 17. 버퍼 설정
        buffer = pygltflib.Buffer(
            byteLength=len(binary_blob),
            extensions={},
            extras={}
        )
        gltf.buffers = [buffer]
        
        # 18. Extensions 필드 초기화
        fix_extensions_extras(gltf)
        
        # 19. 바이너리 데이터 설정
        gltf.set_binary_blob(bytes(binary_blob))
        
        # 20. 파일 저장
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        gltf.save(output_path)
        
        # 21. 보안 조치: 대체 방법으로 모델을 생성
        # 애니메이션만 추출하여 별도 파일로 저장
        anim_only_path = str(temp_dir / "animation_only.glb")
        
        # 깨끗한 상태에서 시작
        clean_gltf = pygltflib.GLTF2()
        clean_gltf.scene = gltf.scene
        clean_gltf.scenes = gltf.scenes
        clean_gltf.nodes = gltf.nodes
        clean_gltf.animations = gltf.animations
        clean_gltf.buffers = gltf.buffers
        clean_gltf.bufferViews = gltf.bufferViews
        clean_gltf.accessors = gltf.accessors
        
        # 새로운 GLB 파일로 저장 (애니메이션 정보만 포함)
        clean_gltf.set_binary_blob(bytes(binary_blob))
        clean_gltf.save(anim_only_path)
        
        # 임시 디렉토리 정리
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
        
        print(f"===== 스킨 모델에 애니메이션 적용 완료 =====")
        print(f"저장 경로: {output_path}")
        print(f"애니메이션: {len(animation.channels)}개 채널, {num_frames}프레임, {fps}fps")
        
        return output_path
        
    except Exception as e:
        print(f"스킨 모델에 애니메이션 적용 중 오류 발생: {str(e)}")
        traceback.print_exc()
        return None


