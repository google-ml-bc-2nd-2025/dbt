"""
GLB 파일 생성 관련 모듈
"""

import os
import numpy as np
import struct
import pygltflib
from pygltflib import GLTF2

def generate_glb_from_pose_data(pose_data, output_path, fps=30, joint_names=None):
    """
    포즈 데이터를 GLB 파일로 변환하여 저장합니다.
    
    Args:
        pose_data: [프레임, 랜드마크, 3차원] 형태의 포즈 데이터 배열
        output_path: 저장할 GLB 파일 경로
        fps: 프레임 레이트
        joint_names: 관절 이름 목록 (없으면 기본값 사용)
        
    Returns:
        성공 여부와 결과 메시지
    """
    try:
        # 관절 이름이 없으면 기본값 사용
        if joint_names is None:
            joint_names = [
                "Hips", "Spine", "Spine1", "Spine2", "Neck", "Head",
                "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
                "RightShoulder", "RightArm", "RightForeArm", "RightHand",
                "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
                "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase"
            ]
        
        # 포즈 데이터 준비
        num_frames = pose_data.shape[0]
        num_joints = min(len(joint_names), pose_data.shape[1])
        
        # GLB 파일 생성 시작
        print("GLB 파일 생성 시작...")
        gltf = GLTF2()

        # 씬 설정
        scene = pygltflib.Scene(nodes=[0])
        gltf.scenes.append(scene)
        gltf.scene = 0

        # 모든 바이너리 데이터를 담을 바이트 배열
        binary_blob = bytearray()

        # 노드 이름과 인덱스 매핑을 위한 딕셔너리
        node_indices = {}

        # 루트 노드 생성
        root_node = pygltflib.Node(name="Armature")
        gltf.nodes.append(root_node)
        node_indices["Armature"] = 0

        # 관절 계층 구조 정의
        hierarchy = {
            # 척추 계층
            "mixamorig:Spine": "mixamorig:Hips",           # 척추는 골반에 연결
            "mixamorig:Spine1": "mixamorig:Spine",         # 척추1은 척추에 연결
            "mixamorig:Spine2": "mixamorig:Spine1",        # 척추2는 척추1에 연결
            "mixamorig:Neck": "mixamorig:Spine2",          # 목은 척추2에 연결
            "mixamorig:Head": "mixamorig:Neck",            # 머리는 목에 연결
            
            # 왼쪽 팔 계층
            "mixamorig:LeftShoulder": "mixamorig:Spine2",  # 왼쪽 어깨는 척추2에 연결
            "mixamorig:LeftArm": "mixamorig:LeftShoulder", # 왼쪽 팔은 왼쪽 어깨에 연결
            "mixamorig:LeftForeArm": "mixamorig:LeftArm",  # 왼쪽 팔뚝은 왼쪽 팔에 연결
            "mixamorig:LeftHand": "mixamorig:LeftForeArm", # 왼쪽 손은 왼쪽 팔뚝에 연결
            
            # 오른쪽 팔 계층
            "mixamorig:RightShoulder": "mixamorig:Spine2",   # 오른쪽 어깨는 척추2에 연결
            "mixamorig:RightArm": "mixamorig:RightShoulder", # 오른쪽 팔은 오른쪽 어깨에 연결
            "mixamorig:RightForeArm": "mixamorig:RightArm",  # 오른쪽 팔뚝은 오른쪽 팔에 연결
            "mixamorig:RightHand": "mixamorig:RightForeArm", # 오른쪽 손은 오른쪽 팔뚝에 연결
            
            # 왼쪽 다리 계층
            "mixamorig:LeftUpLeg": "mixamorig:Hips",       # 왼쪽 윗다리는 골반에 연결
            "mixamorig:LeftLeg": "mixamorig:LeftUpLeg",    # 왼쪽 다리는 왼쪽 윗다리에 연결
            "mixamorig:LeftFoot": "mixamorig:LeftLeg",     # 왼쪽 발은 왼쪽 다리에 연결
            "mixamorig:LeftToeBase": "mixamorig:LeftFoot", # 왼쪽 발가락은 왼쪽 발에 연결
            
            # 오른쪽 다리 계층
            "mixamorig:RightUpLeg": "mixamorig:Hips",        # 오른쪽 윗다리는 골반에 연결
            "mixamorig:RightLeg": "mixamorig:RightUpLeg",    # 오른쪽 다리는 오른쪽 윗다리에 연결
            "mixamorig:RightFoot": "mixamorig:RightLeg",     # 오른쪽 발은 오른쪽 다리에 연결
            "mixamorig:RightToeBase": "mixamorig:RightFoot"  # 오른쪽 발가락은 오른쪽 발에 연결
        }

        # 전체 노드 생성
        for i, joint_name in enumerate(joint_names[:num_joints]):
            # 이미 생성된 노드는 건너뛰기
            if joint_name in node_indices:
                continue
            
            # 노드 생성
            node = pygltflib.Node(name=joint_name)
            
            # 초기 위치 (첫 프레임의 위치)
            initial_pos = pose_data[0, i].tolist()
            node.translation = [float(x) for x in initial_pos]
            
            # Identity 회전 및 스케일 값 설정
            node.rotation = [0.0, 0.0, 0.0, 1.0]  # 쿼터니언 (x, y, z, w)
            node.scale = [1.0, 1.0, 1.0]
            
            gltf.nodes.append(node)
            node_idx = len(gltf.nodes) - 1
            node_indices[joint_name] = node_idx

        # 계층 구조 설정
        for joint_name, parent_name in hierarchy.items():
            if joint_name not in node_indices or parent_name not in node_indices:
                continue
            
            node_idx = node_indices[joint_name]
            parent_idx = node_indices[parent_name]
            
            if not hasattr(gltf.nodes[parent_idx], "children") or gltf.nodes[parent_idx].children is None:
                gltf.nodes[parent_idx].children = []
            gltf.nodes[parent_idx].children.append(node_idx)

        # 애니메이션 생성
        animation = pygltflib.Animation(name="mixamo.com")

        # 시간 데이터 생성
        times = np.linspace(0, num_frames / fps, num_frames, dtype=np.float32)

        # 시간 데이터를 바이너리 블롭에 추가
        time_byte_offset = len(binary_blob)
        binary_blob.extend(times.tobytes())

        # 시간 데이터를 위한 버퍼 뷰 생성
        time_buffer_view = pygltflib.BufferView(
            buffer=0,
            byteOffset=time_byte_offset,
            byteLength=times.nbytes,
            target=None
        )
        gltf.bufferViews.append(time_buffer_view)

        # 시간 데이터를 위한 액세서 생성
        time_accessor = pygltflib.Accessor(
            bufferView=len(gltf.bufferViews) - 1,
            componentType=pygltflib.FLOAT,
            count=num_frames,
            type=pygltflib.SCALAR,
            max=[float(times[-1])],
            min=[float(times[0])]
        )
        gltf.accessors.append(time_accessor)
        time_accessor_index = len(gltf.accessors) - 1

        # 각 관절에 대한 애니메이션 채널 생성
        for i, joint_name in enumerate(joint_names[:num_joints]):
            if joint_name not in node_indices:
                continue
            
            # 노드 인덱스 확인
            node_idx = node_indices[joint_name]
            
            # 해당 관절의 모든 프레임 위치 데이터
            positions = pose_data[:, i]
            
            # 위치 데이터 검증 및 수정
            positions = np.nan_to_num(positions, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 위치 데이터를 바이너리 블롭에 추가
            position_byte_offset = len(binary_blob)
            binary_blob.extend(positions.tobytes())
            
            # 위치 데이터를 위한 버퍼 뷰 생성
            position_buffer_view = pygltflib.BufferView(
                buffer=0,
                byteOffset=position_byte_offset,
                byteLength=positions.nbytes,
                target=None
            )
            gltf.bufferViews.append(position_buffer_view)
            
            # 위치 데이터를 위한 액세서 생성
            max_vals = [float(x) for x in positions.max(axis=0)]
            min_vals = [float(x) for x in positions.min(axis=0)]
            
            position_accessor = pygltflib.Accessor(
                bufferView=len(gltf.bufferViews) - 1,
                componentType=pygltflib.FLOAT,
                count=num_frames,
                type=pygltflib.VEC3,
                max=max_vals,
                min=min_vals
            )
            gltf.accessors.append(position_accessor)
            position_accessor_index = len(gltf.accessors) - 1
            
            # 애니메이션 샘플러 생성
            sampler = pygltflib.AnimationSampler(
                input=time_accessor_index,
                output=position_accessor_index,
                interpolation="LINEAR"
            )
            animation.samplers.append(sampler)
            
            # 애니메이션 채널 생성
            channel = pygltflib.AnimationChannel(
                sampler=len(animation.samplers) - 1,
                target=pygltflib.AnimationChannelTarget(
                    node=node_idx,
                    path="translation"
                )
            )
            animation.channels.append(channel)

        # 애니메이션 추가
        gltf.animations.append(animation)

        # 버퍼 추가
        buffer = pygltflib.Buffer(byteLength=len(binary_blob))
        gltf.buffers.append(buffer)
        
        # 파일 저장
        try:
            # 디렉토리 생성
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 바이너리 형태로 저장
            with open(output_path, 'wb') as f:
                # GLB 파일 헤더
                f.write(b'glTF')  # 매직 바이트
                f.write(struct.pack('<I', 2))  # 버전 2
                
                # JSON 데이터 준비
                json_data = gltf.to_json().encode('utf-8')
                json_length = len(json_data)
                padding = (4 - json_length % 4) % 4
                json_data += b' ' * padding
                
                # 바이너리 데이터 패딩
                bin_length = len(binary_blob)
                bin_padding = (4 - bin_length % 4) % 4
                binary_blob += b'\x00' * bin_padding
                
                # 전체 크기
                total_length = 12 + 8 + len(json_data) + 8 + len(binary_blob)
                f.write(struct.pack('<I', total_length))
                
                # JSON 청크
                f.write(struct.pack('<I', len(json_data)))
                f.write(struct.pack('<I', 0x4E4F534A))  # 'JSON'
                f.write(json_data)
                
                # 바이너리 청크
                f.write(struct.pack('<I', len(binary_blob)))
                f.write(struct.pack('<I', 0x004E4942))  # 'BIN\0'
                f.write(binary_blob)
            
            return True, f"GLB 파일이 성공적으로 저장되었습니다: {output_path}"
        except Exception as e:
            print(f"GLB 파일 저장 오류: {e}")
            return False, f"GLB 파일 저장 중 오류 발생: {str(e)}"
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, f"GLB 생성 중 오류 발생: {str(e)}"

def map_mediapipe_to_glb_joints(pose_array):
    """
    MediaPipe 포즈 데이터를 Mixamo 호환 관절 구조로 매핑합니다.
    
    Args:
        pose_array: MediaPipe 포즈 데이터 배열 (T, N, 3), N은 33 또는 그 이상
        
    Returns:
        매핑된 Mixamo 호환 포즈 데이터 배열 (T, 22, 3)
    """
    # 디버깅: 입력 배열 정보 출력
    print(f"\n===== MediaPipe -> Mixamo GLB 매핑 정보 =====")
    print(f"입력 배열 크기: {pose_array.shape}")
    print(f"프레임 수: {pose_array.shape[0]}")
    print(f"입력 랜드마크 수: {pose_array.shape[1]}")
    
    # MediaPipe 관절 인덱스 정보 (참고용)
    mp_landmarks_info = {
        0: "nose", 
        1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
        4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer",
        7: "left_ear", 8: "right_ear",
        9: "mouth_left", 10: "mouth_right",
        11: "left_shoulder", 12: "right_shoulder",
        13: "left_elbow", 14: "right_elbow",
        15: "left_wrist", 16: "right_wrist",
        17: "left_pinky", 18: "right_pinky",
        19: "left_index", 20: "right_index",
        21: "left_thumb", 22: "right_thumb",
        23: "left_hip", 24: "right_hip",
        25: "left_knee", 26: "right_knee",
        27: "left_ankle", 28: "right_ankle",
        29: "left_heel", 30: "right_heel",
        31: "left_foot_index", 32: "right_foot_index"
    }
    
    # Mixamo에서 사용하는 주요 관절 이름 정의
    mixamo_joint_names = [
        "mixamorig:Hips", 
        "mixamorig:Spine", 
        "mixamorig:Spine1", 
        "mixamorig:Spine2", 
        "mixamorig:Neck", 
        "mixamorig:Head",
        "mixamorig:LeftShoulder", 
        "mixamorig:LeftArm", 
        "mixamorig:LeftForeArm", 
        "mixamorig:LeftHand",
        "mixamorig:RightShoulder", 
        "mixamorig:RightArm", 
        "mixamorig:RightForeArm", 
        "mixamorig:RightHand",
        "mixamorig:LeftUpLeg", 
        "mixamorig:LeftLeg", 
        "mixamorig:LeftFoot", 
        "mixamorig:LeftToeBase",
        "mixamorig:RightUpLeg", 
        "mixamorig:RightLeg", 
        "mixamorig:RightFoot", 
        "mixamorig:RightToeBase"
    ]
    
    print(f"\n생성할 Mixamo 호환 관절 수: {len(mixamo_joint_names)}")
    for i, name in enumerate(mixamo_joint_names):
        print(f"  {i}: {name}")
    
    # 출력 배열 초기화 - 오직 Mixamo 관절만 포함
    num_frames = pose_array.shape[0]
    num_joints = len(mixamo_joint_names)
    output = np.zeros((num_frames, num_joints, 3), dtype=np.float32)
    
    print("\n===== 관절 매핑 처리 시작 =====")
    
    # MediaPipe 좌표계를 Mixamo 좌표계로 변환하기 위한 함수
    def convert_coordinates(point):
        # MediaPipe는 오른손 좌표계, Y가 아래로 증가
        # Mixamo는 오른손 좌표계, Y가 위로 증가
        # Z축 반전 및 Y축 방향 조정
        return np.array([point[0], point[1], -point[2]])
    
    # 각 프레임 처리
    for frame in range(num_frames):
        if frame == 0:  # 첫 프레임만 자세히 로깅
            print(f"\n프레임 {frame} 처리:")
            
        frame_data = pose_array[frame].copy()
        
        # 전체 좌표계 변환 적용
        for i in range(frame_data.shape[0]):
            frame_data[i] = convert_coordinates(frame_data[i])
        
        # 발이 등에 붙는 문제 해결을 위한 스케일 조정
        # Y축 스케일을 조정하여 키를 더 크게 만듦
        y_scale_factor = 1.5  # Y축 스케일 팩터 증가
        for i in range(frame_data.shape[0]):
            frame_data[i, 1] *= y_scale_factor
        
        if frame == 0:
            print(f"  좌표계 변환 및 Y축 스케일 조정 (factor: {y_scale_factor}) 적용됨")
        
        # 1. 골반 및 척추 (중심부)
        hips_pos = (frame_data[23] + frame_data[24]) / 2  # 양쪽 골반의 중간점
        output[frame, 0] = hips_pos  # Hips
        
        if frame == 0:
            print(f"  Hips 위치 계산: ({frame_data[23]}) + ({frame_data[24]}) / 2 = {hips_pos}")
        
        # 등뼈 계층은 골반에서 목까지 균등하게 분배
        spine_base = hips_pos.copy()
        neck_pos = (frame_data[11] + frame_data[12]) / 2  # 양쪽 어깨 중간점
        spine_vec = (neck_pos - spine_base) / 3  # 척추를 3등분
        
        output[frame, 1] = spine_base + spine_vec * 0.33  # Spine (1/3 지점)
        output[frame, 2] = spine_base + spine_vec * 0.66  # Spine1 (2/3 지점)
        output[frame, 3] = spine_base + spine_vec  # Spine2 (목 아래)
        
        if frame == 0:
            print(f"  목 위치 계산: ({frame_data[11]}) + (({frame_data[12]}) - ({frame_data[11]})) / 2 = {neck_pos}")
            print(f"  척추 벡터: ({neck_pos}) - ({spine_base}) / 3 = {spine_vec}")
            print(f"  Spine 위치: {spine_base} + {spine_vec} * 0.33 = {output[frame, 1]}")
            print(f"  Spine1 위치: {spine_base} + {spine_vec} * 0.66 = {output[frame, 2]}")
            print(f"  Spine2 위치: {spine_base} + {spine_vec} = {output[frame, 3]}")
        
        # 2. 머리 부분
        head_base = neck_pos.copy()
        nose_pos = frame_data[0]  # 코 위치
        neck_length = np.linalg.norm(nose_pos - head_base) * 0.3
        neck_dir = (nose_pos - head_base) / np.linalg.norm(nose_pos - head_base)
        
        output[frame, 4] = head_base + neck_dir * neck_length  # Neck (목 위치)
        output[frame, 5] = nose_pos  # Head (머리 위치)
        
        if frame == 0:
            print(f"  코 위치: {nose_pos}")
            print(f"  Neck 위치: {head_base} + {neck_dir} * {neck_length} = {output[frame, 4]}")
            print(f"  Head 위치: {nose_pos}")
        
        # 3. 왼쪽 팔
        l_shoulder = frame_data[11].copy()  # 왼쪽 어깨
        l_elbow = frame_data[13].copy()  # 왼쪽 팔꿈치
        l_wrist = frame_data[15].copy()  # 왼쪽 손목
        
        output[frame, 6] = l_shoulder  # LeftShoulder
        output[frame, 7] = l_elbow     # LeftArm
        output[frame, 8] = l_wrist     # LeftForeArm
        
        # 손 위치 (손목에서 약간 더 연장)
        l_hand_vec = l_wrist - l_elbow
        l_hand_length = np.linalg.norm(l_hand_vec)
        if l_hand_length > 0:
            l_hand_dir = l_hand_vec / l_hand_length
            output[frame, 9] = l_wrist + l_hand_dir * l_hand_length * 0.3  # LeftHand
        else:
            output[frame, 9] = l_wrist  # 방향을 알 수 없으면 손목 위치 사용
        
        if frame == 0:
            print(f"  왼쪽 어깨: {l_shoulder}")
            print(f"  왼쪽 팔꿈치: {l_elbow}")
            print(f"  왼쪽 손목: {l_wrist}")
            print(f"  왼쪽 손 방향 벡터: {l_hand_vec}, 길이: {l_hand_length}")
            if l_hand_length > 0:
                print(f"  왼쪽 손 위치: {l_wrist} + {l_hand_dir} * {l_hand_length} * 0.3 = {output[frame, 9]}")
            else:
                print(f"  왼쪽 손 위치: {l_wrist} (손목 위치 사용)")
        
        # 4. 오른쪽 팔
        r_shoulder = frame_data[12].copy()  # 오른쪽 어깨
        r_elbow = frame_data[14].copy()     # 오른쪽 팔꿈치
        r_wrist = frame_data[16].copy()     # 오른쪽 손목
        
        output[frame, 10] = r_shoulder  # RightShoulder
        output[frame, 11] = r_elbow     # RightArm
        output[frame, 12] = r_wrist     # RightForeArm
        
        # 손 위치 (손목에서 약간 더 연장)
        r_hand_vec = r_wrist - r_elbow
        r_hand_length = np.linalg.norm(r_hand_vec)
        if r_hand_length > 0:
            r_hand_dir = r_hand_vec / r_hand_length
            output[frame, 13] = r_wrist + r_hand_dir * r_hand_length * 0.3  # RightHand
        else:
            output[frame, 13] = r_wrist  # 방향을 알 수 없으면 손목 위치 사용
        
        if frame == 0:
            print(f"  오른쪽 어깨: {r_shoulder}")
            print(f"  오른쪽 팔꿈치: {r_elbow}")
            print(f"  오른쪽 손목: {r_wrist}")
            print(f"  오른쪽 손 방향 벡터: {r_hand_vec}, 길이: {r_hand_length}")
            if r_hand_length > 0:
                print(f"  오른쪽 손 위치: {r_wrist} + {r_hand_dir} * {r_hand_length} * 0.3 = {output[frame, 13]}")
            else:
                print(f"  오른쪽 손 위치: {r_wrist} (손목 위치 사용)")
        
        # 5. 왼쪽 다리
        l_hip = frame_data[23].copy()   # 왼쪽 골반
        l_knee = frame_data[25].copy()  # 왼쪽 무릎
        l_ankle = frame_data[27].copy() # 왼쪽 발목
        l_foot = frame_data[31].copy()  # 왼쪽 발끝
        
        # 발이 등에 붙는 문제 해결 - 발목과 발끝 사이의 위치 보정
        l_foot_vec = l_foot - l_ankle
        l_foot_length = np.linalg.norm(l_foot_vec)
        if l_foot_length > 0:
            l_foot_dir = l_foot_vec / l_foot_length
            # 발끝 위치를 발목에서 일정 거리만큼만 연장
            l_foot = l_ankle + l_foot_dir * l_foot_length * 0.5
        
        output[frame, 14] = l_hip     # LeftUpLeg
        output[frame, 15] = l_knee    # LeftLeg
        output[frame, 16] = l_ankle   # LeftFoot
        output[frame, 17] = l_foot    # LeftToeBase
        
        if frame == 0:
            print(f"  왼쪽 골반: {l_hip}")
            print(f"  왼쪽 무릎: {l_knee}")
            print(f"  왼쪽 발목: {l_ankle}")
            print(f"  왼쪽 발끝 (보정됨): {l_foot}")
        
        # 6. 오른쪽 다리
        r_hip = frame_data[24].copy()   # 오른쪽 골반
        r_knee = frame_data[26].copy()  # 오른쪽 무릎
        r_ankle = frame_data[28].copy() # 오른쪽 발목
        r_foot = frame_data[32].copy()  # 오른쪽 발끝
        
        # 발이 등에 붙는 문제 해결 - 발목과 발끝 사이의 위치 보정
        r_foot_vec = r_foot - r_ankle
        r_foot_length = np.linalg.norm(r_foot_vec)
        if r_foot_length > 0:
            r_foot_dir = r_foot_vec / r_foot_length
            # 발끝 위치를 발목에서 일정 거리만큼만 연장
            r_foot = r_ankle + r_foot_dir * r_foot_length * 0.5
        
        output[frame, 18] = r_hip     # RightUpLeg
        output[frame, 19] = r_knee    # RightLeg
        output[frame, 20] = r_ankle   # RightFoot
        output[frame, 21] = r_foot    # RightToeBase
        
        if frame == 0:
            print(f"  오른쪽 골반: {r_hip}")
            print(f"  오른쪽 무릎: {r_knee}")
            print(f"  오른쪽 발목: {r_ankle}")
            print(f"  오른쪽 발끝 (보정됨): {r_foot}")
    
    # 일부 수정: 골반 중심 위치 조정 (양쪽 골반의 중간점 사용)
    for frame in range(num_frames):
        center_pos = (output[frame, 14] + output[frame, 18]) / 2  # 양쪽 골반의 중간점
        output[frame, 0] = center_pos  # Hips 위치를 정확한 중간점으로 조정
        
        if frame == 0:
            print(f"\n골반 중심 위치 보정: ({output[frame, 14]}) + ({output[frame, 18]}) / 2 = {center_pos}")
    
    # 회전 방향 조정 (왼쪽/오른쪽 구분 명확히)
    print("\n좌우 자세 확인:")
    is_mirrored = False
    if num_frames > 0:
        # 왼쪽/오른쪽 어깨 위치로 좌우 확인
        l_shoulder_x = output[0, 6, 0]  # 왼쪽 어깨 X좌표
        r_shoulder_x = output[0, 10, 0]  # 오른쪽 어깨 X좌표
        
        print(f"  왼쪽 어깨 X좌표: {l_shoulder_x}")
        print(f"  오른쪽 어깨 X좌표: {r_shoulder_x}")
        
        # 왼쪽 어깨가 오른쪽 어깨보다 오른쪽에 있으면 좌우가 뒤집힌 것
        if l_shoulder_x > r_shoulder_x:
            is_mirrored = True
            print("  경고: 좌우가 뒤집혀 있습니다!")
        else:
            print("  좌우 방향이 정상입니다.")
    
    # 좌우가 뒤집혔다면 수정 (선택 사항)
    if is_mirrored:
        # X축 방향을 뒤집어 좌우 조정
        print("  좌우 수정: X축 방향을 뒤집습니다.")
        output[:, :, 0] *= -1
    
    # 최종 결과 요약
    print(f"\n===== 변환 완료 =====")
    print(f"입력: {pose_array.shape[1]} 관절 -> 출력: {len(mixamo_joint_names)} 관절")
    print(f"프레임 수: {num_frames}")
    
    # 첫 프레임의 일부 관절 위치 확인
    if num_frames > 0:
        print("\n첫 번째 프레임의 주요 관절 최종 위치:")
        important_joints = [0, 6, 10, 14, 18]  # 골반, 왼쪽 어깨, 오른쪽 어깨, 왼쪽 골반, 오른쪽 골반
        for idx in important_joints:
            print(f"  {idx}: {mixamo_joint_names[idx]} - {output[0, idx]}")
    
    return output, mixamo_joint_names