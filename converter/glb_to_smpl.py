import os
import numpy as np
import trimesh
import json
from pathlib import Path
import smplx
import torch
from scipy.spatial.transform import Rotation as R

def load_glb_file(glb_path):
    """
    Load GLB file and extract mesh and animation data
    """
    print(f"Loading GLB file: {glb_path}")
    try:
        scene = trimesh.load(glb_path)
        print(f"Successfully loaded GLB scene")
        return scene
    except Exception as e:
        print(f"Error loading GLB file: {e}")
        return None

def load_with_pygltf(glb_path):
    """PyGLTF를 사용하여 GLB 파일 로드하기"""
    try:
        import pygltflib
        gltf = pygltflib.GLTF2().load(glb_path)
        print(f"애니메이션 수: {len(gltf.animations)}")
        return gltf
    except ImportError:
        print("pygltflib가 설치되어 있지 않습니다. pip install pygltflib 명령으로 설치하세요.")
        return None

def extract_skeleton_and_animation(scene):
    """
    Extract skeleton and animation data from the GLB scene
    Using a more direct approach based on trimesh's scene structure
    """
    joints = []
    animations = {}
    
    # 디버깅을 위한 scene 구조 출력
    print("Scene structure analysis:")
    print(f"Scene type: {type(scene)}")
    
    # Scene 객체가 가진 속성들 확인
    scene_attrs = dir(scene)
    print(f"Scene attributes: {[attr for attr in scene_attrs if not attr.startswith('_')]}")
    
    # scene.geometry 속성이 있는지 확인
    if hasattr(scene, 'geometry'):
        print(f"Scene geometry: {list(scene.geometry.keys())}")
    
    # trimesh.Scene 객체는 일반적으로 다음과 같은 속성을 가질 수 있음
    # - graph: 씬 그래프
    # - geometry: 메시와 같은 기하학적 객체들
    # - metadata: 씬 메타데이터
    
    try:
        # 접근 방법 1: scene.graph 직접 검사
        if hasattr(scene, 'graph'):
            print("Examining scene.graph...")
            
            # scene.graph의 구조 확인
            if hasattr(scene.graph, 'nodes'):
                print("Found scene.graph.nodes")
                
                # scene.graph.nodes의 타입 확인
                nodes_type = type(scene.graph.nodes)
                print(f"scene.graph.nodes type: {nodes_type}")
                
                # 노드 수 확인
                try:
                    node_count = len(scene.graph.nodes)
                    print(f"Number of nodes: {node_count}")
                    
                    # 노드 키 출력 (최대 5개)
                    node_keys = list(scene.graph.nodes)[:5] if node_count > 0 else []
                    print(f"Sample node keys: {node_keys}")
                except Exception as e:
                    print(f"Error accessing node count: {e}")
                
                # scene.graph.nodes의 내용을 안전하게 처리
                # 1. 노드가 있는지 확인
                if hasattr(scene.graph, 'nodes') and scene.graph.nodes:
                    try:
                        # 모든 노드 순회 대신 직접적인 본 정보 접근 시도
                        for node_name in list(scene.graph.nodes):
                            # 노드 이름에 'joint'가 포함되어 있는지 확인 
                            if 'joint' in str(node_name).lower():
                                # 본 노드 정보 가져오기
                                try:
                                    # trimesh에서는 transforms를 가져오는 방법이 여러 가지가 있음
                                    transform = None
                                    
                                    # 방법 1: 변환 매트릭스를 직접 조회
                                    if hasattr(scene.graph, 'get'):
                                        node_data = scene.graph.get(node_name)
                                        if isinstance(node_data, dict) and 'transform' in node_data:
                                            transform = node_data['transform']
                                    
                                    # 방법 2: 씬 그래프의 변환 매트릭스 가져오기
                                    if transform is None and hasattr(scene, 'get_transform'):
                                        try:
                                            transform = scene.get_transform(node_name)
                                        except:
                                            pass
                                    
                                    if transform is not None:
                                        print(f"Found transform for node: {node_name}")
                                        if hasattr(transform, 'tolist'):
                                            transform_data = transform.tolist()
                                        else:
                                            transform_data = transform  # 이미 리스트 형태일 경우
                                        
                                        joints.append({
                                            'name': node_name,
                                            'transform': transform_data
                                        })
                                except Exception as e:
                                    print(f"Error processing node {node_name}: {e}")
                    except Exception as e:
                        print(f"Error iterating through nodes: {e}")
            
            # 접근 방법 2: 씬의 XML 구조 확인
            if hasattr(scene, 'export'):
                try:
                    print("Trying to export scene metadata...")
                    # 씬 메타데이터 가져오기
                    if hasattr(scene, 'metadata'):
                        print(f"Scene metadata keys: {scene.metadata.keys() if scene.metadata else 'None'}")
                except Exception as e:
                    print(f"Error exporting scene metadata: {e}")
        
        # 접근 방법 3: 씬의 joint 관련 함수 찾기
        if hasattr(scene, 'joints') and callable(scene.joints):
            try:
                scene_joints = scene.joints()
                print(f"Found {len(scene_joints)} joints using scene.joints()")
                for i, joint in enumerate(scene_joints):
                    print(i, f"joint = {joint}")
                    if hasattr(joint, 'matrix') and hasattr(joint.matrix, 'tolist'):
                        joints.append({
                            'name': getattr(joint, 'name', f"joint_{i}"),
                            'transform': joint.matrix.tolist()
                        })
            except Exception as e:
                print(f"Error getting joints from scene.joints(): {e}")
    
    except Exception as e:
        print(f"General error in scene analysis: {e}")
    
    # 애니메이션 데이터 추출
    if hasattr(scene, 'animations'):
        try:
            for i, animation in enumerate(scene.animations):
                frame_count = getattr(animation, 'frames', 0)
                if hasattr(frame_count, '__len__'):
                    frame_count = len(frame_count)
                
                anim_data = getattr(animation, 'data', None)
                
                animations[f"animation_{i}"] = {
                    'frames': frame_count,
                    'data': anim_data
                }
                print(f"Added animation {i} with {frame_count} frames")
        except Exception as e:
            print(f"Error extracting animations: {e}")
    
    # 본이 없다면 대체 접근법 시도: 메시에서 직접 본 정보 추출
    if not joints and hasattr(scene, 'geometry'):
        try:
            print("No joints found using scene graph. Trying to extract from meshes...")
            for geo_name, geometry in scene.geometry.items():
                if hasattr(geometry, 'visual') and hasattr(geometry.visual, 'skeleton'):
                    skel = geometry.visual.skeleton
                    if (skel):
                        print(f"Found skeleton in geometry {geo_name}")
                        for i, bone in enumerate(skel.bones):
                            if hasattr(bone, 'matrix'):
                                joints.append({
                                    'name': getattr(bone, 'name', f"bone_{i}"),
                                    'transform': bone.matrix.tolist() if hasattr(bone.matrix, 'tolist') else bone.matrix
                                })
        except Exception as e:
            print(f"Error extracting bones from geometry: {e}")
    
    print(f"Final extraction: {len(joints)} joints and {len(animations)} animations")
    return joints, animations

def process_animation_frames(animations, joints, fps=30):
    """
    애니메이션 데이터를 프레임별 SMPL 포즈 파라미터로 변환
    """
    print("\n=== 애니메이션 프레임 처리 시작 ===")
    print(f"전달된 애니메이션 수: {len(animations)}")
    print(f"전달된 관절 수: {len(joints)}")
    
    if animations:
        # 첫 번째 애니메이션 정보
        anim_key = next(iter(animations))
        anim_data = animations[anim_key]
        print(f"첫 번째 애니메이션: {anim_key}")
        print(f"  프레임 수: {anim_data.get('frames', 0)}")
        print(f"  타임라인 길이: {len(anim_data.get('times', []))}")
        print(f"  노드 수: {len(anim_data.get('nodes', {}))}")
        
        # 노드-관절 매핑 가능 여부
        node_indices = [j.get('node_index') for j in joints if 'node_index' in j]
        print(f"노드 인덱스가 있는 관절 수: {len(node_indices)}")
        
        # 애니메이션 노드와 관절 노드의 교집합
        if anim_data.get('nodes'):
            anim_nodes = set(anim_data['nodes'].keys())
            joint_nodes = set(node_indices)
            intersection = anim_nodes.intersection(joint_nodes)
            print(f"애니메이션과 관절의 공통 노드 수: {len(intersection)}")
            if intersection:
                print(f"공통 노드 예시: {list(intersection)[:5]}")
    
    frame_poses = []
    
    if not animations:
        print("애니메이션 데이터가 없습니다. 정적 포즈를 사용합니다.")
        # 정적 포즈 생성 로직
        pose_params = np.zeros(72)
        for i, joint in enumerate(joints):
            if i < 24:  # SMPL은 24개 관절을 사용
                try:
                    transform = np.array(joint['transform']).reshape(4, 4)
                    rotation = R.from_matrix(transform[:3, :3])
                    rot_angles = rotation.as_rotvec()
                    pose_params[i*3:(i+1)*3] = rot_angles
                except Exception as e:
                    print(f"관절 {i} 처리 중 오류: {e}")
        
        # 정적 포즈를 여러 프레임에 복제
        frame_poses = [pose_params.copy() for _ in range(30)]
        return frame_poses
    
    # 첫 번째 애니메이션 사용
    anim_key = next(iter(animations))
    animation = animations[anim_key]
    
    print(f"애니메이션 '{anim_key}' 처리 중 ({animation.get('frames', 0)} 프레임)")
    
    # 시간 정보와 노드 애니메이션 데이터
    times = animation.get('times', [])
    node_animations = animation.get('nodes', {})
    
    if not times:
        print("시간 데이터가 없습니다.")
        return []
    
    # 노드 인덱스와 관절 인덱스 매핑 생성
    node_to_joint = {}
    for i, joint in enumerate(joints):
        if 'node_index' in joint:
            node_to_joint[joint['node_index']] = i
    
    # 각 프레임에 대해 처리
    for frame_idx in range(animation.get('frames', 0)):
        # SMPL 포즈 파라미터 초기화 (72차원)
        pose_params = np.zeros(72)
        
        # 각 노드에 대한 애니메이션 데이터 적용
        for node_idx, node_data in node_animations.items():
            # 이 노드가 관절에 매핑되는지 확인
            if node_idx in node_to_joint:
                joint_idx = node_to_joint[node_idx]
                
                if joint_idx < 24:  # SMPL은 24개 관절 사용
                    # 기본 변환 매트릭스 (정적 포즈)
                    transform = np.array(joints[joint_idx]['transform']).reshape(4, 4)
                    
                    # 애니메이션 데이터 적용 (디버깅 출력 추가)
                    print(f"프레임 {frame_idx}에 노드 {node_idx} (관절 {joint_idx}) 처리 중")
                    
                    # 회전 데이터
                    if 'rotation' in node_data:
                        rot_data = node_data['rotation']
                        rot_values = rot_data.get('values', [])
                        
                        if frame_idx < len(rot_values):
                            # 쿼터니언 회전 적용
                            quat = rot_values[frame_idx]
                            print(f"회전 데이터 적용: {quat}")
                            rotation = R.from_quat(quat)
                            transform[:3, :3] = rotation.as_matrix()
                    
                    # 이동 데이터
                    if 'translation' in node_data:
                        trans_data = node_data['translation']
                        trans_values = trans_data.get('values', [])
                        
                        if frame_idx < len(trans_values):
                            print(f"이동 데이터 적용: {trans_values[frame_idx]}")
                            transform[:3, 3] = trans_values[frame_idx]
                    
                    # 스케일 데이터
                    if 'scale' in node_data:
                        scale_data = node_data['scale']
                        scale_values = scale_data.get('values', [])
                        
                        if frame_idx < len(scale_values):
                            print(f"스케일 데이터 적용: {scale_values[frame_idx]}")
                            scale = scale_values[frame_idx]
                            for i in range(3):
                                transform[i, i] *= scale[i]
                    
                    # 회전 정보 추출 및 SMPL 포즈 매개변수에 적용
                    rotation = R.from_matrix(transform[:3, :3])
                    rot_angles = rotation.as_rotvec()
                    pose_params[joint_idx*3:(joint_idx+1)*3] = rot_angles
        
        # 완성된 프레임 포즈 추가
        frame_poses.append(pose_params)
    
    return frame_poses

def convert_to_smpl_sequence(joints, animations, output_path, pose_sequence=None):
    """
    애니메이션 시퀀스를 SMPL 포맷으로 변환하여 저장
    
    Args:
        joints: 추출된 관절 정보 리스트
        animations: 추출된 애니메이션 정보 딕셔너리
        output_path: 출력 파일 경로
        pose_sequence: 미리 계산된 포즈 시퀀스 (없으면 새로 계산)
        
    Returns:
        SMPL 모델 객체
    """
    import os
    from pathlib import Path
    
    # 현재 프로젝트 경로 내의 models 디렉토리 사용
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'models')
    smpl_model_path = os.path.join(current_dir, 'converted_npy')

    print(f"모델 경로: {model_path}")
    
    # smpl_model_path의 기존 pkl 파일 삭제
    import shutil
    if os.path.exists(smpl_model_path):
        shutil.rmtree(smpl_model_path)
        print(f"기존 변환 파일 삭제: {smpl_model_path}")
    
    # NPZ 파일을 PKL로 변환 시도
    converted = convert_npz_to_pkl(model_path, smpl_model_path)
    if converted:
        print("NPZ 파일을 PKL로 변환했습니다. 이제 모델을 로드합니다.")
    
    # SMPL 모델 초기화 시도
    try:
        # 변환된 npz 파일 사용 시도
        if os.path.exists(smpl_model_path):
            print(f"변환된 NPZ 모델 경로 시도: {smpl_model_path}")
            model = smplx.create(model_path=smpl_model_path, model_type='smpl', gender='neutral')
            print("변환된 NPZ 모델 로드 성공!")
        else:
            # 기본 모델 경로 시도
            print(f"기본 모델 경로 시도: {model_path}")
            model = smplx.create(model_path=model_path, model_type='smpl', gender='neutral')
            print("기본 모델 로드 성공!")
    except Exception as e:
        print(f"모델 로드 오류: {e}")
        print("다른 모델 경로를 시도합니다...")
        
        # 환경 변수 또는 홈 디렉토리에서 대체 경로 시도
        alt_model_paths = [
            os.path.join(current_dir, 'models'),  # 프로젝트 내 models 디렉토리
            os.path.expanduser("~/.smplx/models"),  # 기본 홈 디렉토리 
            os.path.expanduser("~/models"),  # 홈의 models 디렉토리
            "models",  # 현재 디렉토리의 models
            "."  # 현재 디렉토리
        ]
        
        for alt_path in alt_model_paths:
            try:
                print(f"{alt_path} 경로 시도 중...")
                model = smplx.create(model_path=alt_path, model_type='smpl', gender='neutral')
                print(f"성공! {alt_path}에서 모델을 로드했습니다.")
                break
            except Exception as err:
                print(f"{alt_path} 시도 실패: {err}")
                continue
        else:
            raise Exception("SMPL 모델 파일을 찾을 수 없습니다. 모델 파일을 올바른 위치에 배치하세요.")
    
    # 외부에서 전달된 포즈 시퀀스가 없으면 새로 생성
    if pose_sequence is None:
        print("외부에서 전달된 포즈 시퀀스가 없습니다. 매핑을 사용하여 새로 생성합니다.")
        pose_sequence = process_animation_frames_with_mapping(animations, joints)
    
    # 프레임 수 출력
    print(f"총 사용할 프레임 수: {len(pose_sequence)}")
    
    # 각 프레임에 대해 SMPL 모델 적용 및 메시 저장
    base_path = output_path.replace('.obj', '')
    
    # SMPL 파라미터 저장을 위한 딕셔너리
    smpl_sequence = {
        'poses': [],           # 포즈 파라미터 (프레임별)
        'shape': [0.0] * 10,   # 기본 체형 파라미터 
        'trans': [[0.0, 0.0, 0.0] for _ in range(len(pose_sequence))],  # 이동 파라미터 (프레임별)
        'fps': 30              # 초당 프레임 수
    }
    
    # 첫 프레임의 메시 저장
    if pose_sequence:
        first_frame = pose_sequence[0]
        
        # 포즈 크기 확인 및 조정
        if len(first_frame) != 72:
            print(f"SMPL 변환: 포즈 크기 조정 {len(first_frame)} → 72")
            fixed_pose = np.zeros(72)
            fixed_pose[:min(72, len(first_frame))] = first_frame[:min(72, len(first_frame))]
            first_frame = fixed_pose
        
        # 전역 방향(global_orient)과 몸체 포즈(body_pose) 분리
        global_orient = torch.tensor(first_frame[:3]).unsqueeze(0)
        body_pose = torch.tensor(first_frame[3:]).unsqueeze(0)
        
        # SMPL 출력 생성
        try:
            output = model(
                global_orient=global_orient,
                body_pose=body_pose,
                return_verts=True
            )
            
            # 정점과 면 저장
            vertices = output.vertices.detach().cpu().numpy()[0]
            faces = model.faces
            
            # 메시 생성 및 저장
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            mesh.export(output_path)
            print(f"SMPL 메시를 {output_path}에 저장했습니다.")
        except Exception as e:
            print(f"메시 생성 중 오류 발생: {e}")
    
    # 모든 포즈 파라미터를 JSON에 저장 - 크기 조정 확인
    for pose in pose_sequence:
        if len(pose) != 72:
            fixed_pose = np.zeros(72)
            fixed_pose[:min(72, len(pose))] = pose[:min(72, len(pose))]
            smpl_sequence['poses'].append(fixed_pose.tolist())
        else:
            smpl_sequence['poses'].append(pose.tolist())
    
    # SMPL 파라미터 JSON 저장
    json_path = base_path + "_params.json"
    with open(json_path, 'w') as f:
        json.dump(smpl_sequence, f, indent=2)
    
    print(f"모션 시퀀스 파라미터를 {json_path}에 저장했습니다.")
    
    # 모션클립 학습용 단일 NPY 파일 저장
    motionclip_path = base_path + "_motionclip.npy"
    try:
        # 포즈 시퀀스를 numpy 배열로 변환하여 저장
        poses_array = np.array([pose.tolist() if len(pose) == 72 else np.zeros(72) for pose in pose_sequence])
        np.save(motionclip_path, poses_array)
        print(f"모션클립 학습용 데이터를 {motionclip_path}에 저장했습니다.")
    except Exception as e:
        print(f"NPY 파일 저장 중 오류 발생: {e}")
    
    return model

def convert_npz_to_pkl(model_path, target_path):
    """model.npz 파일을 찾아 SMPL_GENDER.pkl로 변환"""
    import pickle
    import os
    
    # 타겟 디렉토리 생성
    target_smpl_dir = os.path.join(target_path, 'smpl')
    os.makedirs(target_smpl_dir, exist_ok=True)
    
    genders = ['female', 'male', 'neutral']
    converted = False
    
    for gender in genders:
        gender_dir = os.path.join(model_path, gender)
        if os.path.exists(gender_dir):
            npz_file = os.path.join(gender_dir, 'model.npz')
            pkl_file = os.path.join(target_smpl_dir, f'SMPL_{gender.upper()}.pkl')
            
            if os.path.exists(npz_file):
                try:
                    print(f"{npz_file}를 {pkl_file}로 변환합니다...")
                    
                    # npz 파일을 딕셔너리로 로드
                    with np.load(npz_file, allow_pickle=True) as data:
                        data_dict = dict(data)
                    
                    # 디버깅: 어떤 키가 있는지 확인
                    print(f"NPZ 파일 키: {data_dict.keys()}")
                    
                    # 로드된 데이터를 pickle로 저장
                    with open(pkl_file, 'wb') as f:
                        pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
                    
                    print(f"변환 성공: {pkl_file}")
                    converted = True
                except Exception as e:
                    print(f"{npz_file} 변환 오류: {e}")
                    # 오류 발생 시 불완전한 파일 삭제
                    if os.path.exists(pkl_file):
                        os.remove(pkl_file)
    
    return converted

def analyze_glb_with_pygltflib(glb_path):
    """PyGLTF 라이브러리를 사용하여 GLB 파일을 자세히 분석"""
    try:
        import pygltflib
        gltf = pygltflib.GLTF2().load(glb_path)
        
        print("\n==== GLB 파일 세부 분석 ====")
        
        # 기본 정보
        print(f"GLB 버전: {gltf.asset.version}")
        print(f"버퍼 수: {len(gltf.buffers)}")
        print(f"메시 수: {len(gltf.meshes)}")
        print(f"노드 수: {len(gltf.nodes)}")
        print(f"씬 수: {len(gltf.scenes)}")
        print(f"애니메이션 수: {len(gltf.animations)}")
        
        # 애니메이션 세부 정보
        if gltf.animations:
            print("\n- 애니메이션 정보:")
            for i, anim in enumerate(gltf.animations):
                print(f"\n  애니메이션 {i}:")
                print(f"  이름: {getattr(anim, 'name', '이름 없음')}")
                print(f"  채널 수: {len(anim.channels)}")
                print(f"  샘플러 수: {len(anim.samplers)}")
                
                # 채널 정보 출력
                if anim.channels:
                    print("  채널 정보:")
                    for j, channel in enumerate(anim.channels[:2]):  # 처음 2개만 출력
                        target_node = channel.target.node if hasattr(channel.target, 'node') else "알 수 없음"
                        target_path = channel.target.path if hasattr(channel.target, 'path') else "알 수 없음"
                        print(f"   채널 {j}: 노드 {target_node}, 경로 {target_path}")
                
                # 샘플러 정보 출력
                if anim.samplers:
                    print("  샘플러 정보:")
                    for j, sampler in enumerate(anim.samplers[:2]):  # 처음 2개만 출력
                        input_id = sampler.input
                        output_id = sampler.output
                        interpolation = sampler.interpolation
                        print(f"   샘플러 {j}: 입력 {input_id}, 출력 {output_id}, 보간 {interpolation}")
        else:
            print("\n애니메이션이 없습니다.")
        
        # 스킨 정보 (애니메이션과 관련될 수 있음)
        print(f"\n스킨 수: {len(gltf.skins)}")
        if gltf.skins:
            print("\n- 스킨 정보:")
            for i, skin in enumerate(gltf.skins):
                print(f"  스킨 {i}:")
                print(f"  이름: {getattr(skin, 'name', '이름 없음')}")
                print(f"  인버스 바인드 매트릭스: {skin.inverseBindMatrices}")
                print(f"  관절 수: {len(skin.joints)}")
                print(f"  스켈레톤: {skin.skeleton}")
        
        # 확장 정보 (애니메이션이 여기에 저장될 수 있음)
        print("\n- 확장 정보:")
        if hasattr(gltf, 'extensions') and gltf.extensions:
            for key, val in gltf.extensions.items():
                print(f"  {key}: {val}")
        else:
            print("  확장 없음")
        
        # 노드 정보 표시 (처음 몇 개만)
        print("\n- 노드 정보 샘플:")
        for i, node in enumerate(gltf.nodes[:5]):
            print(f"  노드 {i}:")
            print(f"  이름: {getattr(node, 'name', '이름 없음')}")
            print(f"  메시: {node.mesh}")
            print(f"  스킨: {node.skin}")
            print(f"  자식: {node.children}")
            print(f"  행렬: {node.matrix}")
            print(f"  회전: {node.rotation}")
            print(f"  이동: {node.translation}")
            print(f"  스케일: {node.scale}")
        
        # 액세서 정보 (애니메이션 데이터는 주로 액세서에 저장됨)
        print(f"\n액세서 수: {len(gltf.accessors)}")
        if gltf.accessors:
            print("\n- 액세서 정보 샘플:")
            for i, accessor in enumerate(gltf.accessors[:5]):
                print(f"  액세서 {i}:")
                print(f"  버퍼뷰: {accessor.bufferView}")
                print(f"  바이트 오프셋: {accessor.byteOffset}")
                print(f"  컴포넌트 타입: {accessor.componentType}")
                print(f"  요소 수: {accessor.count}")
                print(f"  요소 타입: {accessor.type}")
        
        return gltf
    
    except ImportError:
        print("pygltflib가 설치되어 있지 않습니다. pip install pygltflib 명령으로 설치하세요.")
        return None
    except Exception as e:
        print(f"GLB 파일 분석 중 오류 발생: {e}")
        return None

def extract_animations_from_pygltf(glb_path):
    """
    PyGLTFlib를 사용하여 GLB 파일에서 애니메이션 데이터 추출 - 개선된 버전
    """
    import pygltflib
    import numpy as np
    import os
    import struct
    import traceback
    
    try:
        # GLB 파일 로드
        gltf = pygltflib.GLTF2().load(glb_path)
        animations = {}
        joints = []
        
        if not gltf.animations:
            print("PyGLTF: 애니메이션 데이터가 없습니다.")
            return {}, []
        
        print(f"PyGLTF: {len(gltf.animations)}개의 애니메이션을 발견했습니다.")
        
        # 바이너리 데이터 접근 방식 변경
        binary_data = None
        
        # 1. PyGLTF의 binary_blob 사용 시도
        try:
            if hasattr(gltf, 'binary_blob') and gltf.binary_blob is not None:
                # Check if binary_blob is a method that needs to be called
                if callable(gltf.binary_blob):
                    binary_data = gltf.binary_blob()
                else:
                    binary_data = gltf.binary_blob
                print(f"binary_blob에서 {len(binary_data)} 바이트 추출")
            elif hasattr(gltf, 'get_binary_blob'):
                binary_data = gltf.get_binary_blob()
                print(f"get_binary_blob에서 {len(binary_data)} 바이트 추출")
        except Exception as e:
            print(f"binary_blob 접근 오류: {e}")
        
        # 2. 수동 추출 시도
        if binary_data is None:
            print("수동으로 바이너리 데이터 추출 시도...")
            try:
                with open(glb_path, 'rb') as f:
                    # GLB 헤더 확인
                    magic = f.read(4)
                    if magic != b'glTF':
                        print(f"유효하지 않은 GLB 파일입니다: {magic}")
                        return {}, []
                    
                    # 버전 및 길이 정보 읽기
                    version, length = struct.unpack('<II', f.read(8))
                    print(f"GLB 버전: {version}, 길이: {length}")
                    
                    # JSON 청크 길이 및 타입 읽기
                    json_length, json_type = struct.unpack('<II', f.read(8))
                    if json_type != 0x4E4F534A:  # 'JSON'
                        print(f"예상치 못한 청크 타입: {json_type}")
                        return {}, []
                    
                    # JSON 청크 건너뛰기
                    f.seek(json_length, 1)
                    
                    # 바이너리 청크 길이 및 타입 읽기
                    try:
                        bin_header = f.read(8)
                        if len(bin_header) == 8:
                            bin_length, bin_type = struct.unpack('<II', bin_header)
                            if bin_type == 0x004E4942:  # 'BIN\0'
                                binary_data = f.read(bin_length)
                                print(f"수동으로 {len(binary_data)} 바이트 추출")
                    except Exception as e:
                        print(f"바이너리 청크 읽기 오류: {e}")
            except Exception as e:
                print(f"파일 읽기 오류: {e}")
        
        if not binary_data:
            print("바이너리 데이터를 추출할 수 없습니다.")
            return {}, []
        
        # 바이너리 데이터 내용 미리보기
        print(f"바이너리 데이터 크기: {len(binary_data)} 바이트")
        print(f"바이너리 데이터 처음 20바이트: {binary_data[:20]}")
        
        # 스킨 정보에서 조인트 구조 추출
        if gltf.skins:
            print(f"스킨 {len(gltf.skins)}개 발견")
            skin = gltf.skins[0]  # 첫 번째 스킨 사용
            joint_indices = skin.joints
            
            print(f"관절 수: {len(joint_indices)}")
            print(f"관절 인덱스: {joint_indices[:5]}... (처음 5개)")
            
            # 인버스 바인드 매트릭스 가져오기 (있는 경우)
            if skin.inverseBindMatrices is not None:
                print(f"인버스 바인드 매트릭스 액세서: {skin.inverseBindMatrices}")
            
            # 각 관절에 대한 정보 추출 (처음 5개만 출력)
            for i, idx in enumerate(joint_indices[:5]):
                if (idx < len(gltf.nodes)):
                    node = gltf.nodes[idx]
                    print(f"관절 {i} (노드 {idx}): 이름={getattr(node, 'name', 'unnamed')}")
            
            # 관절 구조 생성
            for i, idx in enumerate(joint_indices):
                if idx < len(gltf.nodes):
                    node = gltf.nodes[idx]
                    
                    # 변환 행렬 계산
                    transform = np.eye(4)
                    
                    # 관절 정보 저장
                    joints.append({
                        'name': node.name if node.name else f"joint_{idx}",
                        'transform': transform.tolist(),
                        'node_index': idx  # 노드 인덱스 저장
                    })
            
            print(f"추출된 관절 수: {len(joints)}")
        
        # 애니메이션 처리
        for anim_idx, anim in enumerate(gltf.animations):
            print(f"\n애니메이션 {anim_idx} 처리 중...")
            channels = anim.channels
            samplers = anim.samplers
            
            print(f"채널 수: {len(channels)}, 샘플러 수: {len(samplers)}")
            
            # 각 노드별 애니메이션 데이터를 모으는 딕셔너리
            node_animations = {}
            
            # 타임라인 데이터 추출
            times = []
            if samplers and samplers[0].input < len(gltf.accessors):
                time_accessor = gltf.accessors[samplers[0].input]
                time_buffer_view = gltf.bufferViews[time_accessor.bufferView]
                
                # 시간 데이터 읽기
                time_offset = time_buffer_view.byteOffset + (time_accessor.byteOffset or 0)
                time_count = time_accessor.count
                time_stride = 4  # float32 = 4 bytes
                
                times = []
                for i in range(time_count):
                    pos = time_offset + i * time_stride
                    if pos + 4 <= len(binary_data):
                        time_value = struct.unpack_from('<f', binary_data, pos)[0]
                        times.append(time_value)
                    else:
                        print(f"시간 데이터 인덱스 {i}가 바이너리 데이터 범위를 벗어납니다.")
                        break
                
                print(f"추출된 시간 데이터: {len(times)}개")
                if times:
                    print(f"시간 범위: {min(times)} ~ {max(times)}")
            
            # 채널 데이터 처리 - 실제 애니메이션 데이터 추출
            for chan_idx, channel in enumerate(channels):
                target_node = channel.target.node
                target_path = channel.target.path  # translation, rotation, scale
                sampler_idx = channel.sampler
                
                if sampler_idx >= len(samplers):
                    continue
                
                sampler = samplers[sampler_idx]
                output_accessor_idx = sampler.output
                
                if output_accessor_idx >= len(gltf.accessors):
                    continue
                
                output_accessor = gltf.accessors[output_accessor_idx]
                output_buffer_view = gltf.bufferViews[output_accessor.bufferView]
                
                # 액세서 데이터 세부정보
                accessor_type = output_accessor.type
                component_type = output_accessor.componentType
                
                # 컴포넌트 타입에 따른 바이트 크기 및 언팩 포맷 결정
                if component_type == 5126:  # FLOAT
                    component_size = 4
                    unpack_format = '<f'
                elif component_type == 5121:  # UNSIGNED_BYTE
                    component_size = 1
                    unpack_format = '<B'
                else:
                    print(f"지원되지 않는 컴포넌트 타입: {component_type}")
                    continue
                
                # 데이터 타입에 따른 요소 수
                if accessor_type == 'SCALAR':
                    num_components = 1
                elif accessor_type == 'VEC2':
                    num_components = 2
                elif accessor_type == 'VEC3':
                    num_components = 3
                elif accessor_type == 'VEC4':
                    num_components = 4
                else:
                    print(f"지원되지 않는 액세서 타입: {accessor_type}")
                    continue
                
                # 데이터 추출
                output_offset = output_buffer_view.byteOffset + (output_accessor.byteOffset or 0)
                values = []
                
                for i in range(output_accessor.count):
                    value = []
                    for j in range(num_components):
                        pos = output_offset + i * (num_components * component_size) + j * component_size
                        if pos + component_size <= len(binary_data):
                            val = struct.unpack_from(unpack_format, binary_data, pos)[0]
                            value.append(val)
                        else:
                            print(f"데이터 인덱스 {i}가 바이너리 데이터 범위를 벗어납니다.")
                            break
                    
                    if len(value) == num_components:
                        if num_components == 1:
                            values.append(value[0])
                        else:
                            values.append(value)
                
                # 노드별 애니메이션 데이터 정리
                if target_node not in node_animations:
                    node_animations[target_node] = {}
                
                # 경로별 데이터 저장
                node_animations[target_node][target_path] = {
                    'interpolation': sampler.interpolation,
                    'values': values
                }
                
                print(f"노드 {target_node}, {target_path} 데이터: {len(values)}개 추출")
                if values:
                    if isinstance(values[0], list):
                        print(f"샘플 데이터: {values[0]}")
                    else:
                        print(f"샘플 데이터: {values[0]}")
            
            # 최종 애니메이션 데이터 저장
            animations[f"animation_{anim_idx}"] = {
                'frames': len(times) if times else 0,
                'times': times,
                'nodes': node_animations,
                'name': anim.name if hasattr(anim, 'name') and anim.name else f"animation_{anim_idx}"
            }
        
        # 애니메이션 데이터 요약 출력
        print("\n=== 애니메이션 데이터 요약 ===")
        for anim_name, anim_data in animations.items():
            print(f"애니메이션: {anim_name}")
            print(f"  프레임 수: {anim_data['frames']}")
            print(f"  노드 수: {len(anim_data['nodes'])}")
            
            # 노드 데이터 예시 (첫 번째 노드)
            if anim_data['nodes']:
                first_node = next(iter(anim_data['nodes']))
                print(f"  첫 번째 노드 {first_node}의 애니메이션 타입: {list(anim_data['nodes'][first_node].keys())}")
        
        return animations, joints
                    
    except Exception as e:
        print(f"애니메이션 추출 중 예외 발생: {e}")
        traceback.print_exc()
        return {}, []

def preview_sequence_with_open3d(pose_sequence, model, fps=30, duration=None):
    try:
        import open3d as o3d
        import torch
        import time
        import numpy as np
        
        if not pose_sequence or len(pose_sequence) == 0:
            print("미리볼 포즈 시퀀스가 없습니다.")
            return
        
        # 포즈 데이터의 구조 확인
        first_pose = pose_sequence[0]
        print(f"첫 번째 포즈 형태: {type(first_pose)}, 크기: {len(first_pose)}")
        
        # 포즈 데이터가 72개가 아니면 표준 크기로 조정
        if len(first_pose) != 72:
            print(f"경고: 포즈 데이터 크기가 72가 아닙니다({len(first_pose)}). 포즈 데이터를 조정합니다.")
            
            # 필요한 경우 데이터 변환
            adjusted_sequence = []
            for pose in pose_sequence:
                if len(pose) > 72:
                    # 크기가 너무 크면 자르기
                    adjusted_pose = pose[:72]
                else:
                    # 크기가 작으면 0으로 채우기
                    adjusted_pose = np.zeros(72)
                    adjusted_pose[:len(pose)] = pose
                adjusted_sequence.append(adjusted_pose)
            
            pose_sequence = adjusted_sequence
        
        # 재생할 프레임 수 결정
        total_frames = len(pose_sequence)
        if duration is not None:
            max_frames = int(fps * duration)
            frames_to_play = min(total_frames, max_frames)
        else:
            frames_to_play = total_frames
        
        print(f"포즈 시퀀스 미리보기 준비 중... ({frames_to_play}/{total_frames} 프레임)")
        
        # Open3D 시각화 준비
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="SMPL Animation Preview", width=800, height=800)
        
        # 첫 번째 프레임으로 메시 초기화 (에러 처리 추가)
        try:
            first_pose = pose_sequence[0]
            
            # 데이터 형식 확인 및 변환
            if isinstance(first_pose, list):
                first_pose = np.array(first_pose)
            
            global_orient = torch.tensor(first_pose[:3], dtype=torch.float32).unsqueeze(0)
            body_pose = torch.tensor(first_pose[3:], dtype=torch.float32).unsqueeze(0)
            
            # 차원 확인
            print(f"global_orient 형태: {global_orient.shape}")
            print(f"body_pose 형태: {body_pose.shape}")
            
            # SMPL 모델 출력 생성
            output = model(
                global_orient=global_orient,
                body_pose=body_pose,
                return_verts=True
            )
            print(f"SMPL 모델 출력 형태: {len(output.vertices.shape)}")
            vertices = output.vertices.detach().cpu().numpy()[0]
            faces = model.faces
            print(f"정점 수: {len(vertices)}, 면 수: {len(faces)}")
            print(f"정점 형태: {vertices.shape}, 면 형태: {faces.shape}")
            
            # Open3D 메시 객체 생성
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color([0.7, 0.7, 0.9])
            print("메시 생성 완료")
            # 메시를 시각화 도구에 추가
            vis.add_geometry(mesh)
            print("메시를 시각화 도구에 추가했습니다")
            
            # 카메라 위치 설정
            ctr = vis.get_view_control()
            ctr.set_zoom(0.8)
            
            print(f"미리보기 재생 중... (종료하려면 창을 닫으세요)")
            frame_time = 1.0 / fps
            
            # 애니메이션 재생
            for i in range(frames_to_play):
                pose = pose_sequence[i % total_frames]
                
                # 메시 업데이트
                global_orient = torch.tensor(pose[:3], dtype=torch.float32).unsqueeze(0)
                body_pose = torch.tensor(pose[3:], dtype=torch.float32).unsqueeze(0)
                
                output = model(
                    global_orient=global_orient,
                    body_pose=body_pose,
                    return_verts=True
                )
                vertices = output.vertices.detach().cpu().numpy()[0]
                
                # 메시 정점 업데이트
                mesh.vertices = o3d.utility.Vector3dVector(vertices)
                mesh.compute_vertex_normals()
                
                # 시각화 업데이트
                vis.update_geometry(mesh)
                vis.poll_events()
                vis.update_renderer()
                
                # 프레임 레이트 조절을 위한 대기
                time.sleep(frame_time)
            
            vis.destroy_window()
        
        except Exception as e:
            print(f"메시 생성 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            vis.destroy_window()
    
    except ImportError:
        print("Open3D 라이브러리가 설치되어 있지 않습니다.")
        print("설치하려면: pip install open3d")
    except Exception as e:
        print(f"미리보기 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

def validate_pose_data(pose_sequence):
    """포즈 데이터의 유효성 검사"""
    if not pose_sequence or len(pose_sequence) == 0:
        print("포즈 시퀀스가 비어 있습니다.")
        return False
    
    # 첫 번째 포즈 확인
    first_pose = pose_sequence[0]
    length = len(first_pose)
    
    # 포즈 길이 확인
    if length != 72:
        print(f"경고: SMPL은 일반적으로 72개의 포즈 파라미터를 사용하지만, 현재 {length}개입니다.")
    
    # NaN 또는 무한대 값 확인
    has_nan = any(np.isnan(first_pose))
    has_inf = any(np.isinf(first_pose))
    
    if has_nan or has_inf:
        print("경고: 포즈 데이터에 NaN 또는 무한대 값이 있습니다.")
        return False
    
    return True

# SMPL 본 이름 순서 (24개)
SMPL_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand"
]

# Mixamo → SMPL 관절 이름 매핑
MIXAMO_TO_SMPL = {
    "Hips": "pelvis",
    "Spine": "spine1",
    "Spine1": "spine2",
    "Spine2": "spine3",
    "LeftUpLeg": "left_hip",
    "RightUpLeg": "right_hip",
    "LeftLeg": "left_knee",
    "RightLeg": "right_knee",
    "LeftFoot": "left_ankle",
    "RightFoot": "right_ankle",
    "Neck": "neck",
    "Head": "head",
    "LeftShoulder": "left_shoulder",
    "RightShoulder": "right_shoulder",
    "LeftArm": "left_elbow",
    "RightArm": "right_elbow",
    "LeftForeArm": "left_wrist",
    "RightForeArm": "right_wrist",
    "LeftHand": "left_hand",
    "RightHand": "right_hand",
    "LeftHandIndex1": "left_hand",  # 추가 매핑
    "RightHandIndex1": "right_hand"  # 추가 매핑
}

def map_joints_to_smpl(joints, animations):
    """
    Mixamo/GLB 관절을 SMPL 관절 구조에 매핑
    
    Args:
        joints: 원본 관절 리스트
        animations: 추출된 애니메이션 정보 딕셔너리
    
    Returns:
        SMPL 형식에 맞는 관절 리스트
    """
    print("\n=== SMPL 관절 매핑 시작 ===")
    
    # 관절 이름 기반 매핑 딕셔너리 생성
    joint_name_to_index = {}
    for i, joint in enumerate(joints):
        name = joint.get('name', '')
        if name:
            joint_name_to_index[name] = i
    
    # 디버깅을 위해 모든 관절 이름 출력
    print("발견된 관절 이름들:")
    for name in joint_name_to_index.keys():
        print(f"  - {name}")
    
    # 관절 이름에서 Mixamo 패턴 찾기
    mixamo_pattern_matches = {}
    for joint_name in joint_name_to_index.keys():
        # Mixamo 관절 이름 패턴 확인
        for mixamo_name in MIXAMO_TO_SMPL.keys():
            if mixamo_name.lower() in joint_name.lower():
                mixamo_pattern_matches[mixamo_name] = joint_name
    
    print(f"발견된 Mixamo 관절 패턴: {len(mixamo_pattern_matches)}/{len(MIXAMO_TO_SMPL)}")
    
    # SMPL 관절 매핑 생성
    smpl_joints = []
    for i, joint_name in enumerate(SMPL_JOINT_NAMES):
        # 매핑된 Mixamo 관절 찾기
        found = False
        for mixamo_name, smpl_name in MIXAMO_TO_SMPL.items():
            if smpl_name == joint_name and mixamo_name in mixamo_pattern_matches:
                original_joint_name = mixamo_pattern_matches[mixamo_name]
                original_idx = joint_name_to_index.get(original_joint_name)
                
                if original_idx is not None:
                    # 원본 관절 데이터 복사
                    original_joint = joints[original_idx].copy()
                    original_joint['original_index'] = original_idx
                    original_joint['smpl_index'] = i
                    original_joint['smpl_name'] = joint_name
                    smpl_joints.append(original_joint)
                    found = True
                    print(f"매핑됨: {original_joint_name} → {joint_name} (인덱스: {original_idx} → {i})")
                    break
        
        # 매칭되는 관절이 없으면 기본 관절 생성
        if not found:
            default_joint = {
                'name': joint_name,
                'transform': np.eye(4).tolist(),
                'smpl_index': i,
                'smpl_name': joint_name
            }
            smpl_joints.append(default_joint)
            print(f"기본값 사용: {joint_name} (인덱스: {i})")
    
    print(f"SMPL 관절 매핑 완료: {len(smpl_joints)}개")
    
    # 매핑된 관절 수가 24개인지 확인
    if len(smpl_joints) != 24:
        print(f"경고: 매핑된 관절 수({len(smpl_joints)})가 예상된 SMPL 관절 수(24)와 다릅니다.")
    
    return smpl_joints

def process_animation_frames_with_mapping(animations, joints, fps=30):
    """
    애니메이션 데이터를 SMPL 포맷의 포즈 파라미터로 변환 (관절 매핑 사용)
    """
    print("\n=== 애니메이션 프레임 처리 시작 (매핑 사용) ===")
    
    # 먼저 SMPL 구조에 맞게 관절 매핑
    smpl_joints = map_joints_to_smpl(joints, animations)
    
    frame_poses = []
    
    if not animations:
        print("애니메이션 데이터가 없습니다. 정적 포즈를 사용합니다.")
        # 정적 포즈 생성 (SMPL 형식)
        pose_params = np.zeros(72)  # 24개 관절 × 3 회전 매개변수
        
        # 기본 포즈 정보 추출 (매핑된 관절에서)
        for joint in smpl_joints:
            smpl_idx = joint.get('smpl_index', -1)
            if smpl_idx >= 0 and smpl_idx < 24:
                try:
                    transform = np.array(joint['transform']).reshape(4, 4)
                    rotation = R.from_matrix(transform[:3, :3])
                    rot_angles = rotation.as_rotvec()
                    pose_params[smpl_idx*3:(smpl_idx+1)*3] = rot_angles
                except Exception as e:
                    print(f"관절 {smpl_idx} ({joint.get('smpl_name', '')}) 처리 중 오류: {e}")
        
        # 정적 포즈를 여러 프레임에 복제
        frame_poses = [pose_params.copy() for _ in range(30)]
        return frame_poses
    
    # 첫 번째 애니메이션 사용
    anim_key = next(iter(animations))
    animation = animations[anim_key]
    
    print(f"애니메이션 '{animation.get('name', anim_key)}' 처리 중 ({animation.get('frames', 0)} 프레임)")
    
    # 시간 정보와 노드 애니메이션 데이터
    times = animation.get('times', [])
    node_animations = animation.get('nodes', {})
    
    if not times:
        print("시간 데이터가 없습니다.")
        return []
    
    # 노드 인덱스와 SMPL 관절 인덱스 매핑 생성
    node_to_smpl_joint = {}
    for joint in smpl_joints:
        original_idx = joint.get('original_index')
        if original_idx is not None and 'node_index' in joints[original_idx]:
            node_idx = joints[original_idx]['node_index']
            smpl_idx = joint.get('smpl_index')
            if node_idx is not None and smpl_idx is not None:
                node_to_smpl_joint[node_idx] = smpl_idx
    
    print(f"노드 → SMPL 관절 매핑: {len(node_to_smpl_joint)}개")
    
    # 각 프레임에 대해 처리
    for frame_idx in range(animation.get('frames', 0)):
        # SMPL 포즈 파라미터 초기화 (72차원으로 고정)
        pose_params = np.zeros(72)
        
        # 각 노드에 대한 애니메이션 데이터 적용
        for node_idx, node_data in node_animations.items():
            # 이 노드가 SMPL 관절에 매핑되는지 확인
            if node_idx in node_to_smpl_joint:
                smpl_idx = node_to_smpl_joint[node_idx]
                
                # 기본 변환 매트릭스 (정적 포즈)
                transform = np.eye(4)
                for joint in smpl_joints:
                    if joint.get('smpl_index') == smpl_idx:
                        transform = np.array(joint['transform']).reshape(4, 4)
                        break
                
                # 애니메이션 데이터 적용
                # 회전 데이터
                if 'rotation' in node_data:
                    rot_data = node_data['rotation']
                    rot_values = rot_data.get('values', [])
                    
                    if frame_idx < len(rot_values):
                        # 쿼터니언 회전 적용
                        quat = rot_values[frame_idx]
                        rotation = R.from_quat(quat)
                        transform[:3, :3] = rotation.as_matrix()
                
                # 이동 데이터 (필요한 경우)
                if 'translation' in node_data and smpl_idx == 0:  # pelvis(루트) 관절에만 적용
                    trans_data = node_data['translation']
                    trans_values = trans_data.get('values', [])
                    
                    if frame_idx < len(trans_values):
                        transform[:3, 3] = trans_values[frame_idx]
                
                # 회전 정보 추출 및 SMPL 포즈 매개변수에 적용
                rotation = R.from_matrix(transform[:3, :3])
                rot_angles = rotation.as_rotvec()
                pose_params[smpl_idx*3:(smpl_idx+1)*3] = rot_angles
        
        # 완성된 프레임 포즈 추가
        frame_poses.append(pose_params)
    
    print(f"포즈 시퀀스 생성 완료: {len(frame_poses)} 프레임")
    return frame_poses

def main():
    # Input file path
    glb_path = "/Users/jihyunlee/projects/ml_google_2nd_project/converted_20250409_102943/Walking Backwards.glb"
    
    # Output file path
    output_dir = os.path.dirname(glb_path)
    output_filename = os.path.splitext(os.path.basename(glb_path))[0] + "_smpl.obj"
    output_path = os.path.join(output_dir, output_filename)
    
    print("==== GLB 파일 애니메이션 추출 및 SMPL 변환 ====")
    
    # GLB 파일 로드 및 분석
    scene = load_glb_file(glb_path)
    if not scene:
        print("GLB 파일 로드에 실패했습니다.")
        return
    
    print("\n1. GLB 파일 구조 분석 중...")
    gltf_data = analyze_glb_with_pygltflib(glb_path)
    
    print("\n2. PyGLTF로 애니메이션 데이터 추출 중...")
    animations, joints = extract_animations_from_pygltf(glb_path)
    
    # 애니메이션이 없는 경우 디버그 출력
    if not animations:
        print("!! 애니메이션 데이터를 추출하지 못했습니다 !!")
        print("GLTF 애니메이션 수:", len(gltf_data.animations) if gltf_data and hasattr(gltf_data, 'animations') else 0)
        print("GLTF 스킨 수:", len(gltf_data.skins) if gltf_data and hasattr(gltf_data, 'skins') else 0)
    else:
        # 애니메이션 데이터 디버깅
        anim_key = next(iter(animations))
        print(f"첫 번째 애니메이션: {anim_key}")
        print(f"프레임 수: {animations[anim_key]['frames']}")
        print(f"노드 수: {len(animations[anim_key]['nodes'])}")
        
        # 첫 번째 노드와 프레임의 회전 데이터 출력
        if animations[anim_key]['nodes']:
            first_node = next(iter(animations[anim_key]['nodes']))
            node_data = animations[anim_key]['nodes'][first_node]
            print(f"노드 {first_node}의 애니메이션 타입: {list(node_data.keys())}")
            
            if 'rotation' in node_data:
                rot_values = node_data['rotation']['values']
                print(f"첫 프레임 회전: {rot_values[0] if rot_values else 'None'}")
                print(f"마지막 프레임 회전: {rot_values[-1] if rot_values else 'None'}")
    
    # Trimesh로 추출한 관절 정보가 없을 경우
    if not joints:
        print("PyGLTF에서 관절 정보를 추출하지 못했습니다.")
        # Trimesh를 통한 관절 추출 시도
        trimesh_joints, _ = extract_skeleton_and_animation(scene)
        
        if trimesh_joints:
            print(f"Trimesh에서 {len(trimesh_joints)}개 관절을 발견했습니다.")
            joints = trimesh_joints
        else:
            print("모든 방법으로 관절을 찾을 수 없습니다. 기본 관절을 생성합니다.")
            # 24개의 기본 관절 생성 (SMPL 기본 구조)
            joints = [{
                'name': f"joint_{i}",
                'transform': np.eye(4).tolist()
            } for i in range(24)]
    
    print(f"\n3. 관절 {len(joints)}개, 애니메이션 {len(animations)}개 발견")
    
    # print("\n4. 애니메이션 프레임 처리 중...")
    # pose_sequence = process_animation_frames(animations, joints)
    # print(f"   총 {len(pose_sequence)}개 프레임 생성됨")
    print("\n4. 애니메이션 프레임 처리 중...")
    # 기존 함수 대신 매핑을 활용한 함수 사용
    pose_sequence = process_animation_frames_with_mapping(animations, joints)
    print(f"   총 {len(pose_sequence)}개 프레임 생성됨")
       
    # 포즈 데이터 검사
    if pose_sequence:
        first_pose = pose_sequence[0]
        last_pose = pose_sequence[-1]
        
        print("\n포즈 데이터 샘플:")
        print(f"첫 프레임: min={np.min(first_pose):.4f}, max={np.max(first_pose):.4f}, 평균={np.mean(first_pose):.4f}")
        print(f"마지막 프레임: min={np.min(last_pose):.4f}, max={np.max(last_pose):.4f}, 평균={np.mean(last_pose):.4f}")
        
        # 포즈 차이 계산
        if len(pose_sequence) > 1:
            pose_diff = np.abs(last_pose - first_pose).sum()
            print(f"첫 프레임과 마지막 프레임의 차이 합: {pose_diff:.4f}")
            if pose_diff < 0.01:
                print("경고: 포즈 차이가 매우 작습니다. 애니메이션이 제대로 추출되지 않았을 수 있습니다.")
    
    print("\n5. SMPL 모델로 변환 중...")
    # 매핑된 포즈 시퀀스를 전달하여 변환
    model = convert_to_smpl_sequence(joints, animations, output_path, pose_sequence)    

    print("\n6. 변환 완료!")
    print(f"   SMPL 메시: {output_path}")
    print(f"   패러미터: {output_path.replace('.obj', '_params.json')}")
    print(f"   모션클립: {output_path.replace('.obj', '_motionclip.npy')}")
    
    print("\n7. Open3D를 사용하여 애니메이션 미리보기 중...")
    # 포즈 시퀀스 미리보기 (5초 동안 재생)
    try:
        preview_sequence_with_open3d(pose_sequence, model, fps=30, duration=5)
    except Exception as e:
        print(f"미리보기를 실행할 수 없습니다: {e}")

if __name__ == "__main__":
    main()