"""
포즈 데이터 추출 및 처리 관련 모듈
"""

import os
import tempfile
import numpy as np
import cv2
import mediapipe as mp
from pytube import YouTube
import yt_dlp
import uuid
from pathlib import Path
import trimesh
from trimesh.exchange.gltf import export_glb
import json
from model.mediapipe import mediapipe_bone_list, mp_landmarks_info

# 포즈 데이터를 SMPL 형식으로 변환
def convert_to_smpl_format(pose_array):
    """
    MediaPipe 포즈 데이터를 SMPL 형식으로 변환
    
    Args:
        pose_array: MediaPipe 포즈 배열 (N x 33 x 3)
    
    Returns:
        SMPL 형식의 포즈 배열
    """
    # 디버깅: MediaPipe 랜드마크 정보 출력
    print(f"\n===== MediaPipe 랜드마크 정보 =====")
    print(f"MediaPipe에서 추출한 랜드마크 수: {pose_array.shape[1]}")
    print(f"프레임 수: {pose_array.shape[0]}")
    
    # 첫 번째 프레임의 주요 랜드마크 위치 출력
    if pose_array.shape[0] > 0:
        print("\n첫 번째 프레임의 주요 랜드마크 위치:")
        for idx, name in mp_landmarks_info.items():
            if idx < pose_array.shape[1]:
                print(f"  {idx}: {name} - {pose_array[0, idx]}")
    
    # SMPL은 24개 조인트를 사용하므로 매핑 필요
    smpl_data = np.zeros((pose_array.shape[0], 24, 3), dtype=np.float32)
    
   
    # MediaPipe에서 SMPL로의 올바른 매핑 (신체 부위 간 연관성 유지)
    mp_to_smpl_mapping = {
        # 중심 및 척추
        23: 0,    # left_hip -> pelvis (왼쪽 골반 사용)
        24: 0,    # right_hip -> pelvis (오른쪽 골반도 pelvis 계산에 사용)
        0: 17,    # nose -> head (코는 머리에 매핑)
        
        # 골반 및 척추
        23: 1,    # left_hip -> left_hip (왼쪽 골반)
        24: 2,    # right_hip -> right_hip (오른쪽 골반)
        
        # 다리
        25: 4,    # left_knee -> left_knee (왼쪽 무릎)
        26: 5,    # right_knee -> right_knee (오른쪽 무릎)
        27: 7,    # left_ankle -> left_ankle (왼쪽 발목)
        28: 8,    # right_ankle -> right_ankle (오른쪽 발목)
        31: 22,   # left_foot_index -> left_toe (왼쪽 발가락)
        32: 23,   # right_foot_index -> right_toe (오른쪽 발가락)
        
        # 팔
        11: 9,    # left_shoulder -> left_shoulder (왼쪽 어깨)
        12: 10,   # right_shoulder -> right_shoulder (오른쪽 어깨)
        13: 12,   # left_elbow -> left_elbow (왼쪽 팔꿈치) 
        14: 13,   # right_elbow -> right_elbow (오른쪽 팔꿈치)
        15: 15,   # left_wrist -> left_wrist (왼쪽 손목)
        16: 16,   # right_wrist -> right_wrist (오른쪽 손목)
        19: 18,   # left_index -> left_hand (왼손 검지를 왼손으로)
        20: 19,   # right_index -> right_hand (오른손 검지를 오른손으로)
    }
    
    # 디버깅: MediaPipe -> SMPL 매핑 정보 출력
    print("\n===== MediaPipe -> SMPL 매핑 정보 =====")
    print(f"SMPL 관절 수: {len(smpl_joint_names)}")
    print("매핑된 관절:")
    for mp_idx, smpl_idx in mp_to_smpl_mapping.items():
        mp_name = mp_landmarks_info.get(mp_idx, f"Unknown_{mp_idx}")
        smpl_name = smpl_joint_names[smpl_idx]
        print(f"  MediaPipe {mp_idx} ({mp_name}) -> SMPL {smpl_idx} ({smpl_name})")
    
    # 매핑되지 않은 MediaPipe 랜드마크 출력
    unmapped_mp = [idx for idx in range(min(33, pose_array.shape[1])) if idx not in mp_to_smpl_mapping]
    print("\n매핑되지 않은 MediaPipe 랜드마크:")
    for idx in unmapped_mp:
        mp_name = mp_landmarks_info.get(idx, f"Unknown_{idx}")
        print(f"  {idx}: {mp_name}")
    
    # 매핑되지 않은 SMPL 관절 출력
    mapped_smpl = set(mp_to_smpl_mapping.values())
    unmapped_smpl = [idx for idx in range(len(smpl_joint_names)) if idx not in mapped_smpl]
    print("\n매핑되지 않은 SMPL 관절 (보간 필요):")
    for idx in unmapped_smpl:
        print(f"  {idx}: {smpl_joint_names[idx]}")
    
    # 프레임별 처리
    for frame in range(pose_array.shape[0]):
        frame_data = pose_array[frame]
        
        # 기본 매핑 적용 (직접 연결)
        for mp_idx, smpl_idx in mp_to_smpl_mapping.items():
            if mp_idx < pose_array.shape[1]:
                # 왼쪽/오른쪽 골반의 경우 pelvis 위치 계산에 사용
                if mp_idx in [23, 24] and smpl_idx == 0:
                    continue  # pelvis는 나중에 별도로 계산
                else:
                    smpl_data[frame, smpl_idx] = frame_data[mp_idx]
        
        # pelvis (골반 중심) 위치 계산 - 양쪽 골반의 중간점 사용
        if 23 < frame_data.shape[0] and 24 < frame_data.shape[0]:
            pelvis_pos = (frame_data[23] + frame_data[24]) / 2
            smpl_data[frame, 0] = pelvis_pos
        
        # 보간이 필요한 관절 처리
        
        # spine1 (척추 하단)
        if 0 in mapped_smpl and 9 in mapped_smpl and 10 in mapped_smpl:
            # pelvis와 어깨 중간점 사이의 1/3 지점
            shoulders_middle = (smpl_data[frame, 9] + smpl_data[frame, 10]) / 2
            smpl_data[frame, 3] = smpl_data[frame, 0] + (shoulders_middle - smpl_data[frame, 0]) * 0.33
        
        # spine2 (척추 중간)
        if 3 in mapped_smpl and 9 in mapped_smpl and 10 in mapped_smpl:
            # spine1과 어깨 중간점 사이의 1/2 지점
            shoulders_middle = (smpl_data[frame, 9] + smpl_data[frame, 10]) / 2
            smpl_data[frame, 6] = smpl_data[frame, 3] + (shoulders_middle - smpl_data[frame, 3]) * 0.5
        
        # spine3 (척추 상단)
        if 9 in mapped_smpl and 10 in mapped_smpl:
            # 어깨 중간점
            smpl_data[frame, 11] = (smpl_data[frame, 9] + smpl_data[frame, 10]) / 2
        
        # neck (목)
        if 11 in mapped_smpl and 17 in mapped_smpl:
            # spine3와 head 사이 중간에 약간 더 head쪽으로
            smpl_data[frame, 14] = smpl_data[frame, 11] + (smpl_data[frame, 17] - smpl_data[frame, 11]) * 0.7
        
        # 발(foot) 위치 (발목과 발가락 사이)
        if 7 in mapped_smpl and 22 in mapped_smpl:
            smpl_data[frame, 20] = smpl_data[frame, 7] + (smpl_data[frame, 22] - smpl_data[frame, 7]) * 0.5
        if 8 in mapped_smpl and 23 in mapped_smpl:
            smpl_data[frame, 21] = smpl_data[frame, 8] + (smpl_data[frame, 23] - smpl_data[frame, 8]) * 0.5
    
    # 디버깅: 첫 프레임의 SMPL 관절 위치 출력
    if pose_array.shape[0] > 0:
        print("\n첫 번째 프레임의 SMPL 관절 위치:")
        for i, name in enumerate(smpl_joint_names):
            print(f"  {i}: {name} - {smpl_data[0, i]}")
    
    print("\nSMPL 변환 완료")
    return smpl_data

def convert_smpl_to_glb(smpl_pose_array, output_path):
    """
    Convert SMPL pose data to GLB file format with animation
    
    Args:
        smpl_pose_array: SMPL pose array (N x 24 x 3)
        output_path: Path to save the GLB file
    
    Returns:
        Path to the saved GLB file, and binary GLB data
    """
    # 디버깅: SMPL 데이터 정보 출력
    print(f"\n===== SMPL -> GLB 변환 정보 =====")
    print(f"SMPL 포즈 배열 크기: {smpl_pose_array.shape}")
    print(f"프레임 수: {smpl_pose_array.shape[0]}")
    print(f"관절 수: {smpl_pose_array.shape[1]}")
    
    # SMPL 관절 이름 (참고용)
    smpl_joint_names = [
        "pelvis", "left_hip", "right_hip", "spine1", 
        "left_knee", "right_knee", "spine2", 
        "left_ankle", "right_ankle", "left_shoulder", 
        "right_shoulder", "spine3", "left_elbow", 
        "right_elbow", "neck", "left_wrist", 
        "right_wrist", "head", "left_hand", 
        "right_hand", "left_foot", "right_foot", 
        "left_toe", "right_toe"
    ]
    
    # Create a simple skeleton mesh from the SMPL data
    vertices = []
    faces = []
    
    # 관절 연결을 올바른 계층 구조로 수정
    joint_connections = [
        # 척추 계층 - 이름이 같은 관절끼리 연결
        (0, 3),    # pelvis -> spine1 (인덱스로 직접 연결)
        (3, 6),    # spine1 -> spine2
        (6, 11),   # spine2 -> spine3
        (11, 14),  # spine3 -> neck
        (14, 17),  # neck -> head
        
        # 왼쪽 팔 계층 - 왼쪽 어깨는 척추3이 아닌 어깨에 연결
        (6, 9),    # spine2 -> left_shoulder (spine3 대신 spine2에 연결)
        (9, 12),   # left_shoulder -> left_elbow
        (12, 15),  # left_elbow -> left_wrist
        (15, 18),  # left_wrist -> left_hand
        
        # 오른쪽 팔 계층
        (6, 10),   # spine2 -> right_shoulder (spine3 대신 spine2에 연결)
        (10, 13),  # right_shoulder -> right_elbow
        (13, 16),  # right_elbow -> right_wrist
        (16, 19),  # right_wrist -> right_hand
        
        # 왼쪽 다리 계층
        (0, 1),    # pelvis -> left_hip
        (1, 4),    # left_hip -> left_knee
        (4, 7),    # left_knee -> left_ankle
        (7, 20),   # left_ankle -> left_foot
        (20, 22),  # left_foot -> left_toe
        
        # 오른쪽 다리 계층
        (0, 2),    # pelvis -> right_hip
        (2, 5),    # right_hip -> right_knee
        (5, 8),    # right_knee -> right_ankle
        (8, 21),   # right_ankle -> right_foot
        (21, 23)   # right_foot -> right_toe
    ]
    
    # 디버깅: 본 연결 정보 출력
    print("\n관절 연결 정보 (GLB 생성용):")
    for start_idx, end_idx in joint_connections:
        start_name = smpl_joint_names[start_idx] if start_idx < len(smpl_joint_names) else f"joint_{start_idx}"
        end_name = smpl_joint_names[end_idx] if end_idx < len(smpl_joint_names) else f"joint_{end_idx}"
        print(f"  연결: {start_idx}({start_name}) -> {end_idx}({end_name})")
    
    # Get the number of frames
    num_frames = smpl_pose_array.shape[0]
    
    # Use the first frame for the base mesh
    frame_idx = 0
    joints = smpl_pose_array[frame_idx]
    
    # Scale the joints for better visualization
    scale_factor = 1.0
    joints = joints * scale_factor
    
    # 디버깅: 스케일 조정 정보
    print(f"\n관절 시각화를 위한 스케일 팩터: {scale_factor}")
    
    # Create a mesh from joints
    for i, joint in enumerate(joints):
        vertices.append(joint)
        joint_name = smpl_joint_names[i] if i < len(smpl_joint_names) else f"joint_{i}"
        print(f"  {i}: {joint_name} - 위치: {joint}")
    
    # Create edges between connected joints
    for start_idx, end_idx in joint_connections:
        if start_idx < len(vertices) and end_idx < len(vertices):
            # 여기서 faces를 추가할 수 있으나, 간단한 구현을 위해 생략
            pass
    
    # Create a simple mesh from the vertices
    mesh = trimesh.Trimesh(vertices=vertices)
    print(f"\n메시 생성 완료: {len(vertices)} 정점")
    
    # Create a scene with the mesh
    scene = trimesh.Scene()
    geometry = scene.add_geometry(mesh)
    
    # Add animation data as mesh extras
    animation_frames = []
    
    # GLTF/GLB에서의 애니메이션 시간 단위 (초)
    frame_time = 1.0 / 30.0  # 30 FPS 가정
    
    # 애니메이션 트랙 생성
    time_points = []
    translation_keyframes = []
    
    # 모든 프레임에 대한 변환 데이터 생성
    print(f"\n애니메이션 키프레임 생성 ({num_frames} 프레임):")
    for i in range(num_frames):
        time_points.append(i * frame_time)
        # 각 관절 위치를 변환 데이터로 사용
        transforms = []
        for joint_idx in range(smpl_pose_array.shape[1]):
            if joint_idx < len(vertices):
                transforms.append(smpl_pose_array[i, joint_idx].tolist())
        translation_keyframes.append(transforms)
        
        # 몇 개의 샘플 프레임만 출력 (첫 프레임, 중간 프레임, 마지막 프레임)
        if i == 0 or i == num_frames // 2 or i == num_frames - 1:
            print(f"  프레임 {i} 시간: {i * frame_time:.2f}초")
            for j in range(min(3, len(vertices))):  # 처음 3개 관절만 출력
                joint_name = smpl_joint_names[j] if j < len(smpl_joint_names) else f"joint_{j}"
                print(f"    관절 {j} ({joint_name}): {smpl_pose_array[i, j]}")
            print(f"    ... 외 {len(vertices) - 3} 개 관절")
    
    # GLTF에서 지원하는 애니메이션 형식으로 데이터 구성
    animation = {
        "name": "PoseAnimation",
        "fps": 30,
        "frames": num_frames,
        "timepoints": time_points,
        "translations": translation_keyframes,
        "joint_names": [f"joint_{i}" for i in range(len(vertices))]
    }
    
    # 메시 메타데이터로 애니메이션 정보 추가
    mesh.metadata = {"animation": animation}
    
    # GLB 파일로 저장
    glb_path = os.path.join(output_path, "animation.glb")
    
    # scene의 메타데이터에도 애니메이션 정보 추가
    scene.metadata = {"animation": animation}
    
    # 애니메이션이 포함된 GLB 내보내기
    print(f"\nGLB 파일 내보내기 시작...")
    glb_data = export_glb(scene)
    
    # 파일에 직접 저장
    with open(glb_path, 'wb') as f:
        f.write(glb_data)
    
    print(f"GLB 파일 내보내기 완료: {glb_path}")
    print(f"애니메이션 포함: {num_frames} 프레임, {len(vertices)} 관절")
    
    return glb_path, glb_data


def extract_pose_from_video(url, uploaded_file, start, end, model, fps, smooth, main_only, name, output_dir):
    """
    영상에서 포즈 데이터 추출 함수
    
    Args:
        url: 유튜브/틱톡 URL
        uploaded_file: 직접 업로드한 영상 파일 경로
        start: 시작 시간(초)
        end: 종료 시간(초)
        model: 포즈 추출 모델("MediaPipe", "VIBE", "HumanMotion")
        fps: 목표 프레임 레이트
        smooth: 스무딩 강도
        main_only: 메인 인물만 추출할지 여부
        name: 데이터셋 이름
        output_dir: 출력 디렉토리 경로
        
    Returns:
        원본 영상 경로, 포즈 처리 상태 정보
    """
    # 소스 영상 가져오기
    video_path = None
    
    try:
        # URL 또는 업로드된 파일에서 영상 가져오기
        if url:
            # URL에서 영상 다운로드 로직
            temp_dir = tempfile.mkdtemp()
            if "youtube.com" in url or "youtu.be" in url:
                yt = YouTube(url)
                video_path = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download(temp_dir)
            elif "tiktok.com" in url:
                # yt_dlp 사용하여 틱톡 영상 다운로드
                ydl_opts = {'outtmpl': os.path.join(temp_dir, 'video.mp4')}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                video_path = os.path.join(temp_dir, 'video.mp4')
        elif uploaded_file:
            video_path = uploaded_file
        else:
            return None, {"상태": "오류", "원인": "영상 URL 또는 파일을 제공해야 합니다"}
        
        # 영상 로드
        cap = cv2.VideoCapture(video_path)
        
        # 처리할 범위 설정
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 시작 및 종료 프레임 계산
        start_frame = max(0, int(start * video_fps))
        end_frame = min(total_frames, int(end * video_fps) if end > 0 else total_frames)
        
        # 추출할 프레임 계산 (원본 fps와 목표 fps 비율)
        extract_ratio = video_fps / fps
        
        # 포즈 추출 진행 상태
        progress = {"처리된 프레임": 0, "추출된 포즈": 0}
        
        # 포즈 데이터 저장 배열 (관절 좌표)
        pose_data = []
        
        # 시작 프레임으로 이동
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # 모델에 따라 포즈 추출 객체 생성
        if model == "MediaPipe":
            pose_data = extract_pose_with_mediapipe(
                cap, start_frame, end_frame, extract_ratio, progress, smooth
            )
        elif model == "VIBE":
            # 향후 VIBE 모델 구현
            pass
        elif model == "HumanMotion":
            # 향후 HumanMotion 모델 구현
            pass
        
        # 영상 캡처 해제
        cap.release()
        
        # 포즈 데이터를 배열로 변환
        pose_array = np.array(pose_data, dtype=np.float32)

        # 포즈 데이터를 SMPL 형식으로 변환
        smpl_pose_array = convert_to_smpl_format(pose_array)

        # Generate unique filename for GLB output
        glb_filename = f"{name}_{uuid.uuid4().hex[:8]}.glb"
        glb_path = os.path.join(output_dir, "datasets", name, glb_filename)

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(glb_path), exist_ok=True)

        # Convert SMPL pose array to GLB - 이제 함수는 파일 경로와 바이너리 데이터를 함께 반환합니다
        glb_path_resp, glb_data = convert_smpl_to_glb(smpl_pose_array, os.path.dirname(glb_path))
        
        # 현재 작업 디렉토리에 animation.glb 파일도 저장
        with open("animation.glb", "wb") as f:
            f.write(glb_data)
        print(f'GLB file saved at: {glb_path_resp} and copied to animation.glb')

        # SMPL 포즈 데이터를 NPY 파일로 저장할 경로
        smpl_pose_file = os.path.join(output_dir, "datasets", name, f"{name}_smpl.npy")

        # SMPL 포즈 데이터 저장
        np.save(smpl_pose_file, smpl_pose_array)

        # 원본 포즈 데이터도 함께 저장 (선택 사항)
        original_pose_file = os.path.join(output_dir, "datasets", name, f"{name}_original.npy")
        np.save(original_pose_file, pose_array)

        
        # 출력 디렉토리 생성
        dataset_dir = os.path.join(output_dir, "datasets", name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # 처리 결과
        return (
            video_path,  # 원본 영상 경로
            {
                "상태": "완료",
                "데이터셋 이름": name,
                "처리된 프레임": progress["처리된 프레임"],
                "추출된 포즈": progress["추출된 포즈"],
                "저장 위치": dataset_dir
            }
        )
    
    except Exception as e:
        return video_path, {"상태": "오류", "원인": str(e)}

def extract_pose_with_mediapipe(cap, start_frame, end_frame, extract_ratio, progress, smooth):
    """
    MediaPipe를 사용하여 영상에서 포즈 데이터 추출
    
    Args:
        cap: 영상 캡처 객체
        start_frame: 시작 프레임
        end_frame: 종료 프레임
        extract_ratio: 추출 비율
        progress: 진행 상태 정보
        smooth: 스무딩 강도 (0-1)
    
    Returns:
        포즈 데이터 리스트
    """
    mp_pose = mp.solutions.pose
    pose_data = []
    
    # MediaPipe 포즈 객체 생성
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=smooth > 0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        frame_idx = 0
        frame_count = 0
        
        while cap.isOpened() and frame_idx < (end_frame - start_frame):
            success, image = cap.read()
            if not success:
                break
            
            # 목표 fps에 맞게 프레임 추출
            if frame_idx % extract_ratio < 1.0:
                # 프레임 전처리
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 포즈 감지
                results = pose.process(image_rgb)
                
                # 포즈 랜드마크가 감지된 경우
                if results.pose_landmarks:
                    # 랜드마크 좌표 저장 (정규화된 좌표)
                    landmarks_data = []
                    for landmark in results.pose_landmarks.landmark:
                        landmarks_data.append([landmark.x, landmark.y, landmark.z])
                    
                    pose_data.append(landmarks_data)
                    progress["추출된 포즈"] += 1
                
                frame_count += 1
            
            frame_idx += 1
            progress["처리된 프레임"] = frame_idx
    
    return pose_data