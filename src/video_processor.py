"""
비디오 처리 및 포즈 추출 모듈
"""

import os
import tempfile
import numpy as np
import cv2
import mediapipe as mp
from pytube import YouTube
import yt_dlp
import time

def download_video_from_url(url):
    """
    유튜브 또는 틱톡 URL에서 비디오를 다운로드합니다.
    
    Args:
        url: 비디오 URL
    
    Returns:
        다운로드된 비디오 파일 경로, 임시 디렉토리 경로
    """
    try:
        temp_dir = tempfile.mkdtemp()
        video_path = None
        
        if not url:
            return None, temp_dir
        
        if "youtube.com" in url or "youtu.be" in url:
            # 유튜브 영상 다운로드
            yt = YouTube(url)
            video_path = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download(temp_dir)
            print(f"유튜브 영상 다운로드 완료: {video_path}")
        elif "tiktok.com" in url:
            # 틱톡 영상 다운로드
            ydl_opts = {'outtmpl': os.path.join(temp_dir, 'video.mp4')}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            video_path = os.path.join(temp_dir, 'video.mp4')
            print(f"틱톡 영상 다운로드 완료: {video_path}")
        
        return video_path, temp_dir
    except Exception as e:
        print(f"영상 다운로드 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None, temp_dir

def extract_video_metadata(video_path):
    """
    비디오 파일에서 메타데이터를 추출합니다.
    
    Args:
        video_path: 비디오 파일 경로
    
    Returns:
        (총 프레임 수, fps, 너비, 높이, 총 시간(초))
    """
    if not video_path or not os.path.exists(video_path):
        return None, None, None, None, None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, None, None, None
    
    # 비디오 정보 추출
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    cap.release()
    return total_frames, fps, width, height, duration

def extract_pose_from_video(video_path, start_time=0, end_time=None, 
                          target_fps=30, smooth=0.3, main_only=True, 
                          pose_model="MediaPipe"):
    """
    비디오에서 포즈 데이터를 추출합니다.
    
    Args:
        video_path: 비디오 파일 경로
        start_time: 시작 시간(초)
        end_time: 종료 시간(초), None이면 영상 끝까지
        target_fps: 추출할 프레임 레이트
        smooth: 스무딩 강도 (0~1)
        main_only: 메인 인물만 추출할지 여부
        pose_model: 사용할 포즈 추출 모델 ("MediaPipe", "VIBE", "HumanMotion")
    
    Returns:
        포즈 데이터 배열, 처리 결과 정보
    """
    if not video_path or not os.path.exists(video_path):
        return None, {"상태": "오류", "원인": "유효한 비디오 파일이 아닙니다"}
    
    # 비디오 열기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, {"상태": "오류", "원인": "비디오 파일을 열 수 없습니다"}
    
    # 비디오 정보 추출
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 시작 및 종료 프레임 계산
    start_frame = max(0, int(start_time * video_fps))
    if end_time is None:
        end_frame = total_frames
    else:
        end_frame = min(total_frames, int(end_time * video_fps))
    
    # 추출할 프레임 간격 계산 (원본 fps와 목표 fps 비율)
    if target_fps >= video_fps:
        # 목표 fps가 원본보다 높으면 모든 프레임 사용
        extract_interval = 1
        effective_fps = video_fps
    else:
        # 목표 fps가 원본보다 낮으면 간격을 두고 추출
        extract_interval = video_fps / target_fps
        effective_fps = target_fps
    
    # 포즈 추출 초기화
    pose_data = []
    progress = {"처리된 프레임": 0, "추출된 포즈": 0}
    start_time = time.time()
    
    # MediaPipe 포즈 추출기 초기화
    if pose_model == "MediaPipe":
        mp_pose = mp.solutions.pose
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=smooth > 0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose_detector:
            # 시작 프레임으로 이동
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frame_idx = 0
            extract_count = 0
            
            while cap.isOpened() and frame_idx < (end_frame - start_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 목표 fps에 맞게 프레임 추출
                if frame_idx % extract_interval < 1.0:
                    # 이미지 전처리
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # 포즈 감지
                    results = pose_detector.process(rgb_frame)
                    
                    # 포즈 검출되면 저장
                    if results.pose_landmarks:
                        landmarks_data = []
                        for landmark in results.pose_landmarks.landmark:
                            # 정규화된 좌표를 사용 (0~1 범위)
                            landmarks_data.append([landmark.x, landmark.y, landmark.z])
                        
                        pose_data.append(landmarks_data)
                        progress["추출된 포즈"] += 1
                    
                    extract_count += 1
                    
                    # 진행 상황 계산
                    elapsed_time = time.time() - start_time
                    if elapsed_time > 0:
                        fps_processing = frame_idx / elapsed_time
                        remaining_frames = (end_frame - start_frame) - frame_idx
                        estimated_time = remaining_frames / fps_processing if fps_processing > 0 else 0
                        
                        progress["처리 속도"] = f"{fps_processing:.1f} 프레임/초"
                        progress["예상 남은 시간"] = f"{estimated_time:.1f}초"
                
                frame_idx += 1
                progress["처리된 프레임"] = frame_idx
    
    elif pose_model == "VIBE" or pose_model == "HumanMotion":
        # 다른 포즈 추출 모델 구현
        # 현재는 지원하지 않음
        cap.release()
        return None, {"상태": "오류", "원인": f"{pose_model} 모델은 아직 구현되지 않았습니다"}
    
    # 비디오 캡처 해제
    cap.release()
    
    # 추출된 포즈가 없는 경우
    if not pose_data:
        return None, {"상태": "오류", "원인": "포즈를 추출할 수 없습니다. 다른 영상이나 설정을 시도해보세요."}
    
    # 결과 배열 변환
    pose_array = np.array(pose_data, dtype=np.float32)
    
    # 최종 처리 결과
    result_info = {
        "상태": "완료",
        "처리된 프레임": progress["처리된 프레임"],
        "추출된 포즈": progress["추출된 포즈"],
        "추출 설정": {
            "모델": pose_model,
            "프레임 레이트": effective_fps,
            "스무딩": smooth,
            "메인 인물만": main_only
        },
        "비디오 정보": {
            "원본 FPS": video_fps,
            "총 프레임": total_frames,
            "해상도": f"{width}x{height}"
        }
    }
    
    return pose_array, result_info
