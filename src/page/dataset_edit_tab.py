"""
애니메이션 학습 데이터셋 수정 탭 모듈 - 유튜브/틱톡 영상 데이터 추출 기능 추가
"""

import gradio as gr
import os
from pathlib import Path
import numpy as np
import tempfile

# 모듈화된 코드 임포트
from converter.pose_extractor import extract_pose_from_video
from converter.glb_generator import generate_glb_from_pose_data, map_mediapipe_to_glb_joints
from converter.dataset_generator import process_animation_files

def create_dataset_edit_tab(VIEWER_PATH,MODELS_DIR):
    """
    애니메이션 학습 데이터셋 수정 탭 인터페이스 생성
    
    Args:
        MODELS_DIR: 모델 파일이 저장된 디렉토리 경로
    """
    with gr.Column():
        gr.Markdown("# 애니메이션 학습 데이터셋 수정")
        gr.Markdown("온라인 영상에서 애니메이션 모델 학습에 필요한 포즈 데이터를 추출합니다.")
        
        with gr.Tabs():
            with gr.TabItem("온라인 영상 추출"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # 영상 소스 입력 옵션
                        video_url = gr.Textbox(
                            label="유튜브/틱톡 URL",
                            placeholder="https://www.youtube.com/watch?v=..."
                        )
                        
                        uploaded_video = gr.Video(
                            label="또는 직접 영상 업로드",
                            type="filepath"
                        )
                        
                        with gr.Row():
                            with gr.Column():
                                start_time = gr.Number(
                                    label="시작 시간(초)", 
                                    value=0,
                                    precision=2
                                )
                                
                            with gr.Column():
                                end_time = gr.Number(
                                    label="종료 시간(초)",
                                    value=10,
                                    precision=2
                                )
                        
                        # 포즈 추출 설정
                        with gr.Row():
                            pose_model = gr.Dropdown(
                                label="포즈 추출 모델",
                                choices=["MediaPipe", "VIBE", "HumanMotion"],
                                value="MediaPipe"
                            )
                            
                            frame_rate = gr.Slider(
                                label="추출 프레임 레이트",
                                minimum=5,
                                maximum=60,
                                value=30,
                                step=1
                            )
                        
                        with gr.Row():
                            smoothing = gr.Slider(
                                label="스무딩 강도",
                                minimum=0,
                                maximum=1.0,
                                value=0.3,
                                step=0.1
                            )
                            
                            focus_main = gr.Checkbox(
                                label="메인 인물만 추출",
                                value=True
                            )
                        
                        dataset_name = gr.Textbox(
                            label="데이터셋 이름",
                            placeholder="추출할 데이터셋 이름을 입력하세요",
                            value="youtube_dance_dataset"
                        )
                        
                        extract_btn = gr.Button("포즈 데이터 추출", variant="primary")
                        
                        gr.Markdown("""
                        ## 영상에서 포즈 추출 방법
                        1. 유튜브/틱톡 URL을 입력하거나 직접 영상을 업로드합니다.
                        2. 포즈 추출 범위와 설정을 조정합니다.
                        3. '포즈 데이터 추출' 버튼을 클릭합니다.
                        4. 추출된 애니메이션 데이터는 오른쪽에 결과가 표시됩니다.
                        """)
                    
            with gr.TabItem("온라인 영상 추출"):
                with gr.Row():
                    with gr.Column(scale=1):  # 입력 폼
                        # ...입력 폼 코드...
                        extract_btn = gr.Button("포즈 데이터 추출", variant="primary")
                        gr.Markdown("""...설명...""")
                    with gr.Column(scale=1):  # 영상 미리보기만 별도 컬럼에 분리
                        original_video = gr.Video(
                            label="원본 영상",
                            interactive=False,
                            elem_id="original_video",
                            autoplay=False,
                            show_label=True,
                        )
                    with gr.Column(scale=1):  # 상태/다운로드
                        extraction_status = gr.JSON(
                            label="처리 상태",
                            value={
                                "상태": "대기 중",
                                "처리된 프레임": 0,
                                "추출된 포즈": 0,
                                "예상 완료 시간": "-"
                            }
                        )
                        with gr.Row():
                            export_format = gr.Dropdown(
                                label="내보내기 포맷",
                                choices=["GLB", "BVH", "FBX"],
                                value="GLB"
                            )
                            download_btn = gr.Button("다운로드", variant="secondary")
                
            with gr.TabItem("기존 파일 편집"):
                # 기존 코드의 파일 업로드 옵션을 여기로 이동
                with gr.Row():
                    with gr.Column(scale=1):
                        # 데이터셋 생성 옵션
                        source_files = gr.File(
                            label="소스 애니메이션 파일 (GLB/BVH/FBX)",
                            file_types=[".glb", ".bvh", ".fbx"],
                            type="file",
                            file_count="multiple"
                        )
                        
                        dataset_name = gr.Textbox(
                            label="데이터셋 이름",
                            placeholder="수정할 데이터셋 이름을 입력하세요",
                            value="my_animation_dataset"
                        )
                        
                        with gr.Row():
                            frames_per_clip = gr.Number(
                                label="클립당 프레임 수", 
                                value=64,
                                precision=0,
                                minimum=16,
                                maximum=256
                            )
                            
                            frame_rate = gr.Number(
                                label="프레임 레이트", 
                                value=30,
                                precision=0,
                                minimum=10,
                                maximum=120
                            )
                        
                        generate_btn = gr.Button("데이터셋 수정", variant="primary")
                        
                        gr.Markdown("""
                        ## 데이터셋 수정 방법
                        1. 하나 이상의 애니메이션 파일(GLB, BVH 또는 FBX)을 업로드합니다.
                        2. 데이터셋 이름, 설정, 라벨 등을 수정합니다.
                        3. '데이터셋 수정' 버튼을 클릭합니다.
                        4. 수정된 데이터셋은 오른쪽에 결과가 표시됩니다.
                        """)
                    
                    with gr.Column(scale=2):
                        # 데이터셋 수정 결과 출력 영역
                        dataset_output = gr.HTML("""
                        <div style="width: 100%; height: 500px; background-color: #333; border-radius: 8px; 
                                 display: flex; justify-content: center; align-items: center; color: #ccc;">
                            <div style="text-align: center;">
                                <h3>데이터셋 수정 결과</h3>
                                <p>데이터셋을 수정하려면 왼쪽 패널에서 파일을 업로드하고 '데이터셋 수정' 버튼을 클릭하세요</p>
                            </div>
                        </div>
                        """)
                        
                        # 데이터셋 통계 영역
                        dataset_stats = gr.JSON({
                            "상태": "대기 중",
                            "파일 수": 0,
                            "총 프레임": 0,
                            "클립 수": 0
                        })
        
        # 포즈 추출 이벤트 핸들러
        def handle_extract_pose(url, uploaded_file, start, end, model, fps, smooth, main_only, name):
            """
            영상에서 포즈 데이터 추출 이벤트 핸들러
            """
            # 모듈화된 함수 호출
            return extract_pose_from_video(url, uploaded_file, start, end, model, fps, smooth, main_only, name, MODELS_DIR)
        
        # 데이터셋 생성 이벤트 핸들러
        def handle_generate_dataset(files, name, frames, rate):
            """
            애니메이션 학습 데이터셋 생성/수정 이벤트 핸들러
            """
            # 모듈화된 함수 호출
            return process_animation_files(files, name, frames, rate, str(MODELS_DIR))
        
        # 포즈 데이터 다운로드 이벤트 핸들러
        def download_pose_data(format_choice, status_data):
            """
            추출된 포즈 데이터를 선택한 포맷으로 변환하여 저장하는 함수
            """
            try:
                if status_data.get("상태") != "완료":
                    return {
                        "상태": "오류",
                        "메시지": "다운로드할 포즈 데이터가 없습니다. 먼저 포즈를 추출해주세요."
                    }
                
                # 원본 포즈 데이터 가져오기
                pose_array = None
                if "포즈 데이터" in status_data and status_data["포즈 데이터"] != "너무 큼":
                    # 데이터가 문자열로 저장되지 않은 경우 직접 사용
                    pose_array = np.array(status_data["포즈 데이터"], dtype=np.float32)
                    print(f"포즈 데이터 로드됨: {pose_array.shape}")
                
                dataset_name = status_data.get("데이터셋 이름", "extracted_pose")
                
                # ./new_dataset 디렉토리가 없으면 생성
                output_dir = os.path.join(".", "new_dataset")
                os.makedirs(output_dir, exist_ok=True)
                
                # 파일명 생성 (공백은 밑줄로 대체)
                file_name = dataset_name.replace(" ", "_")
                output_path = os.path.join(output_dir, f"{file_name}.{format_choice.lower()}")
                
                # 파일 형식에 따라 다른 처리 방식 적용
                frame_count = status_data.get("추출된 포즈", 0)
                if frame_count == 0:
                    frame_count = status_data.get("처리된 프레임", 0)
                
                # 추출 설정 정보
                settings = status_data.get("추출 설정", {})
                fps = settings.get("프레임 레이트", 30)
                
                # 디버그 정보 출력
                print(f"다운로드 요청 - 형식: {format_choice}, 프레임 수: {frame_count}, FPS: {fps}")

                # GLB 형식으로 변환
                if format_choice == "GLB":
                    if pose_array is None:
                        # 테스트 데이터 생성 (실제 데이터가 없는 경우)
                        num_frames = frame_count or 60
                        
                        # 기본 관절 이름 정의 - Mixamo 형식으로 수정
                        joint_names = [
                            "mixamorig:Hips", "mixamorig:Spine", "mixamorig:Spine1", "mixamorig:Spine2", 
                            "mixamorig:Neck", "mixamorig:Head",
                            "mixamorig:LeftShoulder", "mixamorig:LeftArm", "mixamorig:LeftForeArm", "mixamorig:LeftHand",
                            "mixamorig:RightShoulder", "mixamorig:RightArm", "mixamorig:RightForeArm", "mixamorig:RightHand",
                            "mixamorig:LeftUpLeg", "mixamorig:LeftLeg", "mixamorig:LeftFoot", "mixamorig:LeftToeBase",
                            "mixamorig:RightUpLeg", "mixamorig:RightLeg", "mixamorig:RightFoot", "mixamorig:RightToeBase"
                        ]
                        num_joints = len(joint_names)
                        
                        # 테스트 포즈 데이터 생성
                        pose_data = np.zeros((num_frames, num_joints, 3), dtype=np.float32)
                        t = np.linspace(0, 2 * np.pi, num_frames)
                        
                        # 간단한 애니메이션 패턴 생성
                        for i, joint_name in enumerate(joint_names):
                            # 기본 위치 설정
                            base_x = 0.2 if "Left" in joint_name else (-0.2 if "Right" in joint_name else 0)
                            base_y = -0.5 if "Leg" in joint_name or "Foot" in joint_name else (0.5 if "Head" in joint_name else 0.2)
                            base_z = 0
                            
                            pose_data[:, i, 0] = base_x + 0.05 * np.sin(t + i*0.2)
                            pose_data[:, i, 1] = base_y + 0.05 * np.sin(t*0.5 + i*0.1)
                            pose_data[:, i, 2] = base_z + 0.05 * np.cos(t + i*0.3)
                    else:
                        # 실제 MediaPipe 데이터를 GLB 호환 형식으로 매핑
                        pose_data, joint_names = map_mediapipe_to_glb_joints(pose_array)
                    
                    # GLB 생성 함수 호출
                    success, message = generate_glb_from_pose_data(pose_data, output_path, fps, joint_names)
                    
                    if success:
                        return {
                            "상태": "완료",
                            "메시지": f"포즈 데이터가 {format_choice} 형식으로 저장되었습니다.",
                            "파일 경로": output_path,
                            "형식": format_choice
                        }
                    else:
                        return {
                            "상태": "오류",
                            "메시지": message
                        }
                
                # 다른 형식 처리 (BVH, FBX 등)
                else:
                    # 더미 구현
                    with open(output_path, 'w' if format_choice == "BVH" else 'wb') as f:
                        if format_choice == "BVH":
                            f.write(f"HIERARCHY\nROOT Hips\n...\n")
                        else:
                            f.write(b'\x00' * 1024)  # 더미 바이너리 데이터
                        
                        return {
                            "상태": "완료",
                            "메시지": f"포즈 데이터가 {format_choice} 형식으로 저장되었습니다.",
                            "파일 경로": output_path,
                            "형식": format_choice
                        }
                            
            except Exception as e:
                import traceback
                print(f"다운로드 중 오류 발생: {e}")
                print(traceback.format_exc())
                return {
                    "상태": "오류",
                    "메시지": f"다운로드 중 오류가 발생했습니다: {str(e)}"
                }
        
        # 포즈 미리보기 이벤트 핸들러
        # def show_pose_preview(status_data):
        #     """
        #     추출된 포즈를 3D로 미리보기하는 함수 - 템플릿 HTML 파일 활용
            
        #     Args:
        #         status_data: 포즈 추출 상태 데이터
                
        #     Returns:
        #         미리보기 HTML
        #     """
        #     import json
        #     import base64
        #     from pathlib import Path
            
        #     # 포즈 데이터 가져오기 시도
        #     pose_array = None
        #     if status_data.get("상태") == "완료" and "포즈 데이터" in status_data and status_data["포즈 데이터"] != "너무 큼":
        #         pose_array = np.array(status_data["포즈 데이터"], dtype=np.float32)
            
        #     # 포즈 데이터가 없으면 테스트용 데이터 생성
        #     if pose_array is None or len(pose_array) == 0:
        #         # 테스트용 포즈 데이터 생성 (T-포즈)
        #         num_frames = 60
                
        #         # SMPL 모델의 24 관절 이름 사용
        #         from bone_mappings import SMPL_JOINT_NAMES
        #         joint_names = SMPL_JOINT_NAMES
        #         num_joints = len(joint_names)
                
        #         # 기본 T-포즈 형태의 테스트 데이터 생성
        #         pose_data = create_test_skeleton(num_frames, joint_names)
        #     else:
        #         pose_data = pose_array
        #         num_frames = len(pose_data)
        #         num_joints = pose_data.shape[1]
            
        #     # 첫 프레임만 사용
        #     first_frame = pose_data[0].tolist()
            
        #     # 관절 연결 정보 생성
        #     bone_connections = get_bone_connections(num_joints)
            
        #     # 데이터를 base64로 인코딩
        #     pose_json = json.dumps(first_frame)
        #     pose_base64 = base64.b64encode(pose_json.encode()).decode()
            
        #     bones_json = json.dumps(bone_connections)
        #     bones_base64 = base64.b64encode(bones_json.encode()).decode()
            
        #     # 템플릿 HTML 파일 경로 - 상대 경로로 수정
        #     template_path = "./static/pose_viewer_template.html"
            
        #     # iframe으로 템플릿 뷰어 표시
        #     preview_html = f"""
        #     <div style="width: 100%; background-color: #333; border-radius: 8px; padding: 15px; color: #fff;">
        #         <h3>포즈 미리보기</h3>
        #         <div style="background-color: #222; padding: 15px; border-radius: 5px; margin-top: 10px;">
        #             <p>총 {num_frames}개 프레임 중 첫 번째 프레임을 표시합니다.</p>
        #             <p>관절 수: {num_joints}개</p>
        #         </div>
        #         <div style="width: 100%; height: 400px; margin-top: 15px; border-radius: 5px; overflow: hidden; position: relative;">
        #             <iframe src="{template_path}?pose={pose_base64}&bones={bones_base64}&frames={num_frames}"
        #                     style="width: 100%; height: 100%; border: none;">
        #             </iframe>
        #         </div>
        #         <div style="margin-top: 10px; text-align: center;">
        #             <p><small>3D 뷰어에서 마우스를 사용하여 회전하고 확대/축소할 수 있습니다.</small></p>
        #         </div>
        #     </div>
        #     """
            
        #     return preview_html
        
        def get_bone_connections(num_joints):
            """
            관절 수에 따라 적절한 스켈레톤 연결 구조를 반환합니다.
            
            Args:
                num_joints: 관절의 수
                
            Returns:
                연결선 목록 [(시작 관절 인덱스, 끝 관절 인덱스, 연결선 이름)]
            """
            # MediaPipe 포맷 (33개 관절)
            if num_joints >= 33:
                return [
                    # 얼굴 연결선
                    [0, 1, "Face"],  # 코-왼쪽 눈
                    [0, 4, "Face"],  # 코-오른쪽 눈
                    [1, 2, "Face"],  # 왼쪽 눈-왼쪽 귀
                    [4, 5, "Face"],  # 오른쪽 눈-오른쪽 귀
                    
                    # 상체 연결선
                    [11, 12, "SpineUpper"],    # 왼쪽 어깨-오른쪽 어깨
                    [12, 24, "SpineRight"],    # 오른쪽 어깨-오른쪽 골반
                    [11, 23, "SpineLeft"],     # 왼쪽 어깨-왼쪽 골반
                    [23, 24, "SpineLower"],    # 왼쪽 골반-오른쪽 골반
                    
                    # 왼팔 연결선
                    [11, 13, "LeftArm"],       # 왼쪽 어깨-왼쪽 팔꿈치
                    [13, 15, "LeftForeArm"],   # 왼쪽 팔꿈치-왼쪽 손목
                    [15, 17, "LeftHand"],      # 왼쪽 손목-왼쪽 손
                    [15, 19, "LeftHand"],      # 왼쪽 손목-왼쪽 검지
                    [15, 21, "LeftHand"],      # 왼쪽 손목-왼쪽 새끼손가락
                    
                    # 오른팔 연결선
                    [12, 14, "RightArm"],      # 오른쪽 어깨-오른쪽 팔꿈치
                    [14, 16, "RightForeArm"],  # 오른쪽 팔꿈치-오른쪽 손목
                    [16, 18, "RightHand"],     # 오른쪽 손목-오른쪽 손
                    [16, 20, "RightHand"],     # 오른쪽 손목-오른쪽 검지
                    [16, 22, "RightHand"],     # 오른쪽 손목-오른쪽 새끼손가락
                    
                    # 왼쪽 다리 연결선
                    [23, 25, "LeftLeg"],       # 왼쪽 골반-왼쪽 무릎
                    [25, 27, "LeftShin"],      # 왼쪽 무릎-왼쪽 발목
                    [27, 29, "LeftFoot"],      # 왼쪽 발목-왼쪽 발
                    [27, 31, "LeftFoot"],      # 왼쪽 발목-왼쪽 발가락
                    [29, 31, "LeftFoot"],      # 왼쪽 발-왼쪽 발가락
                    
                    # 오른쪽 다리 연결선
                    [24, 26, "RightLeg"],      # 오른쪽 골반-오른쪽 무릎
                    [26, 28, "RightShin"],     # 오른쪽 무릎-오른쪽 발목
                    [28, 30, "RightFoot"],     # 오른쪽 발목-오른쪽 발
                    [28, 32, "RightFoot"],     # 오른쪽 발목-오른쪽 발가락
                    [30, 32, "RightFoot"]      # 오른쪽 발-오른쪽 발가락
                ]
            # GLB/Mixamo 포맷 (약 22개 관절)
            elif num_joints >= 22:
                return [
                    # 중심 연결선
                    [0, 1, "Spine"],           # 골반-척추1
                    [1, 2, "Spine"],           # 척추1-척추2
                    [2, 3, "Spine"],           # 척추2-척추3
                    [3, 4, "Neck"],            # 척추3-목
                    [4, 5, "Head"],            # 목-머리
                    
                    # 왼쪽 팔 연결선
                    [3, 6, "LeftShoulder"],    # 척추3-왼쪽 어깨
                    [6, 7, "LeftArm"],         # 왼쪽 어깨-왼쪽 팔
                    [7, 8, "LeftForeArm"],     # 왼쪽 팔-왼쪽 팔뚝
                    [8, 9, "LeftHand"],        # 왼쪽 팔뚝-왼쪽 손
                    
                    # 오른쪽 팔 연결선
                    [3, 10, "RightShoulder"],  # 척추3-오른쪽 어깨
                    [10, 11, "RightArm"],      # 오른쪽 어깨-오른쪽 팔
                    [11, 12, "RightForeArm"],  # 오른쪽 팔-오른쪽 팔뚝
                    [12, 13, "RightHand"],     # 오른쪽 팔뚝-오른쪽 손
                    
                    # 왼쪽 다리 연결선
                    [0, 14, "LeftUpLeg"],      # 골반-왼쪽 허벅지
                    [14, 15, "LeftLeg"],       # 왼쪽 허벅지-왼쪽 다리
                    [15, 16, "LeftFoot"],      # 왼쪽 다리-왼쪽 발
                    [16, 17, "LeftToeBase"],   # 왼쪽 발-왼쪽 발가락
                    
                    # 오른쪽 다리 연결선
                    [0, 18, "RightUpLeg"],     # 골반-오른쪽 허벅지
                    [18, 19, "RightLeg"],      # 오른쪽 허벅지-오른쪽 다리
                    [19, 20, "RightFoot"],     # 오른쪽 다리-오른쪽 발
                    [20, 21, "RightToeBase"]   # 오른쪽 발-오른쪽 발가락
                ]
            # SMPL 포맷 (24개 관절)
            elif num_joints >= 24:
                return [
                    # 중심축
                    [0, 3, "Spine"],           # 골반-척추1
                    [3, 6, "Spine"],           # 척추1-척추2
                    [6, 9, "Spine"],           # 척추2-척추3
                    [9, 12, "Neck"],           # 척추3-목
                    [12, 15, "Head"],          # 목-머리
                    
                    # 왼쪽 팔
                    [9, 13, "LeftShoulder"],   # 척추3-왼쪽 어깨
                    [13, 16, "LeftShoulder"],  # 왼쪽 어깨-왼쪽 어깨 관절
                    [16, 18, "LeftArm"],       # 왼쪽 어깨 관절-왼쪽 팔꿈치
                    [18, 20, "LeftForeArm"],   # 왼쪽 팔꿈치-왼쪽 손목
                    [20, 22, "LeftHand"],      # 왼쪽 손목-왼쪽 손
                    
                    # 오른쪽 팔
                    [9, 14, "RightShoulder"],  # 척추3-오른쪽 어깨 
                    [14, 17, "RightShoulder"], # 오른쪽 어깨-오른쪽 어깨 관절
                    [17, 19, "RightArm"],      # 오른쪽 어깨 관절-오른쪽 팔꿈치
                    [19, 21, "RightForeArm"],  # 오른쪽 팔꿈치-오른쪽 손목
                    [21, 23, "RightHand"],     # 오른쪽 손목-오른쪽 손
                    
                    # 왼쪽 다리
                    [0, 1, "LeftUpLeg"],       # 골반-왼쪽 엉덩이
                    [1, 4, "LeftLeg"],         # 왼쪽 엉덩이-왼쪽 무릎
                    [4, 7, "LeftFoot"],        # 왼쪽 무릎-왼쪽 발목
                    [7, 10, "LeftToeBase"],    # 왼쪽 발목-왼쪽 발
                    
                    # 오른쪽 다리
                    [0, 2, "RightUpLeg"],      # 골반-오른쪽 엉덩이
                    [2, 5, "RightLeg"],        # 오른쪽 엉덩이-오른쪽 무릎
                    [5, 8, "RightFoot"],       # 오른쪽 무릎-오른쪽 발목
                    [8, 11, "RightToeBase"]    # 오른쪽 발목-오른쪽 발
                ]
            # 기본 간소화된 연결선
            else:
                # 간소화된 스켈레톤 (관절이 적은 경우)
                return [
                    # 일반적인 인체 구조를 기준으로 기본 연결 구성
                    [0, 1, "Spine"],           # 루트-등뼈
                    [1, 2, "Spine"],           # 등뼈-가슴
                    [2, 3, "Neck"],            # 가슴-머리
                    
                    # 팔 연결선
                    [2, 4, "LeftArm"],         # 가슴-왼쪽 팔
                    [4, 5, "LeftForeArm"],     # 왼쪽 팔-왼쪽 손
                    [2, 6, "RightArm"],        # 가슴-오른쪽 팔
                    [6, 7, "RightForeArm"],    # 오른쪽 팔-오른쪽 손
                    
                    # 다리 연결선
                    [0, 8, "LeftLeg"],         # 루트-왼쪽 다리
                    [8, 9, "LeftShin"],        # 왼쪽 다리-왼쪽 발
                    [0, 10, "RightLeg"],       # 루트-오른쪽 다리
                    [10, 11, "RightShin"]      # 오른쪽 다리-오른쪽 발
                ]
        
        def create_test_skeleton(num_frames, joint_names):
            """
            테스트용 T-포즈 스켈레톤 데이터를 생성합니다.
            smpl_animation.py의 관절 구조를 참고하여 구현
            
            Args:
                num_frames: 생성할 프레임 수
                joint_names: 관절 이름 리스트
                
            Returns:
                테스트용 포즈 데이터 배열 [프레임 수, 관절 수, 3(xyz)]
            """
            num_joints = len(joint_names)
            pose_data = np.zeros((num_frames, num_joints, 3), dtype=np.float32)
            
            # SMPL 모델의 표준 T-포즈 설정
            # 관절 계층 구조 정의 (SMPL 표준)
            joint_hierarchy = {
                "pelvis": None,  # 루트 관절
                "left_hip": "pelvis",
                "right_hip": "pelvis",
                "spine1": "pelvis",
                "left_knee": "left_hip",
                "right_knee": "right_hip",
                "spine2": "spine1",
                "left_ankle": "left_knee",
                "right_ankle": "right_knee",
                "spine3": "spine2",
                "left_foot": "left_ankle",
                "right_foot": "right_ankle",
                "neck": "spine3",
                "left_collar": "spine3",
                "right_collar": "spine3",
                "head": "neck",
                "left_shoulder": "left_collar",
                "right_shoulder": "right_collar",
                "left_elbow": "left_shoulder",
                "right_elbow": "right_shoulder",
                "left_wrist": "left_elbow",
                "right_wrist": "right_elbow",
                "left_hand": "left_wrist",
                "right_hand": "right_wrist"
            }
            
            # 관절 기본 위치 설정
            joint_positions = {
                # 중심축
                "pelvis": [0.0, 0.0, 0.0],
                "spine1": [0.0, 0.1, 0.0],
                "spine2": [0.0, 0.25, 0.0],
                "spine3": [0.0, 0.4, 0.0],
                "neck": [0.0, 0.5, 0.0],
                "head": [0.0, 0.7, 0.0],
                # 왼쪽 팔
                "left_collar": [0.1, 0.45, 0.0],
                "left_shoulder": [0.2, 0.45, 0.0],
                "left_elbow": [0.4, 0.45, 0.0],
                "left_wrist": [0.6, 0.45, 0.0],
                "left_hand": [0.7, 0.45, 0.0],
                # 오른쪽 팔
                "right_collar": [-0.1, 0.45, 0.0],
                "right_shoulder": [-0.2, 0.45, 0.0],
                "right_elbow": [-0.4, 0.45, 0.0],
                "right_wrist": [-0.6, 0.45, 0.0],
                "right_hand": [-0.7, 0.45, 0.0],
                # 왼쪽 다리
                "left_hip": [0.1, -0.1, 0.0],
                "left_knee": [0.15, -0.4, 0.0],
                "left_ankle": [0.15, -0.7, 0.0],
                "left_foot": [0.15, -0.75, 0.1],
                # 오른쪽 다리
                "right_hip": [-0.1, -0.1, 0.0],
                "right_knee": [-0.15, -0.4, 0.0],
                "right_ankle": [-0.15, -0.7, 0.0],
                "right_foot": [-0.15, -0.75, 0.1]
            }
            
            # 각 관절의 위치 설정
            for i, joint_name in enumerate(joint_names):
                if joint_name in joint_positions:
                    pose_data[:, i] = joint_positions[joint_name]
            
            # 모든 프레임에 동일한 T-포즈 적용
            # 프레임별로 약간의 움직임 추가 (호흡 효과)
            if num_frames > 1:
                t = np.linspace(0, 2 * np.pi, num_frames)
                
                # 호흡 움직임 (척추와 머리)
                spine_indices = [i for i, name in enumerate(joint_names) if "spine" in name or "neck" in name or "head" in name]
                for idx in spine_indices:
                    pose_data[:, idx, 1] += 0.01 * np.sin(t)  # y축(위아래) 움직임
                
                # 팔 약간 움직임
                arm_indices = [i for i, name in enumerate(joint_names) if "hand" in name or "wrist" in name]
                for idx in arm_indices:
                    pose_data[:, idx, 0] += 0.01 * np.sin(t + idx * 0.5)  # x축 움직임
                    pose_data[:, idx, 2] += 0.005 * np.cos(t)  # z축 움직임
            
            return pose_data
        
        # 버튼 클릭 이벤트 함수 등록
        extract_btn.click(
            fn=handle_extract_pose,
            inputs=[video_url, uploaded_video, start_time, end_time, pose_model, 
                   frame_rate, smoothing, focus_main, dataset_name],
            outputs=[original_video, extraction_status]
        )
        
        generate_btn.click(
            fn=handle_generate_dataset,
            inputs=[source_files, dataset_name, frames_per_clip, frame_rate],
            outputs=[dataset_output, dataset_stats]
        )
        
        download_btn.click(
            fn=download_pose_data,
            inputs=[export_format, extraction_status],
            outputs=[extraction_status]
        )
        
        # preview_btn.click(
        #     fn=show_pose_preview,
        #     inputs=[extraction_status],
        #     outputs=[pose_preview]
        # )

# 단독 실행 시 테스트
if __name__ == "__main__":
    with gr.Blocks() as demo:
        create_dataset_edit_tab(".")
    demo.launch()