"""
애니메이션 생성 탭 모듈
"""

import gradio as gr
import os
import uuid  # uuid 모듈 추가
from pathlib import Path
from util.file_utils import apply_animation
from render.smpl_animation import apply_to_glb
from util.i18n import translations  # 다국어 지원 모듈 임포트
from datetime import datetime

import os, tempfile
import numpy as np
from render.humanml3d_renderer import render_humanml3d
import requests

GEN_ENDPOINT = os.getenv('GEN_ENDPOINT', 'http://localhost:8384/predict')

last_generated_file = None  # prompt_result를 저장할 전역 변수

def download_generated_motion(): 
    global last_generated_file

    if last_generated_file and os.path.exists(last_generated_file):
        return gr.File(value=last_generated_file,visible=True, label="Generated Motion File", file_name=f"generated_motion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npy")
    else:
        return "" # 파일이 없을 경우 빈 문자열 반환

def create_animation_generation_tab(LANG_CODE, TEMPLATE_PATH, MODELS_DIR):
    """애니메이션 생성 탭 인터페이스 생성"""
    with gr.Column():
        gr.Markdown(f"# {translations[LANG_CODE]['tab_title_01']}")
        gr.Markdown(f" {translations[LANG_CODE]['tab_title_01_desc']}")
        

        with gr.Column():
            # 3D 모델 뷰어
            viewer = gr.HTML(f"""
            <div style="width: 100%; height: 500px; background-color: #333; border-radius: 8px; 
                        display: flex; justify-content: center; align-items: center; color: #ccc;">
                <div style="text-align: center;">
                    <h3>{translations[LANG_CODE]['viewport_title']}</h3>
                    <p>{translations[LANG_CODE]['viewport_desc_gen']}</p>
                </div>
            </div>
            """)
            
            # 모델 뷰어 바로 아래에 프롬프트 입력 영역 배치
            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    prompt_input = gr.Textbox(
                        label=translations[LANG_CODE]['label_prompt'],
                        placeholder=translations[LANG_CODE]['label_prompt_desc'],
                        lines=3
                    )
                
                with gr.Column(scale=1, min_width=120):
                    prompt_btn = gr.Button(translations[LANG_CODE]['bnt_send'], variant="secondary")
                    download_btn = gr.Button(translations[LANG_CODE]['btn_download'], variant="secondary")
                    file_output = gr.File(label=translations[LANG_CODE]['btn_download'], visible=False)
            
            # 프롬프트 결과 출력 영역
            prompt_result = viewer
        
        # 프롬프트 전송 버튼 클릭 이벤트
        prompt_btn.click(
            fn=send_prompt,
            inputs=prompt_input,
            outputs=prompt_result
        )

        download_btn.click(
            fn=download_generated_motion,
            outputs=file_output
        )


def send_prompt(prompt_text, progress=gr.Progress(track_tqdm=True)):
    """
    프롬프트를 전송하고 결과를 처리하는 함수
    
    Args:
        prompt_text: 사용자 입력 프롬프트
        progress: Gradio Progress 컴포넌트
    """
    
    print(f"프롬프트 전송: {str(prompt_text)}")
    if not prompt_text.strip():
        return "프롬프트를 입력해주세요."
    
    global last_generated_file  # 생성된 파일 경로를 전역 변수로 저장
    last_generated_file = None  # 초기화    
    data = {
                'prompt': str(prompt_text),
                "num_repetitions": 1,
                "output_format": "json_file",
            }
    
    print(f"프롬프트 데이터: {data}")
    try:
        # 프롬프트 전송
        progress(0, desc="프롬프트 전송 중...")
        response = requests.post(
            GEN_ENDPOINT,
            headers={'Content-Type': 'application/json'},
            json=data
        )

        if response.status_code == 200:
            result = response.json()
            try:
                json_file = result.get('json_file', {})
                for key in json_file.keys():
                    print(f"Key: {key}")

            except Exception as e:
                print(f"JSON 파일 저장 오류: {str(e)}")
                raise ValueError(str(e))

            # thetas를 numpy 배열로 변환
            thetas = json_file.get('motions', [])
            pose_array = np.array(thetas, dtype=np.float32)
            # root_translation을 numpy 배열로 변환
            root_translation = json_file.get('root_translation', [])
            trans_array = np.array(root_translation, dtype=np.float32)
            # betas는 빈 배열로 설정
            betas_array = np.array([], dtype=np.float32)
            
            # 전체 모션 데이터를 하나의 numpy 배열로 결합
            motion_array = np.concatenate([
                pose_array.reshape(-1),
                betas_array,
                trans_array.reshape(-1)
            ])

            animation_data = pose_array
            try:
                motion_data_array = animation_data.reshape(pose_array.shape[1], pose_array.shape[2], pose_array.shape[3])
            except ValueError as e:
                print(f"Reshape error: {e}. Ensure animation_data has the correct size.")
                return f"애니메이션 데이터 크기가 올바르지 않습니다. 오류: {e}"
            
            try:
                # 딕셔너리로 래핑하여 #terminalSelection과 같은 형식으로 만들기
                motion_dict = {
                    'motion': motion_data_array,
                    'text': ['good job'],
                    'lengths': np.array([pose_array.shape[1]]),
                    'num_samples': 1,
                    'num_repetitions': 1
                }
                pose_data = motion_dict
                with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                    # 변환된 pose 데이터를 numpy 파일로 저장
                    np.save(tmp_path, pose_data)
                    print(f"애니메이션 데이터를 임시 파일에 저장: {tmp_path}")

                    # 전역 변수에 저장
                    last_generated_file = tmp_path

                    # 임시 파일을 이용하여 애니메이션 처리
                    try:
                        print(f"애니메이션 변환 성공: {tmp_path}")
                        return render_humanml3d(tmp_file)
                    except Exception as e:
                        print(f"애니메이션 변환 실패: {str(e)}")
                        return f"""
                        <div id="prompt-result">
                            <p style="color: red;">애니메이션 처리 오류</p>
                            <p style="color: #666; font-size: 0.9em;">{str(e)}</p>
                        </div>
                        """
            except Exception as e:
                print(f"요청 오류: {str(e)}")
            
            # 최대 재시도 횟수 도달
            return f"""
            <div id="prompt-result">
                <p style="color: orange;">모션 생성 시간 초과</p>
                <p style="color: #666; font-size: 0.9em;">서버가 바쁠 수 있습니다. 잠시 후 다시 시도해주세요.</p>
            </div>
            """
        else:
            return f"""
            <div id="prompt-result">
                <p style="color: red;">서버 오류 발생 (상태 코드: {response.status_code})</p>
                <p style="color: #666; font-size: 0.9em;">잠시 후 다시 시도해주세요.</p>
            </div>
            """
            
    except requests.exceptions.RequestException as e:
        return f"""
        <div id="prompt-result">
            <p style="color: red;">네트워크 오류 발생</p>
            <p style="color: #666; font-size: 0.9em;">{str(e)}</p>
        </div>
        """
    except Exception as e:
        return f"""
        <div id="prompt-result">
            <p style="color: red;">알 수 없는 오류 발생</p>
            <p style="color: #666; font-size: 0.9em;">{str(e)}</p>
        </div>
        """        