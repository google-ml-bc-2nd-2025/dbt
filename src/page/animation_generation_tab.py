"""
애니메이션 생성 탭 모듈
"""

import gradio as gr
import os
import uuid  # uuid 모듈 추가
from pathlib import Path
from util.file_utils import apply_animation
from render.request_to_server import send_prompt
from render.smpl_animation import apply_to_glb
from util.i18n import translations  # 다국어 지원 모듈 임포트
from datetime import datetime

last_generated_file = None  # prompt_result를 저장할 전역 변수

def download_generated_motion(): 
    global last_generated_file

    if last_generated_file and os.path.exists(last_generated_file):
        return last_generated_file
    else:
        return "No generated file available."

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
            
            # 프롬프트 결과 출력 영역
            prompt_result = viewer
        
        # 버튼 클릭 이벤트 처리를 위한 함수 정의
        def process_animation(skin, anim):
            import os
            from pathlib import Path

            # 파일 확장자 확인
            if anim is not None:
                file_ext = Path(anim.name).suffix.lower()
                # Remove the leading dot from the extension
                if file_ext.startswith('.'):
                    file_ext = file_ext[1:]
            else:
                file_ext = 'glb'

            # NPY 또는 NPZ 파일 처리 (SMPL 애니메이션)
            if file_ext in ['npy', 'npz']:
                try:
                    # import 문 추가
                    import numpy as np
                    from converter.convert_mdm_to_glb import create_improved_glb_animation

                    # 애니메이션 데이터 로드 (MDM 형식)
                    motion_data = np.load(anim.name, allow_pickle=True)
                    
                    # motion 키가 있는지 확인
                    if isinstance(motion_data, np.ndarray) and motion_data.dtype == np.dtype('O') and isinstance(motion_data.item(), dict):
                        if 'motion' in motion_data.item():
                            motion_data = motion_data.item()['motion']
                    elif isinstance(motion_data, dict) and 'motion' in motion_data:
                        motion_data = motion_data['motion']
                        
                    print(f"Motion 데이터 형태: {type(motion_data)}")
                    if isinstance(motion_data, np.ndarray):
                        print(f"Shape: {motion_data.shape}, 차원: {motion_data.ndim}")
                        
                    # 애니메이션을 스킨에 직접 적용 (새 함수 사용)
                    unique_id = str(uuid.uuid4())[:8]
                    result_filename = f"anim_{unique_id}.glb"
                    result_path = os.path.join(MODELS_DIR, result_filename)
                    
                    # 새 함수 호출 - 미리 로드된 스킨에 애니메이션 적용
                    output_path = create_improved_glb_animation(motion_data, result_path, file_ext)
                    
                    if output_path:
                        # 애니메이션 적용된 GLB 모델만 뷰어에 전달
                        from types import SimpleNamespace
                        anim_glb = SimpleNamespace()
                        anim_glb.name = output_path
                        return apply_animation(skin, anim_glb, TEMPLATE_PATH, MODELS_DIR, file_ext)
                    else:
                        return translations[LANG_CODE]['desc_failed_to_generate_motion']
                        
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    return f"Error : {str(e)}"
            
            # 기존 GLB/BVH 파일 처리
            print(f"skin = {skin}")
            print(f"anim = {anim}")
            return apply_animation(skin, anim, TEMPLATE_PATH, MODELS_DIR)
        
        # 프롬프트 전송 버튼 클릭 이벤트
        prompt_btn.click(
            fn=send_prompt,
            inputs=prompt_input,
            outputs=prompt_result
        )

        download_btn.click(
            fn=download_generated_motion,
            inputs=[],
            outputs=viewer
        )