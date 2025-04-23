"""
애니메이션 생성 탭 모듈
"""

import gradio as gr
import os
import uuid  # uuid 모듈 추가
from pathlib import Path
from util.file_utils import apply_animation, send_prompt
from render.smpl_animation import apply_to_glb

def create_animation_tab(VIEWER_PATH, MODELS_DIR):
    """애니메이션 생성 탭 인터페이스 생성"""
    with gr.Column():
        gr.Markdown("# 애니메이션 생성")
        gr.Markdown("스킨이 있는 GLB 모델을 지정 후 원하는 애니메이션 데이터를 선택하세요.")
        
        with gr.Row():
            with gr.Column(scale=1):
                # 파일 업로드 영역
                skin_model = gr.File(
                    label="스킨 모델 (GLB)",
                    file_types=[".glb"],
                    type="file"
                )
                
                anim_model = gr.File(
                    label="애니메이션 파일 (GLB/BVH/NPY)",
                    file_types=[".glb", ".bvh", ".npy"],
                    type="file"
                )
                
                apply_btn = gr.Button("적용 및 보기", variant="primary")
                
                gr.Markdown("""
                ## 사용 방법
                1. 스킨이 있는 GLB 모델을 첫 번째 필드에 업로드합니다.
                2. 애니메이션 파일(GLB, BVH 또는 SMPL NPY)을 두 번째 필드에 업로드합니다.
                3. '적용 및 보기' 버튼을 클릭합니다.
                4. 오른쪽 패널에서 애니메이션이 적용된 모델을 확인합니다.
                
                **참고**: 
                - GLB 애니메이션: 두 모델의 리깅(뼈대) 구조가 동일해야 합니다. 즉시 확인 가능
                - BVH 애니메이션: 본 이름이 비슷하면 자동으로 매핑을 시도합니다. 본만 래더링 됨.
                - SMPL NPY 애니메이션: SMPL 호환 모델에 적용됩니다. 에러 발생하면 static/models에 생성된 anim_xx 파일을 선택 후 실행
                """)
                
            with gr.Column(scale=2):
                # 3D 모델 뷰어
                viewer = gr.HTML("""
                <div style="width: 100%; height: 500px; background-color: #333; border-radius: 8px; 
                         display: flex; justify-content: center; align-items: center; color: #ccc;">
                    <div style="text-align: center;">
                        <h3>모델이 표시될 영역</h3>
                        <p>모델을 업로드하고 '적용 및 보기' 버튼을 클릭하세요</p>
                    </div>
                </div>
                """)
                
                # 모델 뷰어 바로 아래에 프롬프트 입력 영역 배치
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3):
                        prompt_input = gr.Textbox(
                            label="프롬프트 입력",
                            placeholder="캐릭터에 적용할 프롬프트를 입력하세요...",
                            lines=3
                        )
                    
                    with gr.Column(scale=1, min_width=120):
                        prompt_btn = gr.Button("전송", variant="secondary")
                
                # 프롬프트 결과 출력 영역
                prompt_result = gr.HTML(visible=True)
        
        # 버튼 클릭 이벤트 처리를 위한 함수 정의
        def process_animation(skin, anim):
            import os
            from pathlib import Path

            # 파일 확장자 확인
            if anim is not None:
                file_ext = Path(anim.name).suffix.lower()
            else:
                file_ext = None

            # NPY 또는 NPZ 파일 처리 (SMPL 애니메이션)
            if file_ext in ['.npy', '.npz']:
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
                    output_path = create_improved_glb_animation(motion_data, result_path, base_glb_path=skin.name)
                    
                    if output_path:
                        # 애니메이션 적용된 GLB 모델만 뷰어에 전달
                        from types import SimpleNamespace
                        anim_glb = SimpleNamespace()
                        anim_glb.name = output_path
                        return apply_animation(skin, anim_glb, VIEWER_PATH, MODELS_DIR)
                    else:
                        return "애니메이션 적용 실패"
                        
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    return f"SMPL 애니메이션 적용 오류: {str(e)}"
            
            # 기존 GLB/BVH 파일 처리
            print(f"skin = {skin}")
            print(f"anim = {anim}")
            return apply_animation(skin, anim, VIEWER_PATH, MODELS_DIR)
        
        # 버튼 클릭 이벤트 함수 등록
        apply_btn.click(
            fn=process_animation,
            inputs=[skin_model, anim_model],
            outputs=viewer
        )
        
        # 프롬프트 전송 버튼 클릭 이벤트
        prompt_btn.click(
            fn=send_prompt,
            inputs=prompt_input,
            outputs=prompt_result
        )