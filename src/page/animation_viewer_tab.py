"""
애니메이션 생성 탭 모듈
"""

import gradio as gr
from pathlib import Path
from util.file_utils import apply_animation
from render.humanml3d_renderer import render_humanml3d
from pathlib import Path
from util.i18n import translations  # 다국어 지원 모듈 임포트

def create_animation_viewer_tab(LANG_CODE, TEMPLATE_PATH, MODELS_DIR):
    """애니메이션 뷰어 탭 인터페이스 생성"""
    with gr.Column():
        gr.Markdown(f"# {translations[LANG_CODE]['tab_title_02']}")
        gr.Markdown(f" {translations[LANG_CODE]['tab_title_02_desc']}")
        
        with gr.Row():
            with gr.Column(scale=1):
                anim_model = gr.File(
                    label="Motion file (GLB/BVH/NPY)",
                    file_types=[".glb", ".bvh", ".npy"],
                    type="file"
                )

                # 파일 업로드 영역
                skin_model = gr.File(
                    label="Skin Model (GLB)",
                    file_types=[".glb"],
                    type="file",
                    visible=False  # Initially hidden
                )
                
                # anim_model 변경 감지 및 skin_model 가시성 처리
                def update_skin_visibility(anim_file):
                    if anim_file is None:
                        return gr.update(visible=False)
                    
                    file_ext = Path(anim_file.name).suffix.lower()
                    # GLB 파일일 때만 skin_model 표시
                    return gr.update(visible=file_ext == '.glb')
                
                anim_model.change(
                    fn=update_skin_visibility,
                    inputs=[anim_model],
                    outputs=[skin_model]
                )
                apply_btn = gr.Button("Apply", variant="primary")
                
                gr.Markdown(translations[LANG_CODE]['desc_viewer'])
                
            with gr.Column(scale=2):
                # 3D 모델 뷰어
                viewer = gr.HTML(f"""
                <div style="width: 100%; height: 500px; background-color: #333; border-radius: 8px; 
                         display: flex; justify-content: center; align-items: center; color: #ccc;">
                    <div style="text-align: center;">
                        <h3>{translations[LANG_CODE]['viewport_title']}</h3>
                        <p>{translations[LANG_CODE]['viewport_desc_viewer']}</p>
                    </div>
                </div>
                """)
        
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

            # NPY 또는 NPZ 파일 처리 (HuamlML3D 형식)
            if file_ext in ['npy', 'npz']:
                return render_humanml3d(anim)

            # glb의 경우 skin, anim 모두 지정 필요.
            # 기존 glb에 skin, anim 모두 있다면 별도 처리 (추후 구현 필요)
            return apply_animation(skin, anim, TEMPLATE_PATH, MODELS_DIR, file_ext)
        
        # 버튼 클릭 이벤트 함수 등록
        apply_btn.click(
            fn=process_animation,
            inputs=[skin_model, anim_model],
            outputs=viewer
        )