"""
애니메이션 생성 탭 모듈
"""

import gradio as gr
import os
import uuid  # uuid 모듈 추가
from pathlib import Path
from util.file_utils import apply_animation, send_prompt
from render.smpl_animation import apply_to_glb
import numpy as np
import base64
import json
from io import BytesIO
from pathlib import Path

def render_humanml3d(anim_file):

    # 데이터 로드
    file_ext = Path(anim_file.name).suffix.lower()
    if file_ext == '.npy':
        npy = np.load(anim_file.name, allow_pickle=True)
        if isinstance(npy, np.ndarray) and npy.dtype == np.dtype('O') and isinstance(npy.item(), dict):
            if 'motion' in npy.item():
                data = npy.item()['motion']
        elif isinstance(npy, dict) and 'motion' in npy:
            data = npy['motion']
            
    elif file_ext == '.npz':
        npz = np.load(anim_file.name, allow_pickle=True)
        # humanml3d 포맷에서 'motion' 또는 'poses' 키 사용
        if 'motion' in npz:
            data = npz['motion']
        elif 'poses' in npz:
            data = npz['poses']
        else:
            data = None
    else:
        return '<div>지원하지 않는 파일 형식입니다.</div>'

    if data is None:
        return '<div>데이터를 읽을 수 없습니다.</div>'

    return humanml3d_viewer(data)


def humanml3d_viewer(data):
    import numpy as np
    import json

    # (F, J, 3) 또는 (J, 3, F) 형태 지원
    if data.ndim == 4:
        data = data[0]
    if data.ndim == 3:
        if data.shape[0] == 22 and data.shape[1] == 3:
            data = np.transpose(data, (2, 0, 1))
        elif data.shape[1] == 22 and data.shape[2] == 3:
            pass
        else:
            return '<div>지원하지 않는 데이터 형태입니다.</div>'
    else:
        return '<div>지원하지 않는 데이터 차원입니다.</div>'

    # NaN/Inf 방지
    data = np.nan_to_num(data)
    frames, joints, _ = data.shape

    # 본 연결 정보 (SMPL 22본 기준)
    skeleton = [
        [0,1],[1,4],[4,7],[7,10],
        [0,2],[2,5],[5,8],[8,11],
        [0,3],[3,6],[6,9],[9,12],[12,13],[13,16],[16,18],[18,20],
        [12,14],[14,17],[17,19],[19,21],
        [12,15]
    ]
    all_poses = data.tolist()

    # 중심 계산 (첫 프레임 기준)
    center = np.mean(data[0], axis=0).tolist()

    skeleton_json = json.dumps(skeleton)
    all_poses_json = json.dumps(all_poses)
    center_json = json.dumps(center)

    html = f'''
    <div id="humanml3d_viewer" style="width:100%;height:500px;"></div>
    <script src="https://cdn.jsdelivr.net/npm/three@0.150.1/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.150.1/examples/js/controls/OrbitControls.js"></script>
    <script>
    const container = document.getElementById('humanml3d_viewer');
    container.innerHTML = '';
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x222222);

    // 데이터
    const skeleton = {skeleton_json};
    const all_poses = {all_poses_json};
    const center = {center_json};
    let frame = 0;

    // 카메라를 데이터 중심에 맞춤
    const camera = new THREE.PerspectiveCamera(45, container.offsetWidth/container.offsetHeight, 0.01, 100);
    camera.position.set(center[0], center[1]+1.5, center[2]+10);
    camera.lookAt(center[0], center[1], center[2]);

    const renderer = new THREE.WebGLRenderer({{antialias:true}});
    renderer.setSize(container.offsetWidth, container.offsetHeight);
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.target.set(center[0], center[1], center[2]);
    controls.update();

    // 조명
    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const light = new THREE.DirectionalLight(0xffffff, 0.7);
    light.position.set(center[0], center[1]+2, center[2]+2);
    scene.add(light);

    // 본 라인 생성
    const lineGeometry = new THREE.BufferGeometry();
    const linePositions = new Float32Array(skeleton.length * 2 * 3);
    lineGeometry.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
    const lineMaterial = new THREE.LineBasicMaterial({{color:0x00ffcc}});
    const skeletonLines = new THREE.LineSegments(lineGeometry, lineMaterial);
    scene.add(skeletonLines);

    // 관절 점 생성
    const jointGeometry = new THREE.BufferGeometry();
    const jointPositions = new Float32Array(all_poses[0].length * 3);
    jointGeometry.setAttribute('position', new THREE.BufferAttribute(jointPositions, 3));
    const jointMaterial = new THREE.PointsMaterial({{color:0xffcc00, size:0.05}});
    const joints = new THREE.Points(jointGeometry, jointMaterial);
    scene.add(joints);

    // 애니메이션 루프
    function updateFrame(f) {{
        // 점 위치 갱신
        for(let i=0; i<all_poses[0].length; i++) {{
            jointPositions[i*3+0] = all_poses[f][i][0];
            jointPositions[i*3+1] = all_poses[f][i][1];
            jointPositions[i*3+2] = all_poses[f][i][2];
        }}
        jointGeometry.attributes.position.needsUpdate = true;

        // 라인 위치 갱신
        for(let i=0; i<skeleton.length; i++) {{
            const [a, b] = skeleton[i];
            linePositions[i*6+0] = all_poses[f][a][0];
            linePositions[i*6+1] = all_poses[f][a][1];
            linePositions[i*6+2] = all_poses[f][a][2];
            linePositions[i*6+3] = all_poses[f][b][0];
            linePositions[i*6+4] = all_poses[f][b][1];
            linePositions[i*6+5] = all_poses[f][b][2];
        }}
        lineGeometry.attributes.position.needsUpdate = true;
    }}

    function animate() {{
        requestAnimationFrame(animate);
        updateFrame(frame);
        frame = (frame+1)%all_poses.length;
        renderer.render(scene, camera);
    }}
    animate();

    window.addEventListener('resize',()=>{{
        camera.aspect = container.offsetWidth/container.offsetHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.offsetWidth, container.offsetHeight);
    }});
    </script>
    '''
    return html

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
                    return render_humanml3d(anim)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    return f"SMPL 애니메이션 미리보기 오류: {str(e)}"
            
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