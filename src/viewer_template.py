"""
3D 모델 뷰어 HTML 템플릿을 생성하는 모듈
"""
import os
import shutil
from pathlib import Path

def create_viewer_html(output_path):
    """
    Three.js 기반 3D 뷰어 HTML 생성하여 지정된 경로에 저장
    
    Args:
        output_path: HTML 파일이 저장될 경로
    
    Returns:
        생성된 HTML 파일 경로
    """
    # 템플릿 HTML 파일 경로
    template_path = Path(__file__).parent / "static" / "viewer_template.html"
    
    # 템플릿 파일이 존재하지 않으면 생성
    if not template_path.exists():
        # 템플릿 디렉토리 확인
        template_path.parent.mkdir(exist_ok=True)
        
        # 초기 템플릿 생성 코드를 별도 함수로 분리
        create_initial_template(template_path)
    
    # 템플릿 파일을 출력 경로로 복사
    shutil.copy2(template_path, output_path)
    
    return output_path

def create_initial_template(template_path):
    """
    템플릿 HTML 파일을 처음 생성하는 함수
    
    Args:
        template_path: 생성할 템플릿 파일 경로
    """
    with open(template_path, "w") as f:
        f.write("""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>애니메이션 뷰어</title>
    <style>
        body { margin: 0; overflow: hidden; }
        #container { width: 100%; height: 100vh; }
        .info { 
            position: absolute; 
            top: 10px; 
            left: 10px; 
            background: rgba(0,0,0,0.7); 
            color: white; 
            padding: 10px; 
            font-family: monospace;
            border-radius: 5px;
        }
        .loading {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: white;
            font-size: 18px;
            background: rgba(0,0,0,0.7);
            padding: 20px;
            border-radius: 10px;
        }
        .error {
            color: #ff4444;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div id="container"></div>
    <div class="info">애니메이션 뷰어</div>
    <div id="loading" class="loading">모델 로딩 중...</div>

    <!-- 스크립트 호출 순서와 로드 방식 변경 -->
    <script type="importmap">
    {
        "imports": {
            "three": "https://cdn.jsdelivr.net/npm/three@0.152.2/build/three.module.js",
            "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.152.2/examples/jsm/"
        }
    }
    </script>
    
    <script type="module">
        import * as THREE from 'three';
        import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
        import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
        import { BVHLoader } from 'three/addons/loaders/BVHLoader.js';
        
        // URL 파라미터 가져오기
        const urlParams = new URLSearchParams(window.location.search);
        const skinModelUrl = urlParams.get('skin');
        const animModelUrl = urlParams.get('anim');
        const animType = urlParams.get('animType') || 'glb'; // 애니메이션 타입 (기본값: glb)
        
        // 디버깅을 위한 URL 출력
        console.log('스킨 모델 URL:', skinModelUrl);
        console.log('애니메이션 URL:', animModelUrl);
        console.log('애니메이션 타입:', animType);

        // 로딩 상태 업데이트 함수
        function updateLoadingStatus(message, isError = false) {
            const loadingElement = document.getElementById('loading');
            loadingElement.innerHTML = message;
            if (isError) {
                loadingElement.classList.add('error');
            } else {
                loadingElement.classList.remove('error');
            }
        }

        // ... 이하 생략 (HTML 파일 내용과 동일) ...
    </script>
</body>
</html>""")