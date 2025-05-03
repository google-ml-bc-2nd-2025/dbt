# i18n.py
translations = {
    "en": {
        "tab_title_01": "Motion Generation",
        "tab_title_01_desc": "Enter what you want a motion in the prompt and press the send button.",
        "tab_title_02": "Motion Viewer",
        "tab_title_02_desc": "Specify a model with a skin and select the desired motion data.",
        "tab_title_03": "Motion Dataset Building",
        "tab_title_03_desc": "Place the files to be used for training in the designated folder and add labeling according to the specifications.",
        "viewport_title": "Viewport",
        "viewport_desc_gen": "Results generated from the prompt will be displayed.",
        "label_prompt" : "Prompt Input",
        "label_prompt_desc" : "Enter the prompt for motion generation.",
        "bnt_send" : "Send",
        "btn_download" : "Download",
        "desc_failed_to_generate_motion" : "Failed to apply motion",
        "desc_viewer" : """
                ## How to use
                1. Upload the animation file (GLB, BVH, NPY(humanml3d format)).
                2. If it is a glb file, you can upload a separate skin model.
                3. Click the 'Apply and View' button.
                4. Check the model with the applied animation in the right panel.
                
                **Note**: 
                - GLB Animation: The rigging structure of both models must be the same. Immediate confirmation possible
                """,
        "viewport_desc_viewer" : "Upload the model and click the 'Apply' button",
        "btn_refresh" : "Refresh",
        "desc_dataset_input": "Motion label (Enter the description and time for the motion interval on each line. Example) Walk forward #0.0#0.5 (newline) Turn back. #0.5#1.2)",
        "desc_dataset_input_desc": "Enter a description of the motion.",
        "btn_update": "Update Labeling",
        "btn_build_dataset": "Create Training File",
        "btn_build_dataset_all": "Create Integrated Training File",
        "label_file_list":"Training Dataset List",
        "label_dataset_path": "Dataset Path",
        "desc_select_motion": "Select a motion."
    },
    "ko": {
        "tab_title_01": "모션 생성",
        "tab_title_01_desc": "원하는 모션션을 프롬프트에 입력 후 전송 버튼을 누르세요.",
        "tab_title_02": "모션 뷰어",
        "tab_title_02_desc": "스킨이 있는 모델을 지정 후 원하는 모션 데이터를 선택하세요.",
        "tab_title_03": "모션 학습 데이터셋 생성",
        "tab_title_03_desc": "학습에 사용할 파일들을 지정된 폴더에 넣고 규격에 맞게 라벨링을 추가하세요.",
        "viewport_title": "뷰포트",
        "viewport_desc_gen": "프롬프트로 생성된 결과가 출력됩니다.",
        "label_prompt" : "프롬프트 입력",
        "label_prompt_desc" : "모션 생성을 위한 프롬프트를 입력하세요.",
        "bnt_send" : "전송",
        "btn_download" : "다운로드",
        "desc_failed_to_generate_motion" : "모션 적용 실패",
        "desc_viewer" : """
                ## 사용 방법
                1. 애니메이션 파일(GLB, BVH, NPY(humanml3d format))을 업로드합니다.
                2. glb 파일일 경우 별도의 스킨 모델을 업로드할 수 있습니다.
                3. '적용 및 보기' 버튼을 클릭합니다.
                4. 오른쪽 패널에서 애니메이션이 적용된 모델을 확인합니다.
                
                **참고**: 
                - GLB 애니메이션: 두 모델의 리깅(뼈대) 구조가 동일해야 합니다. 즉시 확인 가능
                """,
        "viewport_desc_viewer" : "모델을 업로드하고 'Apply' 버튼을 클릭하세요",
        "btn_refresh" : "새로고침",
        "desc_dataset_input": "모션 라벨(한 줄 마다 머션 구간에 대한 설명과 시간을 입력. 예) 앞으로 걸어가다 #0.0#0.5(줄바꿈)뒤로 돌아간다. #0.5#1.2)",
        "desc_dataset_input_desc": "모션에 대한 설명을 입력하세요.",
        "btn_update": "라벨링 업데이트",
        "btn_build_dataset": "학습 파일 생성",
        "btn_build_dataset_all": "학습 파일 통합 생성",
        "label_file_list":"학습 데이터셋 목록",
        "label_dataset_path": "데이터셋 경로",
        "desc_select_motion": "모션을 선택하세요."
    }
}
