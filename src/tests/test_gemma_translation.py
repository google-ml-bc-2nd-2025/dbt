"""
Gemma 모델을 사용한 한글 동작 설명 번역 및 분석 테스트
"""

import torch
from transformers import GemmaForCausalLM, GemmaTokenizer
import json
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gemma():
    # 모델 및 토크나이저 로드
    model_name = "google/gemma-2b"
    
    print("토크나이저 로딩 중...")
    tokenizer = GemmaTokenizer.from_pretrained(model_name)
    
    print("모델 로딩 중...")
    model = GemmaForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16
    )

    # 테스트용 한글 프롬프트
    test_texts = [
        "슬프게 걷기",
        "신나게 뛰기",
        "화나서 발구르기",
        "기쁘게 스킵하기"
    ]

    print("\n동작 분석 테스트 시작...")
    for text in test_texts:
        prompt = f"""Translate and analyze the following Korean motion description: "{text}"

        1. First, translate it to English
        2. Then, analyze the motion and emotion
        3. Provide the result in the following JSON format:
        {{
            "translation": "English translation",
            "motion": {{
                "type": "basic motion type (walk/run/jump/etc)",
                "style": "how the motion is performed",
                "intensity": "low/medium/high",
                "speed": "slow/medium/fast"
            }},
            "emotion": {{
                "type": "primary emotion (happy/sad/angry/neutral)",
                "intensity": "low/medium/high"
            }}
        }}

        Provide only the JSON output, no additional text."""
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,
                num_return_sequences=1
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"\n입력: {text}")
        try:
            # 응답에서 JSON 부분만 추출
            json_str = response[response.find("{"):response.rfind("}")+1]
            result = json.loads(json_str)
            print("분석 결과:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        except json.JSONDecodeError as e:
            print("JSON 파싱 실패. 원본 응답:")
            print(response.strip())

if __name__ == "__main__":
    # GPU 상태 확인
    print("GPU 상태 확인:")
    print("CUDA 사용 가능:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU 모델:", torch.cuda.get_device_name(0))
        print("메모리 정보:")
        try:
            import subprocess
            subprocess.run(["nvidia-smi"])
        except:
            print("nvidia-smi 실행 실패")

    # 테스트 실행
    test_gemma() 