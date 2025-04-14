"""
동작 설명 텍스트를 분석하여 구조화된 정보를 추출하는 모듈
"""

import os
import json
import torch
from typing import Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
from .base import BaseAgent, Message

class PromptAnalyzer(BaseAgent):
    def __init__(self, redis_client, model_name: str = "google/gemma-2b-it"):
        """동작 분석을 위한 프롬프트 분석기 초기화"""
        super().__init__("prompt_analyzer", redis_client)
        self.model_name = model_name
        self._initialize_model()
        
    def _initialize_model(self):
        """모델과 토크나이저 초기화"""
        token = os.getenv("HUGGINGFACE_TOKEN")
        if token:
            login(token)
            
        # 8비트 양자화 설정으로 메모리 사용량 최적화
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=token,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=token,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )

    async def process_message(self, message: Message):
        """메시지 처리"""
        if message.intent == "analyze_prompt":
            prompt = message.content.get("prompt", "")
            
            # 프롬프트 검증
            is_valid, error_msg = self._validate_prompt(prompt)
            if not is_valid:
                await self.send_message(
                    message.sender,
                    "error",
                    {
                        "error": error_msg,
                        "original_prompt": prompt
                    }
                )
                return
            
            # 프롬프트 분석 수행
            result = await self.analyze(prompt)
            
            if result:
                # 워크플로우 플래너에게 결과 전송
                await self.send_message(
                    "workflow_planner",
                    "analysis_result",
                    {
                        "success": True,
                        "result": result,
                        "original_prompt": prompt
                    }
                )
            else:
                # 오류 응답
                await self.send_message(
                    message.sender,
                    "error",
                    {
                        "error": "동작 분석 실패",
                        "original_prompt": prompt
                    }
                )
        
    def _validate_prompt(self, prompt: str) -> Tuple[bool, str]:
        """프롬프트 유효성 검사"""
        if not prompt or not prompt.strip():
            return False, "프롬프트가 비어있습니다."
        
        if len(prompt) < 2:
            return False, "프롬프트가 너무 짧습니다."
            
        if len(prompt) > 100:
            return False, "프롬프트가 너무 깁니다. 100자 이내로 작성해주세요."
        
        return True, ""
        
    async def analyze(self, text: str) -> Dict[str, Any]:
        """동작 설명 텍스트 분석"""
        try:
            prompt = self._create_prompt(text)
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=1024,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    temperature=0.7
                )
                
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            try:
                json_str = response.split("Output:")[-1].strip()
                return json.loads(json_str)
            except json.JSONDecodeError:
                print("JSON 파싱 오류. 원본 응답:", response)
                return {}
                
        except Exception as e:
            print(f"동작 분석 중 오류 발생: {e}")
            return {}
            
    def _create_prompt(self, text: str) -> str:
        """분석을 위한 프롬프트 생성"""
        return f"""Task: Analyze the Korean motion description and extract structured information.
Please provide a detailed analysis in the following JSON format:

Expected Output Format:
{{
    "motion": {{
        "type": "walking/running/dancing/etc",
        "description": "기본 동작 설명",
        "direction": "forward/backward/up/down/etc",
        "path": "straight/curved/circular/etc"
    }},
    "speed": {{
        "level": "slow/medium/fast",
        "value": "1-10",
        "variation": "constant/accelerating/decelerating"
    }},
    "intensity": {{
        "level": "low/medium/high",
        "value": "1-10",
        "effort": "light/normal/strong"
    }},
    "keywords": ["주요", "영어", "키워드들"]
}}

Example:
Input: "느리게 걷기"
Output: {{
    "motion": {{
        "type": "walking",
        "description": "basic walking motion",
        "direction": "forward",
        "path": "straight"
    }},
    "speed": {{
        "level": "slow",
        "value": "2",
        "variation": "constant"
    }},
    "intensity": {{
        "level": "low",
        "value": "3",
        "effort": "light"
    }},
    "keywords": ["slow", "walking", "gentle"]
}}

Now analyze this input:
Input: "{text}"
Output:""" 