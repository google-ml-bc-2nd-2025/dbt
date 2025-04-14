"""
Gemma 모델 CPU 테스트
메모리 사용량을 최소화하고 CPU에서 실행 가능하도록 설정
"""

import asyncio
import torch
from transformers import GemmaForCausalLM, GemmaTokenizer
import logging
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GemmaTest:
    def __init__(self):
        self.model_name = "google/gemma-2b"  # 더 작은 모델 사용
        self.device = "cpu"
        
    async def setup(self):
        """모델과 토크나이저 초기화"""
        try:
            logger.info("토크나이저 로딩 중...")
            self.tokenizer = GemmaTokenizer.from_pretrained(self.model_name)
            
            logger.info("모델 로딩 중... (CPU에서는 시간이 좀 걸릴 수 있습니다)")
            self.model = GemmaForCausalLM.from_pretrained(
                self.model_name,
                device_map="cpu",
                torch_dtype=torch.float32,  # CPU에서는 float32 사용
                low_cpu_mem_usage=True
            )
            
            logger.info("모델 로딩 완료")
            return True
        except Exception as e:
            logger.error(f"모델 로딩 중 오류 발생: {e}")
            return False
            
    async def analyze_emotion(self, text: str) -> str:
        """텍스트에서 감정 분석"""
        try:
            prompt = f"""Analyze the emotion in this action: "{text}"
            Provide the emotion as one of: happy, sad, angry, neutral.
            Answer with just the emotion word."""
            
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.1,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.strip()
            
        except Exception as e:
            logger.error(f"감정 분석 중 오류 발생: {e}")
            return f"Error: {str(e)}"
            
async def run_tests():
    """테스트 실행"""
    tester = GemmaTest()
    if not await tester.setup():
        logger.error("모델 설정 실패")
        return
        
    test_cases = [
        "A person is walking happily in the park.",
        "Someone is crying after losing their phone.",
        "An angry customer is complaining at the counter.",
        "A student is sitting quietly in the library."
    ]
    
    logger.info("\n=== 감정 분석 테스트 시작 ===")
    for text in test_cases:
        logger.info(f"\n입력: {text}")
        emotion = await tester.analyze_emotion(text)
        logger.info(f"감정: {emotion}")
        
if __name__ == "__main__":
    # 시스템 정보 출력
    logger.info("=== 시스템 정보 ===")
    logger.info(f"PyTorch 버전: {torch.__version__}")
    logger.info(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    logger.info(f"사용 장치: CPU")
    
    # 테스트 실행
    asyncio.run(run_tests()) 