"""
Celery 태스크 정의 파일
"""

from celery import shared_task
from pathlib import Path
from agents import AgentController
from celery_config import celery_app

# 에이전트 컨트롤러 인스턴스 생성
controller = AgentController(Path(__file__).parent / "static" / "models")

@celery_app.task(bind=True, name='src.tasks.generate_animation', queue='animation')
def generate_animation(self, prompt: str) -> dict:
    """
    애니메이션 생성을 위한 Celery 태스크
    
    Args:
        prompt (str): 사용자 입력 프롬프트
        
    Returns:
        dict: 태스크 실행 결과
    """
    try:
        # 태스크 상태 업데이트
        self.update_state(state='PROGRESS', meta={'status': '프롬프트 분석 중...'})
        
        # 프롬프트 처리
        success, result, error = controller.process_prompt(prompt)
        
        if not success:
            return {
                'status': 'ERROR',
                'error': error
            }
            
        return {
            'status': 'SUCCESS',
            'result': result
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'error': str(e)
        } 