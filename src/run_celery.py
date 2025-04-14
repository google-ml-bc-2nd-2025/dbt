"""
Celery 워커 실행 스크립트
"""

from celery_config import celery_app

if __name__ == '__main__':
    # Celery 워커 실행
    celery_app.worker_main([
        'worker',
        '--loglevel=INFO',
        '--queues=default,animation',
        '--pool=solo' 
    ])
