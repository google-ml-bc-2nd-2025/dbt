"""
Celery 설정 파일
"""

from celery import Celery
from kombu import Exchange, Queue

# Redis 연결 설정
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0

BROKER_URL = f'redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}'
RESULT_BACKEND = f'redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}'

# Celery 앱 생성
celery_app = Celery(
    'motion_generator',
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
    broker_transport_options={'visibility_timeout': 3600},
    broker_connection_retry_on_startup=True,
    include=['src.tasks']
)

# 기본 설정
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Asia/Seoul',
    enable_utc=True,
    broker_transport='redis',
    result_backend_transport='redis',
    task_track_started=True,
    task_time_limit=3600,
    worker_max_tasks_per_child=50,
    broker_connection_max_retries=10
)

# 큐 설정
default_exchange = Exchange('default', type='direct')
animation_exchange = Exchange('animation', type='direct')

default_queue = Queue('default', default_exchange, routing_key='default')
animation_queue = Queue('animation', animation_exchange, routing_key='animation')

celery_app.conf.task_queues = (default_queue, animation_queue)
celery_app.conf.task_default_queue = 'default'

# 태스크 라우팅 설정
celery_app.conf.task_routes = {
    'src.tasks.generate_animation': {'queue': 'animation'}
} 