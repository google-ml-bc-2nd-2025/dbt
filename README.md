# gobc_web


실행 방법 (로컬 개발 환경 기준)

# 1. Redis 서버 실행
redis-server

# 2. 가상환경 활성화 및 의존성 설치
.\venv\Scripts\activate
pip install -r requirements.txt

# 3. Celery 워커 실행
python src/run_celery.py

# 4. FastAPI 서버 실행
python src/app.py




멀티 에이전트 구성 (MVP 기준)

PromptAnalyzer	: 사용자 프롬프트 유효성 검사, 감정/속도/스타일/방향 등 요소 분석
WorkflowPlanner	: 분석된 프롬프트를 바탕으로 애니메이션 생성 또는 후속 작업 결정 (현재 MVP에서는 단일 경로)
TaskExecutor	: 실제 애니메이션 생성 태스크 실행 (현재는 더미 응답 반환)


현재는 Celery + Redis 기반 비동기 시스템에 PromptAnalyzer → WorkflowPlanner → TaskExecutor 순으로 메시지가 흐르고 있으며,
향후 이펙트 추가, 스킨 자동 생성 등을 위해 WorkflowPlanner 로직이 확장될 예정입니다.