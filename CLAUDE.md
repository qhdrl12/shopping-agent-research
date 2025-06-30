# 스크래핑 에이전트 프로젝트

## 프로젝트 개요
Python, LangChain, LangGraph, MCP를 활용한 고품질 쇼핑 에이전트 구축 프로젝트입니다.

## 아키텍처 설계

### 1. Enhanced Shopping Agent ✨
- **목적**: 사전 검색+스크래핑 파이프라인
- **구현**: `src/agent/enhanced_shopping_agent.py`
- **도구**: Firecrawl 직접 클라이언트, Tavily API
- **특징**: 설정 기반 관리

### 2. 기존 단일 에이전트 (React Agent)
- **목적**: 단순 도구 활용 및 기본 질의응답
- **구현**: `src/agent/shopping_react_agent.py`
- **도구**: Firecrawl, Playwright, Filesystem (MCP)

### 3. 멀티 에이전트 시스템 (계획)

#### 3.1 질문 분석 에이전트 (Question Analyzer)
- **역할**: 사용자 질문 분석 및 구체화
- **기능**: 
  - 질문 의도 분석
  - 필요한 정보 식별
  - 재질문 생성
  - 검색 키워드 추출

#### 3.2 정보 수집 에이전트 (Information Collector)
- **역할**: 다양한 소스에서 정보 수집
- **도구**:
  - Firecrawl: 웹 크롤링
  - Tavily: 웹 검색
  - MCP 통합 도구들

#### 3.3 쇼핑 전문 에이전트 (Shopping Specialist)
- **역할**: 쇼핑 특화 분석 및 추천
- **기능**:
  - 상품 정보 추출
  - 가격 비교
  - 리뷰 분석
  - 추천 생성

#### 3.4 오케스트레이터 (Orchestrator)
- **역할**: 멀티 에이전트 조율
- **기능**:
  - 에이전트 간 워크플로우 관리
  - 결과 통합
  - 품질 보증

## 프로젝트 구조
```
src/
├── agent/
│   ├── enhanced_shopping_agent.py  # ✨ Enhanced Shopping Agent (메인)
│   └── shopping_react_agent.py     # 기존 단일 React 에이전트
├── config/
│   └── agent_config.py            # ✨ 설정 관리
├── tools/                         # 레거시 도구들 (참고용)
│   ├── tavily.py                  # Tavily 검색 도구
│   ├── retriever.py               # 벡터 검색 도구  
│   └── datetime.py                # 시간 도구
├── utils/                         # ✨ 새로 추가된 유틸리티
│   ├── text_processing.py         # 텍스트 처리 및 상품 정보 추출
│   └── retry_helper.py            # API 재시도 메커니즘
└── tests/
    ├── test_enhanced_agent.py     # 메인 에이전트 테스트
    └── test_config_comparison.py  # 설정별 성능 비교 테스트
```

## 주요 기능

### 1. 지능형 질문 분석
- 복합 질문 분해
- 모호한 질문 구체화
- 재질문 생성

### 2. 멀티 소스 정보 수집
- Firecrawl을 통한 웹 크롤링
- Tavily를 통한 검색 엔진 활용
- MCP를 통한 다양한 도구 통합

### 3. 쇼핑 특화 처리
- 상품 정보 추출 및 정규화
- 가격 비교 분석
- 리뷰 감정 분석
- 개인화 추천

### 4. 품질 보증 시스템
- 정보 신뢰성 검증
- 답변 완성도 평가
- 자동 개선 메커니즘

## 워크플로우

### 단일 에이전트 플로우
```
사용자 질문 → React Agent → 도구 실행 → 결과 반환
```

### 멀티 에이전트 플로우
```
사용자 질문 
→ 질문 분석 에이전트 (질문 구체화)
→ 정보 수집 에이전트 (데이터 수집)
→ 쇼핑 전문 에이전트 (분석 및 추천)
→ 오케스트레이터 (결과 통합)
→ 최종 답변
```

## 기술 스택

### 핵심 프레임워크
- **LangChain**: 언어 모델 체인 구성
- **LangGraph**: 에이전트 상태 그래프 관리
- **MCP (Model Context Protocol)**: 다양한 도구 통합

### 외부 서비스
- **Firecrawl**: 웹 크롤링 서비스
- **Tavily**: 검색 API
- **OpenAI GPT**: 언어 모델

### 도구 및 라이브러리
- **Streamlit**: 웹 인터페이스
- **FastMCP**: MCP 클라이언트
- **Playwright**: 브라우저 자동화

## 개발 단계

### Phase 1: 기본 구조 구축 ✅
- [x] 프로젝트 분석
- [x] 아키텍처 설계  
- [x] CLAUDE.md 문서화
- [x] Enhanced Shopping Agent 구현
- [x] 사전 검색+스크래핑 파이프라인
- [x] 설정 기반 관리 시스템

### Phase 2: 코드 품질 개선 ✅
- [x] 유틸리티 함수 분리 및 모듈화
- [x] 상세한 기능별 주석 추가
- [x] API 재시도 메커니즘 구현
- [x] 테스트 구조 체계화 (pytest 기반)
- [x] 프롬프트 엔지니어링 고도화

### Phase 3: 향후 확장 계획
- [ ] 멀티 에이전트 시스템 (질문 분석, 정보 수집, 전문가 에이전트)
- [ ] 개인화 기능 (사용자 선호도 학습)
- [ ] 성능 최적화 (캐싱, 병렬 처리)
- [ ] 고급 에러 처리 및 모니터링


## 실행 방법

### 환경 설정
```bash
# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
cp .env.example .env
# OPENAI_API_KEY, FIRECRAWL_API_KEY, TAVILY_API_KEY 설정

# Streamlit 애플리케이션 실행
streamlit run app.py
```

### 테스트 실행
```bash
# Enhanced Shopping Agent 기본 테스트
python tests/test_enhanced_agent.py

# 설정별 성능 비교 테스트
python tests/test_config_comparison.py

# pytest를 사용한 체계적 테스트
pytest tests/test_config_comparison.py -v

# 기존 에이전트 테스트
python src/agent/shopping_react_agent.py
```

## 📊 코드 품질 개선 사항

### ✅ 완료된 개선사항

#### 1. **모듈화 및 구조 개선**
- `utils/text_processing.py`: 텍스트 처리 및 상품 정보 추출 로직 분리
- `utils/retry_helper.py`: API 재시도 메커니즘 추가
- 코드 중복 제거 및 재사용성 향상

#### 2. **상세한 기능별 주석**
- 모든 주요 메서드에 Args, Returns, Process 설명 추가
- 4단계 워크플로우 각 단계별 상세 문서화
- 핵심 출력 변수 및 처리 로직 명시

#### 3. **안정성 향상**
- Firecrawl API 502 에러 대응 재시도 로직
- 단계별 에러 처리 및 로깅 개선
- 우아한 실패 처리 (일부 단계 실패 시에도 계속 진행)

#### 4. **테스트 구조 체계화**
- pytest 기반 `test_config_comparison.py` 리팩토링
- 설정별 성능 비교 테스트 자동화
- 단위 테스트와 통합 테스트 분리

#### 5. **프롬프트 엔지니어링 고도화**
- 질문 분석 프롬프트: 검색 키워드 품질 집중 개선
- 최종 답변 프롬프트: 전문 컨설턴트 수준의 체계적 구조

### 🎯 품질 목표

#### 1. 정확성
- 정보 신뢰성 95% 이상
- 답변 완성도 90% 이상
- 검색 키워드 관련성 향상

#### 2. 성능
- 응답 시간 30초 이내
- API 재시도로 안정성 확보
- 설정 기반 최적화 지원

#### 3. 사용성
- 직관적 인터페이스
- 실시간 단계별 진행 상황 표시
- 향상된 에러 복구 메커니즘

## 확장 계획

### 1. 도메인 확장
- 패션 → 전자제품 → 생활용품
- 해외 쇼핑몰 지원
- B2B 상품 지원

### 2. 기능 확장
- 가격 알림 시스템
- 위시리스트 관리
- 자동 주문 시스템

### 3. 플랫폼 확장
- 모바일 앱
- 챗봇 통합
- API 서비스

## 참고 자료
- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [MCP Specification](https://modelcontextprotocol.io/)
- [Firecrawl API](https://firecrawl.dev/)
- [Tavily API](https://tavily.com/)