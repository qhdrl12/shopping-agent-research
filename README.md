# 🛍️ Enhanced Shopping Agent

**LangChain, LangGraph, MCP를 활용한 지능형 쇼핑 에이전트 연구**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-최신-green.svg)](https://python.langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-웹앱-red.svg)](https://streamlit.io/)

## 📋 프로젝트 개요

Enhanced Shopping Agent는 사용자의 쇼핑 질문에 대해 웹 검색과 스크래핑을 통해 정확하고 유용한 정보를 제공하는 AI 에이전트입니다. 다단계 워크플로우를 통해 깊이 있는 분석과 개인화된 추천을 제공합니다.

### 🎯 주요 특징

- **지능형 질문 분석**: Structured Output을 통한 체계적 질문 이해
- **멀티소스 정보 수집**: Tavily 검색 + Firecrawl 스크래핑 통합
- **단계별 진행 표시**: 실시간 처리 상태 및 결과 피드백
- **강력한 에러 처리**: API 재시도 메커니즘 및 우아한 실패 처리
- **설정 기반 최적화**: 사용 목적에 따른 맞춤형 설정 지원

## 🔄 에이전트 워크플로우

```
사용자 질문 입력
     ↓
1단계 [질문 분석]
     사용자 의도 파악
     검색 키워드 추출
     스크래핑 대상 식별
     ↓
2단계 [웹 검색]
     Tavily API로 정보 검색
     관련 웹사이트 발견
     기본 URL 수집
     ↓
3단계 [웹 스크래핑]
     Firecrawl API로 페이지 수집
     상품 정보 추출
     가격 및 리뷰 분석
     ↓
4단계 [최종 답변 생성]
     정보 통합 및 분석
     개인화된 추천 생성
     실용적 조언 제공
     ↓
결과 출력 및 피드백
```

## 🚀 설치 및 사용법

### 1. 설치

```bash
# 저장소 복제
git clone <repository-url>
cd scrap-agent

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 설정

```bash
# 환경 변수 파일 생성
cp .env.example .env

# API 키 설정
# .env 파일 내용을 다음과 같이 설정하세요:
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
FIRECRAWL_API_KEY=your_firecrawl_api_key
```

### 3. 실행

#### 웹 인터페이스 (Streamlit)
```bash
streamlit run app.py
```

## 📁 프로젝트 구조

### 핵심 모듈

- **`enhanced_shopping_agent.py`**: 메인 에이전트 클래스
- **`agent_config.py`**: 설정 및 프리셋 관리
- **`text_processing.py`**: 텍스트 처리 및 상품정보 추출
- **`retry_helper.py`**: API 재시도 메커니즘 클래스

### 외부 도구

- **Tavily**: 웹 검색 및 정보 수집
- **Firecrawl**: 웹 페이지 스크래핑 및 콘텐츠 추출
- **OpenAI GPT**: 자연어 처리 및 답변 생성

## 💻 코드 사용 예시

```python
import asyncio
from agent.enhanced_shopping_agent import build_enhanced_agent

async def main():
    # 에이전트 빌드
    agent = await build_enhanced_agent("default")
    
    # 초기 상태
    initial_state = {
        "user_query": "겨울용 패딩 재킷 추천해줘. 10만원 예산으로 따뜻하고 가벼운 제품.",
        "messages": [],
        "processing_status": "시작"
    }
    
    # 실행
    result = await agent.ainvoke(initial_state)
    
    # 결과 출력
    print(result["final_answer"])

asyncio.run(main())
```

## 🧪 테스트

### 기본 테스트
```bash
# 전체 테스트 실행
pytest tests/ -v

# 특정 테스트
pytest tests/test_config_comparison.py -v
```

## 📚 참고 자료

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Tavily API](https://tavily.com/)
- [Firecrawl API](https://firecrawl.dev/)
