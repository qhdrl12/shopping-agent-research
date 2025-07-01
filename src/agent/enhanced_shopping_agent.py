"""
Enhanced Shopping Agent with Pre-Search and Pre-Scraping
LangGraph를 사용한 단일 에이전트 + 사전 검색/스크래핑 파이프라인
"""

import os
import asyncio
import json
from typing import Annotated, TypedDict, List, Dict, Any, Optional
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from tavily import TavilyClient
from firecrawl import FirecrawlApp
from dotenv import load_dotenv
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.agent_config import AgentConfig, get_config
from utils.text_processing import (
    extract_title_from_content,
    extract_product_info_from_content,
    calculate_relevance_score
)
from utils.retry_helper import retry_on_failure

load_dotenv()


class QueryAnalysis(BaseModel):
    """질문 분석 결과를 위한 구조화된 모델"""
    main_product: str = Field(description="주요 상품/카테고리")
    # specific_requirements: Dict[str, str] = Field(description="구체적 요구사항 (색상, 크기, 브랜드 등)")
    price_range: str = Field(description="가격대 정보")
    search_keywords: List[str] = Field(description="검색에 사용할 키워드 리스트", max_items=5)
    target_categories: List[str] = Field(description="대상 카테고리")
    search_intent: str = Field(description="검색 의도 (구매, 비교, 정보수집 등)")


class ShoppingAgentState(TypedDict):
    """Enhanced Shopping Agent 상태 관리"""
    messages: Annotated[list[BaseMessage], add_messages]
    user_query: str
    
    # 질문 분석 결과
    analyzed_query: Dict[str, Any]
    search_keywords: List[str]
    target_categories: List[str]
    
    # 사전 검색 결과
    search_results: List[Dict[str, Any]]
    relevant_urls: List[str]
    
    # 사전 스크래핑 결과
    scraped_content: Dict[str, Any]
    product_data: List[Dict[str, Any]]
    
    # React Agent 컨텍스트
    enriched_context: str
    
    # 최종 결과
    final_answer: str
    processing_status: str
    error_info: Optional[str]


class EnhancedShoppingAgent:
    """
    향상된 쇼핑 에이전트 클래스
    
    LangGraph를 사용하여 구축된 4단계 쇼핑 추천 에이전트:
    1. 질문 분석: 사용자 쿼리를 구조화된 정보로 변환
    2. 사전 검색: Tavily를 통한 관련 정보 수집
    3. 사전 스크래핑: Firecrawl을 통한 상세 콘텐츠 수집
    4. 최종 답변: 수집된 정보를 바탕으로 전문적 추천 제공
    
    Features:
        - Structured output을 통한 안정적인 질문 분석
        - 관련성 점수 기반 URL 선택
        - 설정 기반 API 호출 최적화
        - 단계별 진행 상황 로깅
    """
    
    def __init__(self, config: AgentConfig = None):
        """
        에이전트 초기화
        
        Args:
            config (AgentConfig, optional): 에이전트 설정 객체. 
                                          None인 경우 기본 설정 사용.
        
        Note:
            - OpenAI, Tavily, Firecrawl API 키가 환경변수에 설정되어 있어야 함
            - 설정을 통해 검색/스크래핑 범위 조절 가능
        """
        # 설정 초기화 - 기본값 또는 전달받은 설정 사용
        self.config = config or get_config("default")
        
        # LLM 클라이언트 초기화 (OpenAI GPT 모델)
        self.llm = ChatOpenAI(
            model=self.config.llm_model,
            temperature=self.config.llm_temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # 외부 서비스 클라이언트 초기화
        self.tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        self.firecrawl_client = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
        
        
    def create_workflow(self) -> CompiledStateGraph:
        """
        LangGraph 워크플로우를 생성하고 컴파일합니다.
        
        Returns:
            CompiledStateGraph: 실행 가능한 워크플로우 그래프
            
        Workflow:
            analyze_query → pre_search → pre_scrape → react_agent → END
            
        Note:
            각 노드는 순차적으로 실행되며, 이전 단계의 결과를 다음 단계에서 활용
        """
        workflow = StateGraph(ShoppingAgentState)
        
        # 노드 추가 - 각 단계별 처리 함수 연결
        workflow.add_node("analyze_query", self.analyze_query)    # 1단계: 질문 분석
        workflow.add_node("pre_search", self.pre_search)          # 2단계: 사전 검색
        workflow.add_node("pre_scrape", self.pre_scrape)          # 3단계: 사전 스크래핑
        workflow.add_node("react_agent", self.call_agent)         # 4단계: 최종 답변 생성
        
        # 워크플로우 경로 정의 (선형 실행)
        workflow.set_entry_point("analyze_query")
        workflow.add_edge("analyze_query", "pre_search")
        workflow.add_edge("pre_search", "pre_scrape")
        workflow.add_edge("pre_scrape", "react_agent")
        workflow.add_edge("react_agent", END)
        
        return workflow.compile()
    
    async def analyze_query(self, state: ShoppingAgentState) -> ShoppingAgentState:
        """
        1단계: 사용자 질문을 구조화된 정보로 분석합니다.
        
        Args:
            state (ShoppingAgentState): 현재 에이전트 상태
            
        Returns:
            ShoppingAgentState: 분석 결과가 추가된 상태
            
        Process:
            1. Structured Output을 사용하여 안정적인 파싱
            2. 검색 키워드, 상품 카테고리, 가격대 등 추출
            3. 쇼핑 의도 분석 (구매, 비교, 정보수집 등)
            
        Key Output:
            - analyzed_query: 구조화된 분석 결과
            - search_keywords: 다음 단계에서 사용할 검색 키워드
            - target_categories: 상품 카테고리 정보
        """
        print("\n=== [1/4] 질문 분석 노드 시작 ===")
        user_query = state["user_query"]
        print(f"🎯 분석할 질문: {user_query}")
        
        analysis_prompt = f"""
        당신은 전문 쇼핑 컨설턴트입니다. 사용자의 쇼핑 질문을 심층 분석하여 최적의 상품 검색 전략을 수립해야 합니다.

        🎯 **중요**: search_keywords는 이후 웹 검색과 상품 추천의 핵심이 됩니다. 매우 신중하게 선택하세요.

        **사용자 질문**: "{user_query}"

        **분석 지침**:

        1. **main_product (주요 상품)**: 
           - 사용자가 찾는 정확한 상품명이나 카테고리
           - 예: "패딩 점퍼", "무선 이어폰", "운동화"

        2. **search_keywords (검색 키워드 - 매우 중요!)**: 
           ⚠️ **이 키워드들이 검색 품질을 결정합니다!**
           
           **포함해야 할 키워드 유형:**
           - 핵심 상품명 (예: "패딩", "점퍼", "코트")
           - 구체적 특징 (예: "방수", "경량", "초경량", "구스다운")
           - 브랜드명 (언급된 경우)
           - 용도/시즌 (예: "겨울용", "등산용", "데일리")
           - 성별/연령 (예: "남성", "여성", "아동용")
           - 가격대 키워드 (예: "저렴한", "프리미엄", "가성비")
           
           **키워드 선택 원칙:**
           - 검색 결과의 정확성을 높이는 키워드 우선
           - 너무 일반적이지 않고, 너무 구체적이지도 않은 균형
           - 온라인 쇼핑몰에서 실제 사용되는 검색어
           - 최대 5개까지, 중요도 순으로 배열
           
           **좋은 예시:**
           - "겨울 패딩 추천" → ["겨울패딩", "롱패딩", "다운재킷", "방한복", "아우터"]
           - "무선 이어폰" → ["무선이어폰", "블루투스이어폰", "에어팟", "TWS이어폰", "넥밴드"]

        3. **price_range (가격대)**:
           - 구체적 금액이 언급된 경우: "10만원 이하", "50-100만원"
           - 추상적 표현의 경우: "저렴한", "가성비", "프리미엄"
           - 언급 없으면: "가격 정보 없음"

        4. **target_categories (대상 카테고리)**:
           - 패션, 전자제품, 생활용품, 스포츠/레저, 뷰티, 가전, 자동차, 도서 등
           - 주 카테고리와 서브 카테고리 포함

        5. **search_intent (검색 의도)**:
           - "구매": 바로 구매하려는 의도
           - "비교": 여러 상품을 비교하려는 의도  
           - "정보수집": 상품에 대한 정보를 얻으려는 의도
           - "추천": 추천을 받으려는 의도

        **분석 시 고려사항**:
        - 사용자의 암묵적 요구사항 파악 (예: "회사원" → "비즈니스 캐주얼")
        - 계절성 고려 (예: 겨울 → 방한 제품)
        - 트렌드 반영 (예: "MZ세대 인기" → "트렌디한")
        - 실용성 vs 심미성 균형

        위 지침을 바탕으로 사용자 질문을 정확하고 상세하게 분석해주세요.
        """
        
        try:
            # Function calling 방식으로 structured output 사용
            structured_llm = self.llm.with_structured_output(QueryAnalysis, method="function_calling")
            analysis_result = await structured_llm.ainvoke([HumanMessage(content=analysis_prompt)])
            
            # Pydantic 모델을 dict로 변환
            analyzed_data = analysis_result.model_dump()
            
            state["analyzed_query"] = analyzed_data
            state["search_keywords"] = analyzed_data.get("search_keywords", [])
            state["target_categories"] = analyzed_data.get("target_categories", [])
            state["processing_status"] = "질문 분석 완료"
            
            print(f"✅ 분석 완료:")
            print(f"   - 주요 상품: {analyzed_data.get('main_product')}")
            print(f"   - 가격대: {analyzed_data.get('price_range')}")
            print(f"   - 검색 키워드: {analyzed_data.get('search_keywords')}")
            print(f"   - 검색 의도: {analyzed_data.get('search_intent')}")
            
        except Exception as e:
            print(f"❌ 질문 분석 실패: {str(e)}")
            state["error_info"] = f"질문 분석 실패: {str(e)}"
            state["processing_status"] = "질문 분석 실패"
            # 기본값 설정
            state["search_keywords"] = [user_query]
            state["target_categories"] = ["일반"]
            print(f"🔧 기본값 설정: 키워드=[{user_query}], 카테고리=[일반]")
            
        return state
    
    async def pre_search(self, state: ShoppingAgentState) -> ShoppingAgentState:
        """
        2단계: Tavily API를 사용하여 관련 정보를 검색합니다.
        
        Args:
            state (ShoppingAgentState): 분석된 질문 정보가 포함된 상태
            
        Returns:
            ShoppingAgentState: 검색 결과가 추가된 상태
            
        Process:
            1. 추출된 키워드들을 사용하여 웹 검색 수행
            2. 설정에 따른 검색 개수 제한 (API 호출 최적화)
            3. 관련성 점수 계산 및 결과 정렬
            4. 쇼핑몰 도메인 우선순위 적용
            
        Key Output:
            - search_results: 관련성 점수가 포함된 검색 결과
            - relevant_urls: 다음 단계 스크래핑 대상 URL 목록
            
        Note:
            검색 품질이 최종 추천 품질에 직접적 영향을 미치는 핵심 단계
        """
        print("\n=== [2/4] 사전 검색 노드 시작 ===")
        search_keywords = state["search_keywords"]
        print(f"🔍 검색 키워드: {search_keywords}")
        search_results = []
        relevant_urls = []
        
        try:
            # 설정에 따른 검색 개수 제한
            max_keywords = self.config.search.max_keywords_to_search
            max_results_per_keyword = self.config.search.max_results_per_keyword
            total_max_results = self.config.search.total_max_search_results
            
            # 키워드 우선순위 정렬 (길이가 적당하고 구체적인 키워드 우선)
            sorted_keywords = sorted(search_keywords, key=lambda x: (len(x.split()), -len(x)))
            
            total_results_count = 0
            
            # 키워드별 검색 수행 (제한된 개수)
            for keyword in sorted_keywords[:max_keywords]:
                if total_results_count >= total_max_results:
                    break
                    
                # 쇼핑 키워드 추가 (설정에 따라)
                if self.config.search.add_shopping_keywords:
                    search_query = f"{keyword} 쇼핑 구매 추천"
                else:
                    search_query = keyword
                
                remaining_slots = min(
                    max_results_per_keyword, 
                    total_max_results - total_results_count
                )
                
                if remaining_slots <= 0:
                    break
                
                response = self.tavily_client.search(
                    query=search_query,
                    search_depth=self.config.search.search_depth,
                    max_results=remaining_slots
                )
                
                
                for result in response.get("results", []):
                    if total_results_count >= total_max_results:
                        break
                        
                    search_results.append({
                        "keyword": keyword,
                        "title": result.get("title"),
                        "url": result.get("url"),
                        "content": result.get("content"),
                        "score": result.get("score", 0),
                        "relevance_score": self._calculate_relevance_score(result, keyword)
                    })
                    
                    if result.get("url"):
                        relevant_urls.append(result["url"])
                        
                    total_results_count += 1
            
            # 관련성 점수로 정렬
            search_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            
            state["search_results"] = search_results
            state["relevant_urls"] = list(set(relevant_urls))  # 중복 제거
            state["processing_status"] = f"사전 검색 완료 ({len(search_results)}개 결과)"
            
            print(f"✅ 검색 완료: {len(search_results)}개 결과, {len(relevant_urls)}개 URL 발견")
            if search_results:
                print(f"   - 최고 점수 결과: {search_results[0]['title'][:50]}...")
            
            # 발견된 URL 리스트 표시
            if relevant_urls:
                print(f"🔗 발견된 URL 목록:")
                for i, url in enumerate(relevant_urls[:5], 1):  # 최대 5개까지 표시
                    print(f"   {i}. {url}")
                if len(relevant_urls) > 5:
                    print(f"   ... 외 {len(relevant_urls) - 5}개 URL")
            
        except Exception as e:
            print(f"❌ 검색 실패: {str(e)}")
            state["error_info"] = f"사전 검색 실패: {str(e)}"
            state["processing_status"] = "사전 검색 실패"
            state["search_results"] = []
            state["relevant_urls"] = []
            
        return state
    
    def _calculate_relevance_score(self, result: Dict[str, Any], keyword: str) -> float:
        """
        검색 결과의 관련성 점수를 계산합니다.
        
        Args:
            result (Dict[str, Any]): Tavily 검색 결과 객체
            keyword (str): 검색에 사용된 키워드
            
        Returns:
            float: 계산된 관련성 점수 (높을수록 관련성이 높음)
            
        Note:
            유틸리티 함수를 사용하여 일관된 점수 계산
        """
        return calculate_relevance_score(result, keyword)
    
    async def pre_scrape(self, state: ShoppingAgentState) -> ShoppingAgentState:
        """
        3단계: Firecrawl API를 사용하여 선별된 URL의 상세 콘텐츠를 수집합니다.
        
        Args:
            state (ShoppingAgentState): 검색 결과와 URL 목록이 포함된 상태
            
        Returns:
            ShoppingAgentState: 스크래핑된 콘텐츠와 상품 정보가 추가된 상태
            
        Process:
            1. 관련성 점수 기반으로 최적의 URL 선택
            2. Firecrawl을 통한 구조화된 콘텐츠 추출
            3. 상품 정보 자동 추출 (제목, 가격, 설명 등)
            4. 콘텐츠 길이 제한 및 정제
            
        Key Output:
            - scraped_content: URL별 스크래핑된 마크다운 콘텐츠
            - product_data: 추출된 상품 정보 리스트
            
        Note:
            최종 답변의 구체성과 정확성을 결정하는 중요한 단계
        """
        print("\n=== [3/4] 사전 스크래핑 노드 시작 ===")
        relevant_urls = state["relevant_urls"]
        search_results = state.get("search_results", [])
        print(f"🔗 스크래핑 대상 URL: {len(relevant_urls)}개")
        scraped_content = {}
        product_data = []
        
        try:
            # 스크래핑할 URL 개수 제한
            max_urls_to_scrape = self.config.scraping.max_urls_to_scrape
            
            if not relevant_urls:
                print("⚠️ 스크래핑할 URL이 없습니다")
                state["scraped_content"] = {}
                state["product_data"] = []
                state["processing_status"] = "스크래핑할 URL 없음"
                return state
            
            # 최적의 URL 선택
            best_urls = self._select_best_urls_for_scraping(relevant_urls, search_results, max_urls_to_scrape)
            print(f"🎯 선택된 최적 URL: {len(best_urls)}개")
            
            # 선택된 URL 상세 표시
            if best_urls:
                print("📋 스크래핑 예정 URL:")
                for i, url in enumerate(best_urls, 1):
                    print(f"   {i}. {url}")
            else:
                print("⚠️ 스크래핑할 URL이 선택되지 않았습니다")
            
            if best_urls:
                # Firecrawl 직접 클라이언트 사용
                for url in best_urls:
                    try:
                        print(f"📄 스크래핑 시작: {url}")
                        
                        # Firecrawl로 스크래핑 (재시도 메커니즘 포함)
                        scrape_result = await self._scrape_url_with_retry(url)
                        
                        if scrape_result and scrape_result.success:
                            # content = scrape_result.get("data", {}).get("markdown", "")
                            content = scrape_result.markdown
                            
                            # 콘텐츠 길이 제한
                            content_limit = self.config.scraping.content_max_length
                            limited_content = content[:content_limit] if len(content) > content_limit else content
                            
                            scraped_content[url] = {
                                "title": self._extract_title(limited_content),
                                "content": limited_content,
                                "timestamp": datetime.now().isoformat(),
                                "content_length": len(content),
                                "content_truncated": len(content) > content_limit,
                                "original_data": content
                            }
                            
                            # 상품 데이터 추출
                            extracted_product = self._extract_product_info(limited_content, url)
                            if extracted_product:
                                product_data.append(extracted_product)
                        else:
                            # 스크래핑 실패
                            error_msg = scrape_result.error if scrape_result else "응답 없음"
                            scraped_content[url] = {
                                "title": "스크래핑 실패",
                                "content": f"오류: {error_msg}",
                                "timestamp": datetime.now().isoformat(),
                                "error": True
                            }
                            
                    except Exception as url_error:
                        # 개별 URL 스크래핑 실패
                        scraped_content[url] = {
                            "title": "스크래핑 실패",
                            "content": f"오류: {str(url_error)}",
                            "timestamp": datetime.now().isoformat(),
                            "error": True
                        }
            
            state["scraped_content"] = scraped_content
            state["product_data"] = product_data
            state["processing_status"] = f"사전 스크래핑 완료 ({len(scraped_content)}개 URL)"
            
            print(f"✅ 스크래핑 완료: {len(scraped_content)}개 URL, {len(product_data)}개 상품 정보 추출")
            if product_data:
                print(f"   - 상품 추출 예시: {product_data[0]['name'][:30]}...")
            
        except Exception as e:
            print(f"❌ 스크래핑 실패: {str(e)}")
            state["error_info"] = f"사전 스크래핑 실패: {str(e)}"
            state["processing_status"] = "사전 스크래핑 실패"
            state["scraped_content"] = {}
            state["product_data"] = []
            
        return state
    
    def _select_best_urls_for_scraping(self, relevant_urls: List[str], search_results: List[Dict], max_count: int) -> List[str]:
        """스크래핑을 위한 최적의 URL 선택"""
        if not relevant_urls:
            return []
        
        # URL별 점수 계산
        url_scores = {}
        
        # 검색 결과에서 점수 가져오기
        for result in search_results:
            url = result.get("url")
            if url and url in relevant_urls:
                url_scores[url] = result.get("relevance_score", 0)
        
        # 검색 결과에 없는 URL은 기본 점수 부여
        for url in relevant_urls:
            if url not in url_scores:
                url_scores[url] = 0.0
        
        # 쇼핑몰 도메인 우선순위 추가
        for url in url_scores:
            for domain in self.config.scraping.preferred_shopping_domains:
                if domain in url.lower():
                    url_scores[url] += 0.3  # 쇼핑몰 도메인 가산점
                    break
        
        # 점수 순으로 정렬하여 상위 URL 선택
        sorted_urls = sorted(url_scores.items(), key=lambda x: x[1], reverse=True)
        best_urls = [url for url, score in sorted_urls[:max_count]]
        
        return best_urls
    
    @retry_on_failure(max_retries=2, delay=1.0)
    async def _scrape_url_with_retry(self, url: str):
        """
        재시도 메커니즘이 포함된 URL 스크래핑
        
        Args:
            url (str): 스크래핑할 URL
            
        Returns:
            Firecrawl 응답 객체
            
        Note:
            502 에러 등 일시적 장애에 대응하여 최대 2회 재시도
        """
        import asyncio
        
        # Firecrawl은 동기 API이므로 비동기 래퍼 사용
        loop = asyncio.get_event_loop()
        
        def sync_scrape():
            return self.firecrawl_client.scrape_url(url, formats=["markdown"])
        
        return await loop.run_in_executor(None, sync_scrape)
    
    def _extract_title(self, content: str) -> str:
        """
        콘텐츠에서 제목을 추출합니다.
        
        Args:
            content (str): 분석할 마크다운 콘텐츠
            
        Returns:
            str: 추출된 제목
            
        Note:
            유틸리티 함수로 위임하여 일관된 제목 추출
        """
        return extract_title_from_content(content)
    
    def _extract_product_info(self, content: str, url: str) -> Optional[Dict[str, Any]]:
        """
        콘텐츠에서 상품 정보를 추출합니다.
        
        Args:
            content (str): 스크래핑된 콘텐츠
            url (str): 상품 페이지 URL
            
        Returns:
            Optional[Dict[str, Any]]: 추출된 상품 정보 또는 None
            
        Note:
            유틸리티 함수를 사용하여 표준화된 상품 정보 추출
        """
        return extract_product_info_from_content(content, url)
    
    async def call_agent(self, state: ShoppingAgentState) -> ShoppingAgentState:
        """
        4단계: 수집된 모든 정보를 종합하여 전문적인 쇼핑 추천 답변을 생성합니다.
        
        Args:
            state (ShoppingAgentState): 모든 이전 단계의 결과가 포함된 상태
            
        Returns:
            ShoppingAgentState: 최종 답변이 포함된 완료 상태
            
        Process:
            1. 분석 결과, 검색 결과, 상품 정보를 통합된 컨텍스트로 구성
            2. 전문 쇼핑 컨설턴트 페르소나로 구체적이고 실용적인 답변 생성
            3. 개인화된 추천, 가격대별 옵션, 구매 가이드 포함
            4. 장단점 분석과 대안 상품 제시로 신뢰성 확보
            
        Key Output:
            - final_answer: 완성된 쇼핑 추천 답변
            - enriched_context: 생성에 사용된 통합 컨텍스트
            
        Note:
            모든 이전 단계의 성과가 집약되는 최종 단계
        """
        print("\n=== [4/4] React Agent 노드 시작 ===")
        print("🤖 수집된 데이터를 기반으로 최종 답변 생성 중...")
        
        # 컨텍스트 구성
        context_parts = []
        
        # 질문 분석 결과
        if state.get("analyzed_query"):
            context_parts.append(f"질문 분석 결과: {json.dumps(state['analyzed_query'], ensure_ascii=False, indent=2)}")
        
        # 검색 결과
        if state.get("search_results"):
            search_summary = []
            for result in state["search_results"][:10]:  # 상위 10개
                search_summary.append(f"- {result['title']}: {result['content'][:200]}...")
            context_parts.append(f"검색 결과:\n" + "\n".join(search_summary))
        
        # 상품 데이터
        if state.get("product_data"):
            product_summary = []
            for product in state["product_data"][:5]:  # 상위 5개
                product_summary.append(f"- {product['name']}: {product['price']} ({product['url']})")
            context_parts.append(f"수집된 상품 정보:\n" + "\n".join(product_summary))
        
        enriched_context = "\n\n".join(context_parts)
        state["enriched_context"] = enriched_context
        
        # React Agent 프롬프트 구성
        system_prompt = """**당신은 전문 쇼핑 컨설턴트입니다.**

        **역할**: 사용자에게 최고의 쇼핑 경험을 제공하는 것이 목표입니다. 단순한 상품 나열이 아닌, 개인화된 맞춤 추천을 통해 사용자가 만족할 수 있는 완벽한 답변을 제공하세요.

        **🎯 답변 구성 원칙**:

        **1. 개인화된 인사 및 요구사항 확인**
        - 사용자의 질문을 정확히 이해했음을 보여주세요
        - 분석된 요구사항을 재확인하며 공감대 형성

        **2. 핵심 추천 상품**
        각 상품마다 다음 정보를 체계적으로 제공:
        - **상품명**: 명확하고 구체적인 제품명
        - **핵심 특징**: 왜 이 상품을 추천하는지 명확한 이유
        - **가격 정보**: 구체적인 상품 가격
        - **장점**: 사용자 요구사항과 연결된 장점
        - **주의사항**: 솔직한 단점이나 고려사항 (신뢰성 향상)
        - **구매처**: 구체적인 온라인몰이나 구매 방법

        **3. 가격대별 세분화 추천**
        - **경제적 선택**: 가성비 중심 옵션
        - **균형 선택**: 가격과 품질의 균형
        - **프리미엄 선택**: 최고 품질/성능 중심

        **4. 실용적 구매 가이드**
        - **구매 시 체크포인트**: 사이즈, 색상, 배송, A/S 등
        - **계절성/시기 고려사항**: 언제 사는 것이 유리한지
        - **대안 상품**: 재고 부족이나 예산 초과 시 대체재

        **5. 전문가 팁 & 개인화 조언**
        - 해당 카테고리의 전문적 인사이트
        - 사용자 상황에 맞는 맞춤 조언
        - 향후 구매를 위한 트렌드 정보

        **🎨 답변 스타일 가이드**:
        - **친근하고 전문적**: 딱딱하지 않으면서도 신뢰할 수 있는 톤
        - **구체적이고 실용적**: 모호한 표현보다는 명확한 정보
        - **균형잡힌 시각**: 장점만이 아닌 솔직한 단점도 포함
        - **행동 유도**: 사용자가 다음에 무엇을 해야 할지 명확히 제시

        **⚠️ 주의사항**:
        - 수집된 정보가 부족한 경우, 솔직하게 한계를 인정하세요
        - 과장된 표현보다는 객관적 정보를 우선하세요
        - 가격은 변동 가능함을 명시하세요
        - 개인 취향과 상황에 따라 다를 수 있음을 안내하세요

        **📊 수집된 컨텍스트 정보**:
        {context}

        위 정보를 바탕으로 사용자에게 최고의 쇼핑 경험을 선사하는 완벽한 답변을 작성해주세요.
        """
        
        try:
            messages = [
                SystemMessage(content=system_prompt.format(context=enriched_context)),
                HumanMessage(content=state["user_query"])
            ]
            
            response = await self.llm.ainvoke(messages)
            state["final_answer"] = response.content
            state["processing_status"] = "처리 완료"
            
            print(f"✅ 최종 답변 생성 완료 ({len(response.content)}자)")
            print(f"   - 답변 미리보기: {response.content[:100]}...")
            
            # 메시지 히스토리에 추가
            state["messages"].extend([
                HumanMessage(content=state["user_query"]),
                response
            ])
            
        except Exception as e:
            print(f"❌ React Agent 실행 실패: {str(e)}")
            state["error_info"] = f"React Agent 실행 실패: {str(e)}"
            state["processing_status"] = "처리 실패"
            state["final_answer"] = "죄송합니다. 답변 생성 중 오류가 발생했습니다."
            
        print("\n=== 🎉 Enhanced Shopping Agent 처리 완료 ===\n")
        return state


# 도구 함수들
@tool
def get_current_time() -> str:
    """현재 시간을 반환합니다."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


async def build_enhanced_agent(config_name: str = "credit_saving") -> CompiledStateGraph:
    """Enhanced Shopping Agent 빌드"""
    config = get_config(config_name)
    agent = EnhancedShoppingAgent(config)
    return agent.create_workflow()


async def run_agent_test():
    """에이전트 테스트 실행"""
    agent = await build_enhanced_agent()
    
    # 테스트 쿼리
    test_query = "겨울용 따뜻한 패딩 점퍼 추천해줘. 10만원 이하로 검은색이면 좋겠어."
    
    initial_state = {
        "user_query": test_query,
        "messages": [],
        "processing_status": "시작"
    }
    
    print("=== Enhanced Shopping Agent 테스트 시작 ===")
    print(f"질문: {test_query}")
    print()
    
    # 에이전트 실행
    async for chunk in agent.astream(
        initial_state,
        stream_mode="values"
    ):
        messages = chunk["messages"]

        for msg in messages:
            msg.pretty_print()

    # result = await agent.ainvoke(initial_state)
    
    # # 결과 출력
    # print(f"처리 상태: {result.get('processing_status')}")
    # print(f"최종 답변: {result.get('final_answer')}")
    
    # if result.get('error_info'):
    #     print(f"오류 정보: {result['error_info']}")
    
    # # 상세 정보 출력
    # print("\n=== 처리 과정 상세 ===")
    # if result.get('analyzed_query'):
    #     print(f"질문 분석: {json.dumps(result['analyzed_query'], ensure_ascii=False, indent=2)}")
    
    # if result.get('search_results'):
    #     print(f"검색 결과 수: {len(result['search_results'])}")
    
    # if result.get('product_data'):
    #     print(f"수집된 상품 수: {len(result['product_data'])}")


if __name__ == "__main__":
    asyncio.run(run_agent_test())