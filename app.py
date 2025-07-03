# app.py
"""
AI 쇼핑 어시스턴트 웹 인터페이스

이 모듈은 Streamlit을 사용하여 AI 쇼핑 어시스턴트의 웹 인터페이스를 구현합니다.
LangChain/LangGraph 에이전트와 FireCrawl 도구를 활용하여 사용자 질의에 응답하며,
LangGraph의 checkpoint_ns를 활용한 정확한 도구 추적 및 에러 처리 시스템을 제공합니다.

주요 기능:
- 실시간 스트리밍 채팅 인터페이스
- 도구 실행 상태 추적 및 시각화
- 그룹 단위 에러 처리 (checkpoint_ns 기반)
- 도구 실행 시간 및 상세 정보 표시
"""

import streamlit as st
import asyncio
import json
import traceback
import uuid

from datetime import datetime
from dotenv import load_dotenv
from langchain_core.messages import ToolMessage, AIMessage
from typing import Dict, Any, List, Optional, Set, Tuple
# from src.agent.enhanced_shopping_agent import build_enhanced_agent as build_agent
from src.utils.local_prompt_manager import LocalPromptManager
from src.agent.shopping_react_agent import build_agent

load_dotenv()


# =============================================================================
# Streamlit 앱 설정 및 초기화
# =============================================================================

st.set_page_config(
    page_title="쇼핑 어시스턴트", 
    page_icon="🛍️", 
    layout="wide"
)

# 세션 상태 초기화 - Streamlit의 상태 관리 시스템
if 'agent' not in st.session_state:
    st.session_state.agent = None  # LangChain 에이전트 인스턴스
if 'messages' not in st.session_state:
    st.session_state.messages = []  # 채팅 메시지 히스토리
if 'history' not in st.session_state:
    st.session_state.history = []  # LangChain 대화 히스토리

# 프롬프트 관리 세션 상태 초기화
if 'prompt_manager' not in st.session_state:
    st.session_state.prompt_manager = LocalPromptManager()
if 'active_prompt_name' not in st.session_state:
    st.session_state.active_prompt_name = 'default'
if 'prompt_edit_mode' not in st.session_state:
    st.session_state.prompt_edit_mode = False
if 'show_prompt_manager' not in st.session_state:
    st.session_state.show_prompt_manager = False
if 'current_editing_prompt' not in st.session_state:
    st.session_state.current_editing_prompt = None
if 'show_new_prompt_form' not in st.session_state:
    st.session_state.show_new_prompt_form = False


# =============================================================================
# 도구 실행 추적 클래스
# =============================================================================

class ToolExecutionTracker:
    """
    LangGraph 도구 실행을 추적하고 관리하는 클래스
    
    이 클래스는 LangGraph의 checkpoint_ns 시스템을 활용하여 도구 그룹을 정확히 추적하고,
    도구 실행 실패 시 그룹 내 모든 도구에 일관된 에러 처리를 제공합니다.
    
    주요 구성 요소:
    - tool_calls: 개별 도구 실행 정보 저장 (run_id 기반)
    - tools_groups: 도구 그룹 추적 (checkpoint_ns 기반)
    - completed_tools: 완료된 도구 추적
    """
    
    def __init__(self):
        """추적기 초기화"""
        # 개별 도구 실행 정보를 run_id로 추적
        # 각 도구의 시작시간, 종료시간, 입력, 출력, 에러 상태 등을 저장
        self.tool_calls: Dict[str, Dict[str, Any]] = {}
        
        # LangGraph의 checkpoint_ns로 도구 그룹을 추적
        # 형태: {"tools:uuid": {run_id1, run_id2, run_id3}}
        # 같은 요청에서 실행되는 여러 도구들이 동일한 namespace를 공유
        self.tools_groups: Dict[str, Set[str]] = {}
        
        # 완료된 도구들의 run_id 집합
        # 중복 처리 방지 및 상태 추적용
        self.completed_tools: Set[str] = set()
    
    def extract_tools_namespace(self, event: Dict[str, Any]) -> Optional[str]:
        """
        LangGraph 이벤트에서 tools namespace를 추출
        
        LangGraph는 관련된 도구들을 'tools:uuid' 형태의 namespace로 그룹화합니다.
        이를 통해 동일한 요청에서 실행되는 여러 도구들을 하나의 그룹으로 관리할 수 있습니다.
        
        Args:
            event: LangGraph 이벤트 객체
            
        Returns:
            tools namespace 문자열 또는 None
        """
        metadata = event.get('metadata', {})
        checkpoint_ns = metadata.get('langgraph_checkpoint_ns', '')
        
        # 'tools:'로 시작하는 namespace만 유효한 것으로 간주
        if checkpoint_ns and checkpoint_ns.startswith('tools:'):
            return checkpoint_ns
        return None
    
    def start_tool_execution(
        self, 
        run_id: str, 
        tool_name: str, 
        input_data: Any, 
        event: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        도구 실행 시작을 추적 및 등록
        
        새로운 도구 실행이 시작될 때 호출되며, 도구의 기본 정보를 설정하고
        해당 도구를 적절한 그룹에 배정합니다.
        
        Args:
            run_id: 도구 실행의 고유 식별자
            tool_name: 실행되는 도구의 이름
            input_data: 도구에 전달되는 입력 데이터
            event: LangGraph 이벤트 객체
            
        Returns:
            생성된 도구 호출 데이터 딕셔너리
        """
        tools_namespace = self.extract_tools_namespace(event)
        
        # 도구 실행 정보 구조체 생성
        call_data = {
            "run_id": run_id,
            "name": tool_name,
            "input": input_data,
            "output": None,
            "finished": False,
            "error": None,
            "start_time": asyncio.get_event_loop().time(),
            "tools_namespace": tools_namespace,
            "end_time": None
        }
        
        # run_id로 도구 정보 저장
        self.tool_calls[run_id] = call_data
        
        # tools 그룹에 도구 추가
        # 같은 namespace의 도구들은 함께 관리됨
        if tools_namespace:
            if tools_namespace not in self.tools_groups:
                self.tools_groups[tools_namespace] = set()
            self.tools_groups[tools_namespace].add(run_id)
        
        return call_data
    
    def finish_tool_execution(
        self, 
        run_id: str, 
        output: Any, 
    ) -> Optional[Dict[str, Any]]:
        """
        도구 실행 완료 처리
        
        도구 실행이 완료되었을 때 호출되며, 결과를 저장하고 에러 상태를 판단합니다.
        None이나 빈 결과의 경우 임시 에러 메시지를 생성하여 나중에 더 구체적인
        에러로 업데이트될 수 있도록 합니다.
        
        Args:
            run_id: 완료된 도구의 실행 ID
            output: 도구의 실행 결과
            event: LangGraph 이벤트 객체
            
        Returns:
            업데이트된 도구 호출 데이터 또는 None
        """
        if run_id not in self.tool_calls:
            return None
            
        call_data = self.tool_calls[run_id]
        call_data["output"] = output
        call_data["finished"] = True
        call_data["end_time"] = asyncio.get_event_loop().time()
        
        # 완료된 도구로 마킹
        self.completed_tools.add(run_id)
        
        # 결과 상태 분석 및 에러 처리
        self._analyze_output_and_set_error(call_data, output)
        
        return call_data
    
    def _analyze_output_and_set_error(self, call_data: Dict[str, Any], output: Any) -> None:
        """
        도구 출력을 분석하여 에러 상태 설정
        
        도구의 출력 결과를 분석하여 성공/실패를 판단하고, 실패인 경우
        적절한 에러 메시지를 설정합니다.
        
        Args:
            call_data: 도구 호출 데이터
            output: 도구 출력
        """
        if output is None:
            # None 결과는 실행 실패로 간주
            call_data["error"] = "도구가 결과를 반환하지 않았습니다."
            call_data["output"] = f"ToolException: {call_data['error']}"
            call_data["_is_placeholder_error"] = True
            
        elif isinstance(output, str):
            if not output.strip():
                # 빈 문자열도 실패로 간주
                call_data["error"] = "도구가 빈 결과를 반환했습니다."
                call_data["output"] = f"ToolException: {call_data['error']}"
                call_data["_is_placeholder_error"] = True
            elif self._is_error_string(output):
                # 에러 키워드가 포함된 문자열
                call_data["error"] = output
                call_data["_is_placeholder_error"] = False
                
        elif isinstance(output, (dict, list)) and len(output) == 0:
            # 빈 딕셔너리나 리스트도 실패로 간주
            call_data["error"] = "도구가 빈 결과를 반환했습니다."
            call_data["output"] = f"ToolException: {call_data['error']}"
            call_data["_is_placeholder_error"] = True
    
    def _is_error_string(self, output: str) -> bool:
        """문자열이 에러 메시지인지 판단"""
        error_keywords = ['error', 'exception', 'failed', 'timeout', 'fail']
        return any(keyword in output.lower() for keyword in error_keywords)
    
    def handle_group_error(self, error_event: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        도구 그룹 전체에 대한 에러 처리
        
        LangGraph에서 on_chain_stream 이벤트를 통해 전달되는 그룹 레벨 에러를
        처리합니다. checkpoint_ns를 사용하여 해당 그룹의 모든 도구에 동일한
        에러 메시지를 적용합니다.
        
        Args:
            error_event: 에러 이벤트 객체
            
        Returns:
            업데이트된 도구 호출 데이터 리스트
        """
        tools_namespace = self.extract_tools_namespace(error_event)
        updated_calls = []
        
        if not tools_namespace or tools_namespace not in self.tools_groups:
            return updated_calls
        
        # 에러 메시지 추출
        error_message = self._extract_error_message(error_event)
        
        # 해당 그룹의 모든 도구에 에러 적용
        for run_id in self.tools_groups[tools_namespace].copy():
            if run_id in self.tool_calls:
                call_data = self.tool_calls[run_id]
                
                # 에러 정보 업데이트
                call_data["output"] = error_message
                call_data["error"] = error_message
                call_data["finished"] = True
                
                if not call_data.get("end_time"):
                    call_data["end_time"] = asyncio.get_event_loop().time()
                
                # self.completed_tools.add(run_id)
                updated_calls.append(call_data)
        
        return updated_calls
    
    def _extract_error_message(self, error_event: Dict[str, Any]) -> str:
        """
        에러 이벤트에서 구체적인 에러 메시지 추출
        
        Args:
            error_event: 에러 이벤트 객체
            
        Returns:
            추출된 에러 메시지
        """
        # 기본 에러 메시지
        default_message = "도구 실행 중 오류가 발생했습니다."
        
        chunk = error_event.get("data", {}).get("chunk", {})
        if not isinstance(chunk, dict):
            return default_message
        
        messages = chunk.get("messages", [])
        for msg in messages:
            if hasattr(msg, 'content') and msg.content:
                return str(msg.content)
        
        return default_message
    
    def handle_unfinished_tools(self, timeout_seconds: float = 5.0) -> List[Dict[str, Any]]:
        """
        미완료 도구들을 타임아웃 처리
        
        스트림이 종료되었지만 아직 완료되지 않은 도구들을 찾아 타임아웃으로
        처리합니다. 이는 예상치 못한 상황에서 UI가 무한 대기하는 것을 방지합니다.
        
        Args:
            timeout_seconds: 타임아웃 기준 시간 (초)
            
        Returns:
            타임아웃 처리된 도구 호출 데이터 리스트
        """
        unfinished = []
        current_time = asyncio.get_event_loop().time()
        
        for call_data in self.tool_calls.values():
            if call_data["finished"] or call_data["run_id"] in self.completed_tools:
                continue
                
            execution_time = current_time - call_data["start_time"]
            
            # 타임아웃 메시지 설정
            if execution_time > timeout_seconds:
                call_data["output"] = f"ToolException: 도구 실행 타임아웃 ({execution_time:.1f}초)"
                call_data["error"] = f"도구 실행이 {timeout_seconds}초를 초과하여 타임아웃되었습니다."
            else:
                call_data["output"] = "ToolException: 도구 실행이 정상적으로 완료되지 않았습니다."
                call_data["error"] = "도구 실행이 정상적으로 완료되지 않았습니다."
            
            call_data["finished"] = True
            call_data["end_time"] = current_time
            self.completed_tools.add(call_data["run_id"])
            unfinished.append(call_data)
        
        return unfinished
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """
        실행 요약 정보 반환 (디버깅 및 모니터링용)
        
        Returns:
            실행 통계 및 상태 정보
        """
        return {
            "total_tools": len(self.tool_calls),
            "completed_tools": len(self.completed_tools),
            "tools_groups": {ns: list(run_ids) for ns, run_ids in self.tools_groups.items()},
            "execution_times": {
                run_id: call_data.get("end_time", 0) - call_data["start_time"]
                for run_id, call_data in self.tool_calls.items()
                if call_data.get("end_time")
            }
        }


# =============================================================================
# 에이전트 관리 함수
# =============================================================================

async def initialize_agent():
    """
    LangChain 에이전트 초기화
    
    에이전트가 아직 초기화되지 않은 경우에만 build_agent()를 호출하여
    새로운 에이전트 인스턴스를 생성하고 세션 상태에 저장합니다.
    
    중복 초기화 방지:
    - 세션 상태를 확인하여 이미 초기화된 경우 재사용
    - MCP 서버의 중복 실행을 방지하여 리소스 절약
    """
    if st.session_state.agent is None:
        with st.spinner("🔧 AI 쇼핑 어시스턴트를 준비하는 중입니다..."):
            try:
                # 임시 프롬프트가 있는지 확인
                # if hasattr(st.session_state, 'temp_prompts'):
                #     # 임시 프롬프트를 사용해서 에이전트 빌드
                #     from src.agent.enhanced_shopping_agent import EnhancedShoppingAgent
                #     from src.config.agent_config import get_config
                    
                #     config = get_config("credit_saving")
                #     temp_agent = EnhancedShoppingAgent(config, st.session_state.active_prompt_name)
                    
                #     # 임시 프롬프트로 오버라이드
                #     temp_agent.analysis_prompt_template = st.session_state.temp_prompts['analysis']
                #     temp_agent.response_prompt_template = st.session_state.temp_prompts['response']
                    
                #     agent = temp_agent.create_workflow()
                # else:
                #     # 선택된 개별 프롬프트들을 사용해서 에이전트 빌드
                #     from src.agent.enhanced_shopping_agent import EnhancedShoppingAgent
                #     from src.config.agent_config import get_config
                    
                #     config = get_config("credit_saving")
                #     agent_instance = EnhancedShoppingAgent(config, st.session_state.active_prompt_name)
                    
                #     # 선택된 개별 프롬프트들로 오버라이드
                #     selected_analysis_data = st.session_state.prompt_manager.get_prompt_by_type(
                #         st.session_state.selected_analysis_prompt, "query_analysis"
                #     )
                #     selected_response_data = st.session_state.prompt_manager.get_prompt_by_type(
                #         st.session_state.selected_response_prompt, "model_response"
                #     )
                    
                #     if selected_analysis_data:
                #         agent_instance.analysis_prompt_template = selected_analysis_data.get('content', '')
                #     if selected_response_data:
                #         agent_instance.response_prompt_template = selected_response_data.get('content', '')
                    
                #     agent = agent_instance.create_workflow()
            
                agent = await build_agent()
                
                # 에이전트가 제대로 생성되었는지 확인
                if agent is not None:
                    st.session_state.agent = agent
                    st.success("✅ 에이전트가 성공적으로 초기화되었습니다.")
                    # 디버그: 에이전트 타입 확인
                    st.info(f"🔍 에이전트 타입: {type(agent).__name__}")
                    return True
                else:
                    st.error("❌ 에이전트 생성 실패: build_agent()가 None을 반환했습니다.")
                    return False
                    
            except Exception as e:
                st.error(f"❌ 에이전트 초기화 실패: {str(e)}")
                st.info("페이지를 새로고침하거나 잠시 후 다시 시도해주세요.")
                # 디버그: 상세 에러 정보
                with st.expander("🐛 에러 상세 정보"):
                    st.code(str(e))
                return False
    else:
        # 이미 초기화된 경우
        st.info("✅ 에이전트가 이미 초기화되어 있습니다.")
        return True


def ensure_agent_ready() -> bool:
    """
    에이전트가 준비되었는지 확인하고 필요시 초기화
    
    이 함수는 사용자 입력 처리 전에 호출되어 에이전트가 
    사용 가능한 상태인지 확인합니다.
    
    Returns:
        에이전트 준비 상태 (True: 준비됨, False: 실패)
    """
    if st.session_state.agent is None:
        # 동기 함수에서 비동기 함수 호출
        return asyncio.run(initialize_agent())
    return True


# =============================================================================
# 응답 스트리밍 함수
# =============================================================================

async def get_response(agent, user_input: str, history: List[Tuple[str, str]]):
    """
    에이전트로부터 응답을 스트리밍으로 받아 처리
    
    이 함수는 LangGraph의 astream_events를 사용하여 실시간으로 이벤트를 처리하고,
    도구 실행 상태를 추적하며, 에러를 적절히 처리합니다.
    
    Args:
        agent: LangChain 에이전트 인스턴스
        user_input: 사용자 입력 메시지
        history: 이전 대화 히스토리
        
    Yields:
        다양한 타입의 이벤트 딕셔너리 (content, tool_start, tool_end, error 등)
    """
    
    # 시스템 프롬프트 정의
#     system_prompt = """당신은 사용자의 복합적인 쇼핑 요구사항을 지능적으로 분석하고, 단계적 검색 전략을 통해 즉시 구매 가능한 최적 상품을 찾아 추천하는 전문 쇼핑 어시스턴트입니다.

# # 💡 주요 기능
# - **지능형 요구사항 분석**: 사용자의 요청을 `핵심 키워드`, `필터링 조건`, `부가 조건`으로 분해하여 검색 전략 수립
# - **단계적 스마트 검색**: `기본 검색` → `유사어 확장` → `결과 필터링` → `구매 가능성 검증`의 4단계 프로세스 수행
# - **중복 상품 제거 및 다양성 확보**: 동일/유사 상품을 제거하고, `브랜드`, `가격대`, `스타일`의 다양성을 보장하여 최종 추천
# - **검색 실패시 지능형 대응**: 검색 결과가 부족할 경우, `키워드 변형`, `카테고리 확장`, `조건 완화` 등 단계적으로 검색 범위 확장

# # 📝 응답 형식
# - **검색 과정 투명화**: 사용자의 요청 분석 결과, 검색 단계, 필터링, 중복 제거 과정을 명확히 보고
# - **조건별 상품 분류 추천**: `완벽 조건 만족`, `주요 조건 만족`, `대안 추천` 등 조건 충족 수준에 따라 상품을 분류하여 제안
# - **다양성 보장된 최종 추천**: 각 상품의 `브랜드`, `상품명`, `가격`, `고유 특징`을 명시하고, 중복이 제거된 다양한 옵션을 제공

# # 🔧 검색 최적화 규칙
# - **플랫폼별 검색 전략**: `네이버쇼핑`(가격 비교), `SSG몰`(프리미엄), `무신사`(패션/트렌드) 등 플랫폼 특성에 맞는 검색 수행
# - **시간 효율성 최적화**: 5분 내 결과 도출을 목표로, 빠른 판단과 우선순위 설정에 기반한 효율적 검색 진행
# """
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    system_prompt = """당신은 AI 기반 전문 쇼핑 컨설턴트로서, 반드시 실시간 검색을 통한 검증된 정보를 바탕으로 고객의 구매 여정을 지원합니다. 메모리나 추측에 의존한 답변은 절대 금지되며, 모든 추천은 도구를 통해 수집된 최신 데이터에 기반해야 합니다.
현재 시간 정보: {CURRENT_DATETIME}
🎯 핵심 미션 (Core Mission)
"실시간 검색된 검증 정보만을 사용하여 고객이 후회하지 않는 구매 결정을 내릴 수 있도록 맞춤형 쇼핑 솔루션을 제공한다."

🚫 절대 금지 사항 (Absolute Prohibitions)
❌ 메모리 기반 답변 완전 금지

기존 학습 데이터나 메모리 정보로 제품 추천 절대 금지
"일반적으로 알려진", "보통", "대체로" 등의 표현 사용 금지
브랜드명, 모델명, 가격 정보를 메모리로 제공 금지
검색 없이 제품 비교나 순위 제공 금지

❌ 검색 전 임시 답변 금지

정보 수집 전 어떠한 제품 언급도 금지
"우선 이런 제품들이 있고, 나중에 자세히 찾아보겠습니다" 방식 금지
불완전한 정보 기반 중간 보고 금지


✅ 필수 준수 사항 (Mandatory Requirements)
🔍 필수 도구 사용 규칙
모든 쇼핑 관련 질문에 대해 다음 단계를 반드시 순서대로 실행:

요구사항 분석 완료 후 즉시 검색 실행
충분한 정보 수집까지 답변 생성 금지
검증된 정보만으로 최종 답변 구성

📝 강제적 검색 프로토콜
사용자가 다음과 같은 요청을 할 때 반드시 도구 사용:

제품 추천 요청 → firecrawl_search + firecrawl_crawl 필수
가격 문의 → firecrawl_search + firecrawl_scrape 필수
제품 비교 → 각 제품별 firecrawl_scrape + firecrawl_extract 필수
리뷰 정보 → firecrawl_crawl + web_search 필수
할인/프로모션 정보 → firecrawl_search + web_search 필수
브랜드/모델 문의 → firecrawl_scrape + firecrawl_extract 필수

⚠️ 검색 실패 시 대응

검색 결과가 불충분한 경우: "추가 검색이 필요합니다" 명시 후 재검색
검색 도구 오류 시: "현재 정확한 정보 수집이 어려운 상황입니다" 안내
절대 메모리 정보로 대체하지 않음


🔧 개선된 운영 프로세스 (Enhanced Operating Process)
Phase 1: 요구사항 분석 (Requirements Analysis)
목표: 검색에 필요한 모든 정보 수집
1. 사용자 질문 분해
   - 제품 카테고리 식별
   - 예산 범위 확인
   - 주요 기능 요구사항 파악
   - 사용 목적/환경 확인

2. 검색 계획 수립
   - 필요한 검색 도구 결정
   - 검색 키워드 및 범위 설정
   - 정보 검증 방법 계획

3. 검색 실행 선언
   "정확한 정보를 위해 실시간 검색을 시작하겠습니다."
Phase 2: 강제적 정보 수집 (Mandatory Information Collection)
절대 규칙: 이 단계에서는 어떠한 추천이나 제품 언급도 금지
🔍 필수 검색 순서:
1. firecrawl_search: 광범위한 제품 및 가격 검색
2. firecrawl_crawl: 제품 관련 정보 수집
3. firecrawl_scrape: 구체적인 제품 상세 정보 수집
4. firecrawl_extract: 특정 정보 추출
5. web_search: 최신 리뷰, 트렌드, 뉴스 정보 (보완적 사용)

⚠️ 검색 중 절대 금지:
- "일반적으로 이런 제품들이 좋습니다" 
- "제가 알기로는..."
- "보통 추천되는 제품은..."
Phase 3: 정보 검증 및 분석 (Verification & Analysis)
목표: 수집된 정보의 신뢰성 검증 및 종합 분석
📊 검증 프로세스:
1. 다중 소스 교차 검증
   - 최소 3개 이상 독립 소스 확인
   - 가격 정보 일치성 검토
   - 제품 사양 정확성 확인

2. 정보 품질 평가
   - 최신성 확인 (발행일, 업데이트일)
   - 신뢰할 수 있는 소스인지 판단
   - 편향성 또는 광고성 내용 식별

3. 종합 분석 실시
   - 가성비 분석
   - 사용자 요구사항 적합성 평가
   - 장단점 객관적 비교
Phase 4: 검증된 솔루션 제시 (Verified Solution Delivery)
규칙: 검증 완료된 정보만 사용하여 답변 구성

📋 필수 답변 템플릿
markdown🔍 **정보 수집 현황**
- 실시간 검색 완료: [수행 시간]
- 검증된 소스: [주요 소스 수량]

🎯 **검증된 추천 결과**
[검색 결과 기반 추천 내용]

📊 **상세 비교 분석**
[실제 검색된 제품들의 비교 정보]

💰 **실제 가격 정보**
[검색으로 확인된 최신 가격]

⭐ **실제 사용자 리뷰 종합**
[웹 검색으로 수집된 리뷰 정보]

🛒 **구매 가이드**
[검색 기반 구매 조언]

---
📋 **참고한 정보 출처**
🔗 **주요 참고 사이트**
- [실제 검색한 사이트들]

💡 **정보 주의사항**
- 모든 정보는 [검색 수행 시간] 기준입니다
- 가격 및 재고는 실시간 변동 가능합니다
- 최종 구매 전 해당 쇼핑몰에서 재확인 권장합니다

🛡️ 내부 검색 강제 실행 체크리스트
(사용자에게 노출하지 않고 내부적으로만 확인)
✅ 답변 전 필수 내부 확인사항:

 firecrawl_search/주요 쇼핑채널 교차 검색 완료
 가격/상품명/브랜드/상세 모두 최신 실시간 정보 반영
 대표 인기상품/후기/실사용 팁 종합
 메모리 기반 답변 일체 없음
 관련 제품에 대해 firecrawl_search 실행했는가?
 제품 관련 정보를 firecrawl_crawl로 수집했는가?
 구체적인 제품 정보를 firecrawl_scrape로 확인했는가?
 필요한 특정 정보를 firecrawl_extract로 추출했는가?
 가격 정보가 실시간 검색 결과인가?
 모든 제품명/모델명이 검색으로 확인된 것인가?
 추측성 내용이 완전히 제거되었는가?

❌ 답변 거부 조건:

검색 도구를 사용하지 않고 답변 시도
"일반적으로", "보통" 등의 표현 사용
메모리 정보 기반 제품 추천
검색 결과 없이 가격/사양 정보 제공

중요: 위 체크리스트는 사용자 답변에 포함하지 않고, 시스템 내부에서만 확인하여 품질을 보장하는 용도로 사용

⚠️ 예외 처리 및 오류 방지
🔍 검색 실패 시 대응
"죄송합니다. 현재 다음과 같은 이유로 정확한 정보 수집이 어려운 상황입니다:
- [구체적인 검색 실패 이유]
- [대안적 정보 수집 방법]
- [사용자가 직접 확인할 수 있는 방법]

메모리나 추측으로 답변드리는 것보다는, 정확한 정보를 위해 다음과 같이 안내드립니다:
[구체적인 대안 방안]"
🚫 절대 사용 금지 표현

"일반적으로 추천되는..."
"제가 알기로는..."
"보통 이런 제품들이..."
"대체로 좋은 평가를..."
"아마도..." / "추정하건대..."

✅ 권장 표현

"실시간 검색 결과에 따르면..."
"방금 확인한 정보로는..."
"현재 시점 검색 결과..."
"최신 검색 정보 기준으로..."


🎯 성능 지표 및 품질 관리
📊 필수 달성 지표

도구 사용률: 100% (쇼핑 관련 질문 시)
메모리 기반 답변: 0%
검색 전 추천 제공: 0건
정보 출처 명시율: 100%

🔄 지속적 개선

검색 실패 케이스 분석
정보 수집 효율성 개선
사용자 만족도 기반 프로세스 최적화


🔥 핵심 원칙 재강조:

검색 없는 답변은 절대 금지
모든 제품 정보는 실시간 검색 결과 사용
메모리 기반 추천은 100% 차단
정보 수집 완료 전까지 추천 지연
검증된 정보만으로 최종 답변 구성
"""

    
    # 도구 실행 추적기 초기화
    tracker = ToolExecutionTracker()
    try:
        print(f"system_prompt : {system_prompt.format(CURRENT_DATETIME=current_datetime)}")
    except Exception as e:
        print(f"system_prompt error: {e}")
    
    # Enhanced Agent 상태 구성
    initial_state = {
        "user_query": user_input,
        # "messages": [("system", system_prompt)] + history + [("user", user_input)],
        "messages": [("system", system_prompt.format(CURRENT_DATETIME=current_datetime))] + history + [("user", user_input)],
        "processing_status": "시작"
    }

    try:
        # LangGraph 이벤트 스트림 처리
        # print(f"initial_state: {initial_state}")
        async for event in agent.astream_events(initial_state, version="v1"):
            event_type = event["event"]
            # print(f"event: {event}")
            
            if event_type == "on_chat_model_stream":
                # LLM 응답 텍스트 스트리밍
                content = event["data"]["chunk"].content
                if content:
                    yield {"type": "content", "data": content}
                    
            elif event_type == "on_tool_start":
                # 도구 실행 시작
                run_id = str(event['run_id'])
                tool_name = event['name']
                tool_input = event['data'].get('input')
                
                call_data = tracker.start_tool_execution(run_id, tool_name, tool_input, event)
                
                yield {
                    "type": "tool_start",
                    "run_id": run_id,
                    "name": tool_name,
                    "input": tool_input,
                    "call_data": call_data
                }
                
            elif event_type == "on_tool_end":
                print(f"event tool end: {event}")
                # 도구 실행 완료
                run_id = str(event['run_id'])
                tool_output = event['data'].get('output')
                
                updated_call_data = tracker.finish_tool_execution(run_id, tool_output)
                
                if updated_call_data:
                    yield {
                        "type": "tool_end",
                        "run_id": run_id,
                        "name": updated_call_data['name'],
                        "output": updated_call_data['output'],
                        "call_data": updated_call_data
                    }
                    
            elif event_type == "on_chain_stream" and event["metadata"].get("langgraph_node") == "tools":
                print(f"event on_chain_stream tools: {event}")

                
                chunk = event.get("data", {}).get("chunk", {})
                messages = chunk.get("messages", [])

                for msg in messages:
                    if isinstance(msg, ToolMessage) and hasattr(msg, 'status') and msg.status == 'error':
                        updated_cells = tracker.handle_group_error(event)

                        for call_data in updated_cells:
                            yield { 
                                "type": "tool_error",
                                "run_id": call_data["run_id"],
                                "tool_name": call_data["name"],
                                "error_message": call_data["error"],
                                "call_data": call_data,
                                "tools_namespace": call_data.get("tools_namespace")                                
                            }
                
                
            elif event_type == "on_chat_model_start":
                print(f"on_chat_model_start: {event}")
        
                                
    except Exception as e:
        # 예상치 못한 에러 처리
        error_message = f"응답 스트림 처리 중 오류가 발생했습니다: {str(e)}"
        yield {
            "type": "stream_error",
            "error": error_message,
            "traceback": traceback.format_exc()
        }


# =============================================================================
# UI 렌더링 함수
# =============================================================================

def render_tool_call(call_data: Dict[str, Any]) -> None:
    """
    도구 호출 정보를 Streamlit UI로 렌더링
    
    도구의 이름, 입력, 출력, 실행 시간, 에러 정보 등을 사용자가 이해하기 쉽게
    표시합니다. 에러가 있는 경우 시각적으로 구분하여 표시합니다.
    
    Args:
        call_data: 렌더링할 도구 호출 데이터
    """
    tool_name = call_data['name']
    tool_input = call_data.get('input', {})
    output = call_data.get('output')
    error = call_data.get('error')
    tools_namespace = call_data.get('tools_namespace', 'Unknown')
    is_error = error is not None or (isinstance(output, str) and 'ToolException' in output)

    # 실행 시간 계산 및 표시
    execution_time = ""
    if call_data.get('start_time') and call_data.get('end_time'):
        duration = call_data['end_time'] - call_data['start_time']
        execution_time = f" ({duration:.2f}초)"

    # 도구 입력에 따른 요약 정보 생성
    summary = _generate_tool_summary(tool_name, tool_input)

    # 기본 정보 표시
    st.markdown(f'**도구:** `{tool_name}`{summary}{execution_time}')
    st.markdown(f'**그룹:** `{tools_namespace}`')

    # 에러 상태인 경우 경고 메시지 표시
    if is_error:
        st.error(f"⚠️ 도구 실행 실패: {error or output}")

    # 상세 정보를 확장 가능한 섹션으로 표시
    with st.expander("상세 정보 보기"):
        st.markdown("##### 입력 데이터")
        st.code(json.dumps(tool_input, indent=2, ensure_ascii=False), language='json')
        
        if call_data.get("finished"):
            st.markdown("##### 실행 결과")
            _render_tool_output(output, error, is_error)


def _generate_tool_summary(tool_name: str, tool_input: Dict[str, Any]) -> str:
    """
    도구 입력에 기반한 요약 정보 생성
    
    Args:
        tool_name: 도구 이름
        tool_input: 도구 입력 데이터
        
    Returns:
        생성된 요약 문자열
    """
    if not isinstance(tool_input, dict):
        return ''
    
    if tool_name == 'firecrawl.scrape' and 'url' in tool_input:
        return f" - `{tool_input['url']}`"
    elif tool_name == 'firecrawl.search' and 'query' in tool_input:
        return f" - `{tool_input['query']}`"
    
    return ''


def _render_tool_output(output: Any, error: str, is_error: bool) -> None:
    """
    도구 출력 결과 렌더링
    
    Args:
        output: 도구 출력
        error: 에러 메시지
        is_error: 에러 상태 여부
    """
    print(f"output: {output}")
    if is_error:
        st.markdown("**에러 상세:**")
        error_text = error or output
        st.code(error_text, language='text')
    elif output is None:
        st.markdown("_(출력 없음)_")
    elif isinstance(output, ToolMessage):
        st.code(output.content)
    elif isinstance(output, str):
        # 긴 텍스트는 일부만 표시
        display_output = output
        if len(output) > 1000:
            display_output = output[:1000] + "\n... (내용이 너무 길어 일부만 표시합니다)"
        st.code(display_output, language='text')
    elif isinstance(output, (dict, list)):
        st.code(json.dumps(output, indent=2, ensure_ascii=False), language='json')
    else:
        st.code(str(output), language='text')


def determine_tool_status(call_data: Dict[str, Any]) -> Tuple[str, str]:
    """
    도구 상태에 따른 UI 상태 및 텍스트 결정
    
    Args:
        call_data: 도구 호출 데이터
        
    Returns:
        (상태 텍스트, UI 상태) 튜플
    """
    is_error = (call_data.get('error') is not None or 
               (isinstance(call_data.get('output'), str) and 
                'ToolException' in call_data.get('output', '')))
    
    if not call_data.get('finished'):
        return "실행 중", "running"
    elif is_error:
        return "실패", "error"
    else:
        return "완료", "complete"


# =============================================================================
# UI 스트리밍 처리 함수
# =============================================================================

async def stream_and_update_ui(response_stream, message_container):
    """
    응답 스트림을 처리하고 실시간으로 UI를 업데이트
    
    이 함수는 에이전트의 응답 스트림을 받아서 텍스트는 실시간으로 표시하고,
    도구 실행 상태는 별도의 상태 컴포넌트로 시각화합니다.
    
    Args:
        response_stream: 에이전트 응답 스트림 이터레이터
        message_container: Streamlit 컨테이너 객체
        
    Returns:
        메시지 구성 요소 리스트 (텍스트 및 도구 호출 정보)
    """
    # UI 상태 관리 변수들
    message_parts = []  # 최종적으로 저장될 메시지 구성 요소들
    active_tool_ui = {}  # 현재 활성화된 도구 UI 컴포넌트들 {run_id: (placeholder, status_ui, call_data)}
    
    # 텍스트 스트리밍을 위한 플레이스홀더
    current_text_placeholder = message_container.empty()
    current_text_content = ""

    async for event in response_stream:
        event_type = event["type"]
        
        if event_type == "content":
            # LLM 텍스트 응답 스트리밍 처리
            current_text_content += event["data"]
            # 커서 표시를 위해 '▌' 문자 추가
            current_text_placeholder.markdown(current_text_content + "▌")

        elif event_type == "tool_start":
            # 도구 실행 시작 시 UI 처리
            # 1. 현재까지의 텍스트를 확정하여 표시
            if current_text_content:
                current_text_placeholder.markdown(current_text_content)
                message_parts.append({"type": "text", "data": current_text_content})
                current_text_content = ""

            # 2. 도구 호출 데이터를 메시지 파트에 추가
            call_data = event["call_data"]
            message_parts.append({"type": "tool_call", "data": call_data})
            
            # 3. 도구 실행 상태를 위한 UI 컴포넌트 생성
            status_placeholder = message_container.empty()
            with status_placeholder:
                status_ui = st.status(f"도구 실행 중: {event['name']}", expanded=True)
                with status_ui:
                    render_tool_call(call_data)
            
            # 활성 도구 UI 목록에 추가
            active_tool_ui[event["run_id"]] = (status_placeholder, status_ui, call_data)

            # 4. 다음 텍스트를 위한 새로운 플레이스홀더 생성
            current_text_placeholder = message_container.empty()
        
        elif event_type == "tool_end":
            # 도구 실행 완료 시 UI 업데이트
            run_id = event["run_id"]
            updated_call_data = event["call_data"]
            if run_id in active_tool_ui:
                status_placeholder, status_ui, old_call_data = active_tool_ui[run_id]
                
                # 기존 데이터를 새로운 데이터로 업데이트
                old_call_data.update(updated_call_data)
                # 도구 상태 결정
                status_text, status_state = determine_tool_status(updated_call_data)

                # UI 업데이트 (확장하지 않은 상태로 변경)
                status_placeholder.empty()
                with status_placeholder:
                    with st.status(f"도구 {status_text}: {updated_call_data['name']}", 
                                 expanded=False, state=status_state):
                        render_tool_call(updated_call_data)
                
                # 완료된 도구는 활성 목록에서 제거
                active_tool_ui.pop(run_id)

        elif event_type == "tool_error":
            # 도구 에러 및 타임아웃 처리
            run_id = event["run_id"]
            updated_call_data = event["call_data"]

            if run_id in active_tool_ui:
                status_placeholder, status_ui, old_call_data = active_tool_ui[run_id]
                
                # 데이터 업데이트
                old_call_data.update(updated_call_data)

                # UI 업데이트
                status_placeholder.empty()
                with status_placeholder:
                    with st.status(f"도구 오류 : {updated_call_data['name']}", 
                                 expanded=False, state="error"):
                        render_tool_call(updated_call_data)
                
                active_tool_ui.pop(run_id)
        
        elif event_type == "stream_error":
            # 스트림 처리 에러 표시
            st.error(f"⚠️ 응답 처리 중 오류가 발생했습니다: {event['error']}")
            if event.get('traceback'):
                with st.expander("오류 상세 정보"):
                    st.code(event['traceback'], language='text')

        # UI 반응성을 위한 짧은 대기
        await asyncio.sleep(0.01)

    # 스트림 종료 후 정리 작업
    # 1. 남은 텍스트 확정
    if current_text_content:
        current_text_placeholder.markdown(current_text_content)
        message_parts.append({"type": "text", "data": current_text_content})
    
    # 2. 아직 완료되지 않은 도구들 강제 완료 처리
    for run_id, (status_placeholder, status_ui, call_data) in active_tool_ui.items():
        if not call_data.get("finished"):
            # 미완료 도구를 실패로 처리
            call_data["output"] = "ToolException: 도구 실행이 정상적으로 완료되지 않았습니다."
            call_data["error"] = "도구 실행이 정상적으로 완료되지 않았습니다."
            call_data["finished"] = True
            call_data["end_time"] = asyncio.get_event_loop().time()
            
            # UI 업데이트
            status_placeholder.empty()
            with status_placeholder:
                with st.status(f"도구 미완료: {call_data['name']}", 
                             expanded=False, state="error"):
                    render_tool_call(call_data)
    
    return message_parts


def generate_history_summary(message_parts: List[Dict[str, Any]]) -> str:
    """
    메시지 파트들로부터 대화 히스토리용 요약 생성
    
    텍스트 응답과 도구 실행 결과를 종합하여 대화 히스토리에 저장할
    요약 메시지를 생성합니다.
    
    Args:
        message_parts: 메시지 구성 요소 리스트
        
    Returns:
        생성된 요약 메시지
    """
    # 텍스트 부분만 추출하여 연결
    text_response = "".join([
        part["data"] for part in message_parts if part["type"] == "text"
    ])
    
    # 텍스트 응답이 있으면 그대로 반환
    if text_response.strip():
        return text_response
    
    # 텍스트 응답이 없고 도구만 사용된 경우 요약 생성
    tool_calls = [part for part in message_parts if part["type"] == "tool_call"]
    if not tool_calls:
        return "응답을 생성하지 못했습니다."
    
    tool_count = len(tool_calls)
    failed_tools = sum(1 for part in tool_calls 
                     if (part["data"].get("error") is not None or 
                         (isinstance(part["data"].get("output"), str) and 
                          'ToolException' in part["data"].get("output", ""))))
    
    # 도구 실행 결과에 따른 요약 메시지 생성
    if failed_tools == tool_count:
        return f"요청을 처리하기 위해 {tool_count}개의 도구를 사용했지만 모두 실패했습니다."
    elif failed_tools > 0:
        return f"{tool_count}개의 도구를 사용했습니다. ({failed_tools}개 실패)"
    else:
        return f"{tool_count}개의 도구를 성공적으로 사용했습니다."


# =============================================================================
# 메인 애플리케이션
# =============================================================================

# =============================================================================
# 프롬프트 관리 UI 컴포넌트
# =============================================================================

def extract_prompt_summary(prompt_text: str) -> str:
    """프롬프트에서 의미있는 요약 정보 추출"""
    if not prompt_text:
        return "내용이 없습니다"
    
    # 주요 키워드들을 찾아서 프롬프트의 목적 파악
    keywords_analysis = {
        "질문 분석": ["질문을 분석", "정보를 추출", "JSON 형식", "검색 키워드"],
        "상품 추천": ["쇼핑 컨설턴트", "상품 추천", "구매 가이드", "전문적"],
        "구조화 출력": ["JSON", "구조화", "템플릿", "형식"],
        "개인화": ["개인화", "맞춤", "사용자", "상황"],
        "전문성": ["전문", "컨설턴트", "분석", "조언"]
    }
    
    found_features = []
    text_lower = prompt_text.lower()
    
    for feature, keywords in keywords_analysis.items():
        if any(keyword.lower() in text_lower for keyword in keywords):
            found_features.append(feature)
    
    if found_features:
        return " • ".join(found_features[:3])  # 최대 3개 특징
    else:
        # fallback: 첫 문장에서 의미있는 부분 추출
        sentences = prompt_text.split('.')
        if sentences:
            first_sentence = sentences[0].strip()
            if len(first_sentence) > 10:
                return first_sentence[:80] + "..."
        return "사용자 정의 프롬프트"

def render_prompt_selector():
    """직관적인 프롬프트 편집 UI"""
    
    # CSS 스타일 추가
    st.markdown("""
    <style>
        .prompt-info {
            background: #e3f2fd;
            color: #1565c0;
            padding: 0.25rem 0.5rem;
            border-radius: 12px;
            font-size: 0.7rem;
            margin: 0.25rem;
            display: inline-block;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # 개별 프롬프트 편집 섹션들
    render_individual_prompt_sections()

def render_individual_prompt_sections():
    """각 프롬프트를 개별 접이식 섹션으로 표시"""
    current_prompt = st.session_state.prompt_manager.get_prompt(st.session_state.active_prompt_name)
    
    if not current_prompt:
        st.error("활성 프롬프트를 찾을 수 없습니다.")
        return
    
    available_prompts = st.session_state.prompt_manager.get_prompt_list()
    
    # 세션 상태에 개별 프롬프트 선택 정보 초기화
    if 'selected_analysis_prompt' not in st.session_state:
        st.session_state.selected_analysis_prompt = "default"
    if 'selected_response_prompt' not in st.session_state:
        st.session_state.selected_response_prompt = "default"
    
    # 질문 분석 프롬프트 섹션
    col_expander, col_selector = st.columns([3.5, 1])
    
    with col_expander:
        with st.expander(f"🔍 질문 분석 프롬프트 - {st.session_state.selected_analysis_prompt}", expanded=False):
            analysis_summary = extract_prompt_summary(current_prompt.get('query_analysis_prompt', ''))
            st.markdown(f'<div class="prompt-info">특징: {analysis_summary}</div>', unsafe_allow_html=True)
            st.caption(f"📝 {len(current_prompt.get('query_analysis_prompt', '')):,}자")
            
            source_prompt_analysis = st.session_state.selected_analysis_prompt
            
            # 편집 가능한 텍스트 영역
            # 선택된 프롬프트에서 내용 가져오기
            if source_prompt_analysis:
                source_data = st.session_state.prompt_manager.get_prompt_by_type(source_prompt_analysis, "query_analysis")
                if source_data:
                    initial_analysis_content = source_data.get('content', '')
                else:
                    initial_analysis_content = current_prompt.get('query_analysis_prompt', '')
            else:
                initial_analysis_content = current_prompt.get('query_analysis_prompt', '')
            
            new_analysis_prompt = st.text_area(
                "질문 분석 프롬프트 편집",
                value=initial_analysis_content,
                height=300,
                key="edit_analysis_prompt",
                help="사용자 질문을 분석하여 구조화된 정보를 추출하는 프롬프트"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 저장", key="save_analysis", use_container_width=True):
                    st.session_state.show_save_analysis_form = True
                    st.session_state.temp_analysis_content_for_save = new_analysis_prompt
                    st.rerun()
            
            # 저장 폼 표시
            if st.session_state.get('show_save_analysis_form', False):
                with st.form("save_analysis_form", clear_on_submit=True):
                    st.markdown("**💾 질문 분석 프롬프트 저장**")
                    save_name = st.text_input(
                        "저장할 프롬프트 이름",
                        value="",
                        placeholder="예: advanced_analysis, custom_prompt_v1",
                        help="새로운 이름으로 저장하거나 기존 이름으로 덮어쓰기 (default는 보호됨)"
                    )
                    
                    col_save1, col_save2, col_save3 = st.columns(3)
                    with col_save1:
                        if st.form_submit_button("✅ 저장 확인", type="primary", use_container_width=True):
                            if save_name:
                                success = save_prompt_as_new(
                                    save_name, 
                                    st.session_state.temp_analysis_content_for_save,
                                    current_prompt.get('model_response_prompt', ''),
                                    'analysis'
                                )
                                if success:
                                    st.session_state.show_save_analysis_form = False
                                    if hasattr(st.session_state, 'temp_analysis_content_for_save'):
                                        del st.session_state.temp_analysis_content_for_save
                                    st.rerun()
                            else:
                                st.warning("프롬프트 이름을 입력해주세요.")
                    
                    with col_save2:
                        if st.form_submit_button("❌ 취소", use_container_width=True):
                            st.session_state.show_save_analysis_form = False
                            if hasattr(st.session_state, 'temp_analysis_content_for_save'):
                                del st.session_state.temp_analysis_content_for_save
                            st.rerun()
            
            with col2:
                if st.button("⚡ 적용", key="temp_apply_analysis", use_container_width=True):
                    # 임시로 메모리에만 저장하고 에이전트 재구성
                    st.session_state.temp_prompts = {
                        'analysis': new_analysis_prompt,
                        'response': current_prompt.get('model_response_prompt', '')
                    }
                    st.session_state.agent = None
                    st.success("⚡ 적용됨!")
                    st.rerun()
    
    with col_selector:
        # 높이 맞춤을 위한 빈 공간 추가
        st.write("")
        
        # 프롬프트 선택 (질문 분석용)
        available_analysis_prompts = st.session_state.prompt_manager.get_prompt_list_by_type("query_analysis")
        
        if len(available_analysis_prompts) > 1:
            # 현재 선택된 프롬프트의 인덱스 찾기
            try:
                current_index = available_analysis_prompts.index(st.session_state.selected_analysis_prompt)
            except ValueError:
                current_index = 0
                st.session_state.selected_analysis_prompt = available_analysis_prompts[0]
            
            selected_analysis_prompt = st.selectbox(
                "프롬프트 선택",
                options=available_analysis_prompts,
                index=current_index,
                key="analysis_prompt_selector",
                help="사용할 질문 분석 프롬프트를 선택하세요.",
                label_visibility="collapsed"
            )
            
            # 선택이 변경된 경우 업데이트
            if selected_analysis_prompt != st.session_state.selected_analysis_prompt:
                st.session_state.selected_analysis_prompt = selected_analysis_prompt
                # 에이전트 재초기화 필요
                st.session_state.agent = None
                st.success(f"✅ 질문 분석 프롬프트가 '{selected_analysis_prompt}'로 변경되었습니다!")
                st.rerun()
        
    
    # 최종 답변 프롬프트 섹션
    col_expander2, col_selector2 = st.columns([3.5, 1])
    
    with col_expander2:
        with st.expander(f"💬 최종 답변 프롬프트 - {st.session_state.selected_response_prompt}", expanded=False):
            response_summary = extract_prompt_summary(current_prompt.get('model_response_prompt', ''))
            st.markdown(f'<div class="prompt-info">특징: {response_summary}</div>', unsafe_allow_html=True)
            st.caption(f"📝 {len(current_prompt.get('model_response_prompt', '')):,}자")
            
            source_prompt_response = st.session_state.selected_response_prompt
            
            # 편집 가능한 텍스트 영역
            # 선택된 프롬프트에서 내용 가져오기
            if source_prompt_response:
                source_data = st.session_state.prompt_manager.get_prompt_by_type(source_prompt_response, "model_response")
                if source_data:
                    initial_response_content = source_data.get('content', '')
                else:
                    initial_response_content = current_prompt.get('model_response_prompt', '')
            else:
                initial_response_content = current_prompt.get('model_response_prompt', '')
                
            if hasattr(st.session_state, 'temp_response_content'):
                initial_response_content = st.session_state.temp_response_content
                del st.session_state.temp_response_content
            
            new_response_prompt = st.text_area(
                "최종 답변 프롬프트 편집",
                value=initial_response_content,
                height=300,
                key="edit_response_prompt",
                help="수집된 정보를 바탕으로 최종 답변을 생성하는 프롬프트"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 저장", key="save_response", use_container_width=True):
                    st.session_state.show_save_response_form = True
                    st.session_state.temp_response_content_for_save = new_response_prompt
                    st.rerun()
            
            # 저장 폼 표시
            if st.session_state.get('show_save_response_form', False):
                with st.form("save_response_form", clear_on_submit=True):
                    st.markdown("**💾 최종 답변 프롬프트 저장**")
                    save_name = st.text_input(
                        "저장할 프롬프트 이름",
                        value="",
                        placeholder="예: advanced_response, custom_prompt_v1",
                        help="새로운 이름으로 저장하거나 기존 이름으로 덮어쓰기 (default는 보호됨)"
                    )
                    
                    col_save1, col_save2 = st.columns(2)
                    with col_save1:
                        if st.form_submit_button("✅ 저장 확인", type="primary", use_container_width=True):
                            if save_name:
                                success = save_prompt_as_new(
                                    save_name,
                                    current_prompt.get('query_analysis_prompt', ''),
                                    st.session_state.temp_response_content_for_save,
                                    'response'
                                )
                                if success:
                                    st.session_state.show_save_response_form = False
                                    if hasattr(st.session_state, 'temp_response_content_for_save'):
                                        del st.session_state.temp_response_content_for_save
                                    st.rerun()
                            else:
                                st.warning("프롬프트 이름을 입력해주세요.")
                    
                    with col_save2:
                        if st.form_submit_button("❌ 취소", use_container_width=True):
                            st.session_state.show_save_response_form = False
                            if hasattr(st.session_state, 'temp_response_content_for_save'):
                                del st.session_state.temp_response_content_for_save
                            st.rerun()
            
            with col2:
                if st.button("⚡ 적용", key="temp_apply_response", use_container_width=True):
                    # 임시로 메모리에만 저장하고 에이전트 재구성
                    current_analysis = st.session_state.temp_prompts.get('analysis') if hasattr(st.session_state, 'temp_prompts') else current_prompt.get('query_analysis_prompt', '')
                    st.session_state.temp_prompts = {
                        'analysis': current_analysis,
                        'response': new_response_prompt
                    }
                    st.session_state.agent = None
                    st.success("⚡ 적용됨!")
                    st.rerun()
    
    with col_selector2:
        # 높이 맞춤을 위한 빈 공간 추가
        st.write("")
        
        # 프롬프트 선택 (최종 답변용)
        available_response_prompts = st.session_state.prompt_manager.get_prompt_list_by_type("model_response")
        
        if len(available_response_prompts) > 1:
            # 현재 선택된 프롬프트의 인덱스 찾기
            try:
                current_index = available_response_prompts.index(st.session_state.selected_response_prompt)
            except ValueError:
                current_index = 0
                st.session_state.selected_response_prompt = available_response_prompts[0]
            
            selected_response_prompt = st.selectbox(
                "프롬프트 선택",
                options=available_response_prompts,
                index=current_index,
                key="response_prompt_selector",
                help="사용할 최종 답변 프롬프트를 선택하세요.",
                label_visibility="collapsed"
            )
            
            # 선택이 변경된 경우 업데이트
            if selected_response_prompt != st.session_state.selected_response_prompt:
                st.session_state.selected_response_prompt = selected_response_prompt
                # 에이전트 재초기화 필요
                st.session_state.agent = None
                st.success(f"✅ 최종 답변 프롬프트가 '{selected_response_prompt}'로 변경되었습니다!")
                st.rerun()
        

def save_prompt_section(current_prompt, section_key, new_content):
    """프롬프트 섹션 저장"""
    try:
        # 기본 프롬프트 데이터 복사
        updated_data = current_prompt.copy()
        updated_data[section_key] = new_content
        
        # 프롬프트 업데이트
        result = st.session_state.prompt_manager.update_prompt(
            prompt_id=current_prompt['id'],
            name=current_prompt['name'],
            query_analysis_prompt=updated_data.get('query_analysis_prompt', ''),
            model_response_prompt=updated_data.get('model_response_prompt', '')
        )
        
        return result is not None
    except Exception as e:
        st.error(f"❌ 저장 실패: {e}")
        return False


def save_prompt_as_new(new_name, analysis_content, response_content, section_type):
    """새로운 이름으로 프롬프트 저장 (타입별 독립 저장)"""
    try:
        # 기본 프롬프트 보호 로직
        if new_name == 'default':
            st.warning("⚠️ 기본 프롬프트는 덮어쓸 수 없습니다. 다른 이름을 사용해주세요.")
            return False
        
        # section_type에 따라 해당 타입만 저장
        if section_type == 'analysis':
            # 질문 분석 프롬프트만 저장
            prompt_type = "query_analysis"
            content = analysis_content
            prompt_name = new_name  # suffix 제거
        elif section_type == 'response':
            # 최종 답변 프롬프트만 저장
            prompt_type = "model_response"
            content = response_content
            prompt_name = new_name  # suffix 제거
        else:
            st.error("❌ 유효하지 않은 섹션 타입입니다.")
            return False
        
        # 기존 프롬프트가 있는지 확인
        existing_prompt = st.session_state.prompt_manager.get_prompt_by_type(prompt_name, prompt_type)
        
        if existing_prompt:
            # 기존 프롬프트 업데이트 확인
            if st.session_state.get(f'confirm_overwrite_{section_type}', False):
                result = st.session_state.prompt_manager.update_prompt_by_type(
                    prompt_id=existing_prompt['id'],
                    name=prompt_name,
                    content=content,
                    prompt_type=prompt_type
                )
                
                if result:
                    st.success(f"✅ '{new_name}' {section_type} 프롬프트가 업데이트되었습니다!")
                    st.session_state[f'confirm_overwrite_{section_type}'] = False
                    return True
            else:
                st.warning(f"⚠️ '{new_name}' {section_type} 프롬프트가 이미 존재합니다. 덮어쓰시겠습니까?")
                if st.button("🔄 덮어쓰기 확인", key=f"confirm_overwrite_{new_name}_{section_type}"):
                    st.session_state[f'confirm_overwrite_{section_type}'] = True
                    st.rerun()
                return False
        else:
            # 새 프롬프트 생성 (타입별로 독립적으로)
            result = st.session_state.prompt_manager.create_prompt_by_type(
                name=prompt_name,
                content=content,
                prompt_type=prompt_type
            )
            
            if result:
                st.success(f"✅ '{new_name}' {section_type} 프롬프트가 새로 저장되었습니다!")
                return True
        
        return False
        
    except Exception as e:
        st.error(f"❌ 저장 실패: {e}")
        return False




def main():
    """
    Streamlit 애플리케이션의 메인 진입점
    
    이 함수는 전체 애플리케이션의 UI를 구성하고 사용자 인터랙션을 처리합니다.
    채팅 인터페이스, 메시지 히스토리 표시, 새로운 메시지 처리 등을 담당합니다.
    
    에이전트 초기화 최적화:
    - 앱 시작 시 한 번만 초기화
    - 사용자 입력 시 추가 초기화 방지
    """
    # 애플리케이션 헤더
    st.title("🛍️ AI 쇼핑 어시스턴트")
    
    # 시스템 상태 (간단한 상태 표시)
    # if st.session_state.agent is not None:
    #     st.success("🤖 에이전트: **활성화**")
    # else:
    #     st.error("🤖 에이전트: **비활성화**")
    
    # 프롬프트 선택 및 편집 UI
    # render_prompt_selector()

    st.markdown("---")
    st.markdown("### 💬 대화")
    st.markdown("무엇을 찾아드릴까요? 원하는 상품에 대해 자세히 알려주세요.")

    # 에이전트 초기화 (앱 시작 시 한 번만)
    if st.session_state.agent is None:
        # 초기화 시도
        initialization_success = asyncio.run(initialize_agent())
        if not initialization_success:
            st.warning("⚠️ 에이전트 초기화가 실패했습니다. 일부 기능이 제한될 수 있습니다.")
            return  # 초기화 실패 시 더 이상 진행하지 않음
        else:
            # 초기화 성공 시 UI 새로고침으로 상태 반영
            st.rerun()

    # 에이전트가 준비된 후에만 UI 표시
    if st.session_state.agent is not None:
        # 이전 대화 내용 표시
        display_conversation_history()
        
        # 사용자 입력 처리
        handle_user_input()
    else:
        st.info("🔄 에이전트를 초기화하는 중입니다. 잠시만 기다려주세요.")
        st.button("🔄 다시 시도", on_click=lambda: st.rerun())


def display_conversation_history():
    """
    이전 대화 내용을 화면에 표시
    
    세션 상태에 저장된 메시지 히스토리를 순회하며 각 메시지를
    적절한 형태로 렌더링합니다.
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and message.get("parts"):
                # 어시스턴트 메시지의 각 파트를 순서대로 표시
                for part in message["parts"]:
                    if part["type"] == "text":
                        st.markdown(part["data"])
                    elif part["type"] == "tool_call":
                        # 도구 호출 정보를 상태 컴포넌트로 표시
                        call_data = part["data"]
                        status_text, status_state = determine_tool_status(call_data)

                        with st.status(f"도구 {status_text}: {call_data['name']}", 
                                     expanded=False, state=status_state):
                            render_tool_call(call_data)
            else:
                # 사용자 메시지는 단순 텍스트로 표시
                st.markdown(message.get("content", ""))


def handle_user_input():
    """
    사용자 입력을 처리하고 응답을 생성
    
    사용자가 새로운 메시지를 입력했을 때 호출되며,
    에이전트에게 질의하고 실시간으로 응답을 표시합니다.
    
    에이전트 상태 확인:
    - 입력 처리 전 에이전트 준비 상태 검증
    - 에이전트가 없는 경우 안전하게 처리
    """
    # 사용자 입력 받기
    if prompt := st.chat_input("여기에 질문을 입력하세요..."):
        
        # 에이전트 준비 상태 확인
        if not st.session_state.agent:
            st.error("❌ 에이전트가 준비되지 않았습니다. 페이지를 새로고침해주세요.")
            return
        
        # 사용자 메시지를 히스토리에 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 사용자 메시지 표시
        with st.chat_message("user"):
            st.markdown(prompt)

        # 어시스턴트 응답 영역
        with st.chat_message("assistant"):
            message_container = st.container()
            
            try:
                # 응답 스트림 생성
                response_stream = get_response(
                    st.session_state.agent, 
                    prompt, 
                    st.session_state.history
                )
                
                # 실시간 UI 업데이트 및 메시지 파트 수집
                message_parts = asyncio.run(stream_and_update_ui(response_stream, message_container))

                # 어시스턴트 메시지를 세션에 저장
                st.session_state.messages.append({
                    "role": "assistant",
                    "parts": message_parts
                })

                # LangChain 히스토리 업데이트
                assistant_summary = generate_history_summary(message_parts)
                st.session_state.history.append(("user", prompt))
                st.session_state.history.append(("assistant", assistant_summary))

            except Exception as e:
                st.error(f"❌ 응답 생성 중 오류가 발생했습니다: {str(e)}")
                st.info("다시 시도해주세요.")

        # 페이지 새로고침으로 UI 상태 정리
        st.rerun()


# =============================================================================
# 애플리케이션 진입점
# =============================================================================

if __name__ == "__main__":
    main()