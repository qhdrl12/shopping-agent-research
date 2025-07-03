# -*- coding: utf-8 -*-
"""
이 스크립트는 LangChain, LangGraph 및 MCP(Model-View-Controller Protocol)를 사용하여
다양한 도구와 상호 작용할 수 있는 AI 에이전트를 구축하고 실행하는 방법을 보여줍니다.
주요 기능은 다음과 같습니다:
- Zapier, Firecrawl, Playwright 등 외부 서비스의 도구를 MCP를 통해 동적으로 로드합니다.
- 로드된 도구를 LangChain 에이전트가 사용할 수 있는 형식으로 변환합니다.
- 인증 오류와 같은 특정 MCP 오류를 처리하기 위한 래퍼(wrapper)를 제공합니다.
- ReAct 프레임워크를 기반으로 하는 에이전트를 생성하고 실행합니다.
"""

import asyncio
import os
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import tool, StructuredTool, ToolException
from langchain_openai import ChatOpenAI

from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from mcp import ClientSession, McpError
from mcp.client.streamable_http import streamablehttp_client
from src.tools.datetime import get_current_time

# 에이전트가 사용할 특정 도구 목록 (현재 코드에서는 직접 사용되지 않음)
target_tools = ["tavily_search"]


async def get_zapier_mcp(
    server_url: str = "https://mcp.zapier.com/api/mcp/s/ZDc5MDU5ZjYtMGRiYy00YjhkLTg3YWUtMzBhZDg2ZDllNWFjOmNmYzE4ZTYwLTMyNGQtNDljMC1iYTcyLTM5ODkyMzY3OTc5NQ==/mcp",
):
    """
    지정된 Zapier MCP 서버 URL에 연결하여 사용 가능한 모든 도구를 가져와
    LangChain 호환 도구 목록으로 변환합니다.

    Args:
        server_url (str): 연결할 Zapier MCP 서버의 URL.

    Returns:
        list: LangChain의 `StructuredTool` 객체 목록.
    """
    # MCP 서버와 통신하기 위한 HTTP 전송 계층 설정
    transport = StreamableHttpTransport(server_url)
    # MCP 클라이언트 생성
    client = Client(transport=transport)
    tools = []

    # 클라이언트 컨텍스트 내에서 비동기 작업 수행
    async with client:
        # 서버에서 사용 가능한 MCP 도구 목록을 가져옴
        mcp_tools = await client.list_tools()

        # 각 MCP 도구를 LangChain 도구로 변환
        for mcp_tool in mcp_tools:
            langchian_tool = await create_langchain_mcp_tool(mcp_tool, server_url)
            # 필요에 따라 인증 오류 처리를 위한 래퍼를 추가할 수 있습니다.
            # wrap_tool = await wrap_mcp_auth_tool(langchian_tool)
            # print(f"wrap_tool: {wrap_tool}")
            # tools.append(wrap_tool)
            tools.append(langchian_tool)
        
        return tools


async def create_langchain_mcp_tool(
    mcp_tool, mcp_server_url: str = "", headers: dict | None = None
):
    """
    MCP 도구 정의를 기반으로 LangChain의 `StructuredTool`을 동적으로 생성합니다.
    이 함수는 LangChain의 `@tool` 데코레이터를 사용하여 MCP 도구를 래핑하는 함수를 반환합니다.

    Args:
        mcp_tool: 변환할 MCP 도구 객체. (이름, 설명, 입력 스키마 포함)
        mcp_server_url (str): MCP 서버의 URL.
        headers (dict | None): 요청에 추가할 사용자 지정 헤더.

    Returns:
        StructuredTool: LangChain 에이전트가 사용할 수 있는 새로운 도구.
    """

    # LangChain의 @tool 데코레이터를 사용하여 동적으로 도구 생성
    # mcp_tool의 메타데이터(이름, 설명, 입력 스키마)를 그대로 사용
    @tool(mcp_tool.name, description=mcp_tool.description, args_schema=mcp_tool.inputSchema)
    async def new_tool(**kwargs):
        """이 내부 함수는 실제 도구 호출 로직을 포함합니다."""
        # MCP 서버와 스트리밍 HTTP 연결을 설정
        async with streamablehttp_client(mcp_server_url, headers=headers) as streams:
            read_stream, write_stream, _ = streams
            # MCP 클라이언트 세션을 시작
            async with ClientSession(read_stream, write_stream) as tool_session:
                # 세션을 초기화하고 실제 MCP 도구를 호출
                await tool_session.initialize()
                return await tool_session.call_tool(mcp_tool.name, arguments=kwargs)

    return new_tool


async def wrap_mcp_auth_tool(tool: StructuredTool) -> StructuredTool:
    """
    LangChain 도구를 래핑하여 MCP 관련 인증 오류를 처리합니다.
    특히, 사용자 상호작용이 필요한 오류(예: OAuth 로그인)를 감지하고,
    사용자에게 방문해야 할 URL이 포함된 `ToolException`을 발생시킵니다.

    Args:
        tool (StructuredTool): 래핑할 LangChain 도구.

    Returns:
        StructuredTool: 오류 처리 로직이 추가된 래핑된 도구.
    """
    print(f"StructuredTool: {tool}")
    # 기존 도구의 비동기 실행 함수를 저장
    old_coroutine = tool.coroutine

    async def wrapped_mcp_coroutine(**kwargs):
        """래핑된 새로운 비동기 실행 함수입니다."""

        def _find_first_mcp_error_nested(exc: BaseException) -> McpError | None:
            """중첩된 예외 구조에서 첫 번째 McpError를 재귀적으로 찾습니다."""
            if isinstance(exc, McpError):
                return exc
            # ExceptionGroup은 여러 예외를 포함할 수 있으므로 재귀적으로 탐색
            if isinstance(exc, ExceptionGroup):
                for sub_exc in exc.exceptions:
                    if found := _find_first_mcp_error_nested(sub_exc):
                        return found
            return None

        try:
            # 원래 도구의 로직을 실행
            return await old_coroutine(**kwargs)
        except BaseException as e_orig:
            # 발생한 예외에서 McpError를 찾음
            mcp_error = _find_first_mcp_error_nested(e_orig)

            if not mcp_error:
                raise e_orig  # McpError가 아니면 원래 예외를 다시 발생시킴

            error_details = mcp_error.error
            # 오류 코드가 -32003이면 사용자 상호작용이 필요함을 의미 (MCP 표준)
            is_interaction_required = getattr(error_details, "code", None) == -32003
            error_data = getattr(error_details, "data", None) or {}

            if is_interaction_required:
                # 사용자에게 표시할 오류 메시지와 URL을 구성
                message_payload = error_data.get("messages", {})
                error_message_text = "Required interaction"
                if isinstance(message_payload, dict):
                    error_message_text = message_payload.get("text") or error_message_text

                if url := error_data.get("url"):
                    error_message_text = f'{error_message_text} {url}'
                # LangChain 에이전트가 처리할 수 있는 ToolException을 발생시켜
                # 사용자에게 인증 URL을 안내하도록 함
                raise ToolException(error_message_text) from e_orig

            raise e_orig  # 그 외의 McpError는 다시 발생시킴

    # 도구의 실행 함수를 래핑된 함수로 교체
    tool.coroutine = wrapped_mcp_coroutine
    return tool


async def get_tools():
    """
    에이전트가 사용할 도구들을 설정하고 반환합니다.
    MultiServerMCPClient를 사용하여 firecrawl, playwright, filesystem과 같은
    로컬 MCP 서버를 실행하고 해당 도구들을 가져옵니다.
    """
    # 환경 변수에서 FireCrawl API 키를 가져옴
    fc_api_key = os.getenv("FIRECRAWL_API_KEY")
    if not fc_api_key:
        raise RuntimeError(
            "⚠️  FIRECRAWL_API_KEY not found. "
            "Add it to a .env file or export it before running."
        )


    """에이전트가 사용할 도구들을 반환합니다."""
    # 여러 MCP 서버를 관리하는 클라이언트 설정
    client = MultiServerMCPClient(
        {
            # FireCrawl MCP 서버 설정: 웹 크롤링 도구
            "firecrawl": {
                "command": "npx",
                "args": ["-y", "firecrawl-mcp"],
                "env": {"FIRECRAWL_API_KEY": fc_api_key},
                "transport": "stdio",  # 표준 입출력으로 통신
            },
            # Playwright MCP 서버 설정: 브라우저 자동화 도구
            "playwright": {
                "command": "npx",
                "args": [
                    "@playwright/mcp@latest"
                ],
                "transport": "stdio",
            },
            # 파일 시스템 접근을 위한 MCP 서버 설정
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/jenson.lee/Demo"],
                "transport": "stdio",
            }
        }
    )    

    # 설정된 모든 서버에서 도구를 비동기적으로 가져옴
    tools = await client.get_tools()
    print(f"tools: {tools}")
    # tools.append(get_current_time)
    return tools


async def build_agent() -> CompiledStateGraph:
    """
    쇼핑 에이전트를 빌드하고 컴파일된 상태 그래프를 반환합니다.
    LLM, 도구들을 사용하여 ReAct 기반 에이전트를 생성합니다.
    """

    # 사용할 LLM 설정 (GPT-4.1-mini)
    # llm = ChatOpenAI(model="o4-mini", reasoning_effort="medium")
    llm = ChatOpenAI(model="gpt-4.1")

    # `get_tools` 함수를 호출하여 에이전트가 사용할 도구 목록을 가져옴
    tools = await get_tools()

    # LLM과 도구를 사용하여 ReAct 에이전트를 생성
    return create_react_agent(llm, tools)



async def main():
    """
    메인 실행 함수. Zapier 도구를 가져와 에이전트를 생성하고,
    미리 정의된 질문으로 에이전트를 실행하여 응답을 스트리밍합니다.
    이 함수는 주로 `get_zapier_mcp`의 사용법을 시연하는 예제입니다.
    """
    # Zapier MCP 서버에서 도구를 가져옴
    tools = await get_zapier_mcp()
    
    # LLM 설정
    llm = ChatOpenAI(model="gpt-4.1", temperature=0)

    # LLM과 Zapier 도구로 ReAct 에이전트 생성
    agent = create_react_agent(llm, tools)

    # 에이전트의 응답을 비동기 스트림으로 받음
    async for chunk in agent.astream(
        {"messages": ["구글에서 오늘 날씨 검색해줘. 오늘 날씨가 몇일인지 포함해줘. 기후가 안좋다면 안좋은 시간대를 같이 알려줘야해."]},
        stream_mode="values"
    ):
        # 각 청크에서 메시지 목록을 추출
        messages = chunk["messages"]

        # 각 메시지를 예쁘게 출력
        for msg in messages:
            msg.pretty_print()
        print("\n")

# 이 스크립트가 직접 실행될 때 main 함수를 비동기적으로 실행
if __name__ == "__main__":
    asyncio.run(main())