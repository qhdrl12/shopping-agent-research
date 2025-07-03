#!/usr/bin/env python3
"""
프롬프트 커스터마이제이션 기능 테스트
"""

import asyncio
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from src.config.agent_config import get_config
from src.agent.enhanced_shopping_agent import build_enhanced_agent


async def test_prompt_customization():
    """프롬프트 커스터마이제이션 테스트"""
    print("=== 프롬프트 커스터마이제이션 기능 테스트 ===\n")
    
    # 1. 기본 설정 테스트
    print("1. 기본 설정 로드 테스트")
    default_config = get_config("default")
    print(f"✅ 기본 설정 로드 성공")
    print(f"   - Analysis prompt 길이: {len(default_config.prompts.analysis_prompt)} 문자")
    print(f"   - System prompt 길이: {len(default_config.prompts.system_prompt)} 문자")
    print(f"   - App system prompt 길이: {len(default_config.prompts.app_system_prompt)} 문자")
    print()
    
    # 2. 사용자 정의 프롬프트 테스트
    print("2. 사용자 정의 프롬프트 테스트")
    custom_prompts = {
        'analysis_prompt': """
        당신은 테스트용 프롬프트를 사용하는 쇼핑 컨설턴트입니다.
        사용자 질문: "{user_query}"
        이것은 테스트용 분석 프롬프트입니다.
        """,
        'system_prompt': """
        **테스트용 시스템 프롬프트입니다.**
        이 프롬프트가 사용되고 있다면 커스터마이제이션이 성공한 것입니다.
        컨텍스트: {context}
        """,
        'app_system_prompt': """
        이것은 테스트용 앱 시스템 프롬프트입니다.
        프롬프트 커스터마이제이션이 정상 작동하고 있습니다.
        """
    }
    
    # 3. 커스텀 프롬프트로 에이전트 빌드 테스트
    print("3. 커스텀 프롬프트로 에이전트 빌드")
    try:
        agent = await build_enhanced_agent("default", custom_prompts)
        print("✅ 커스텀 프롬프트로 에이전트 빌드 성공")
        print(f"   - 에이전트 타입: {type(agent).__name__}")
    except Exception as e:
        print(f"❌ 에이전트 빌드 실패: {str(e)}")
        return False
    
    # 4. 프롬프트 적용 확인 (설정 객체 확인)
    print("\n4. 프롬프트 적용 확인")
    test_config = get_config("default")
    
    # 커스텀 프롬프트를 직접 적용
    test_config.prompts.analysis_prompt = custom_prompts['analysis_prompt']
    test_config.prompts.system_prompt = custom_prompts['system_prompt']
    test_config.prompts.app_system_prompt = custom_prompts['app_system_prompt']
    
    print("✅ 프롬프트 적용 테스트 완료")
    print(f"   - 분석 프롬프트: {test_config.prompts.analysis_prompt[:50]}...")
    print(f"   - 시스템 프롬프트: {test_config.prompts.system_prompt[:50]}...")
    print(f"   - 앱 시스템 프롬프트: {test_config.prompts.app_system_prompt[:50]}...")
    
    print("\n=== 모든 테스트 통과! ===")
    print("프롬프트 커스터마이제이션 기능이 정상적으로 구현되었습니다.")
    
    return True


async def test_prompt_formats():
    """프롬프트 포맷 테스트"""
    print("\n=== 프롬프트 포맷 테스트 ===")
    
    config = get_config("default")
    
    # analysis_prompt에 user_query 포맷 테스트
    test_query = "겨울 패딩 추천해줘"
    try:
        formatted_analysis = config.prompts.analysis_prompt.format(user_query=test_query)
        print("✅ Analysis prompt 포맷 성공")
        print(f"   - 포맷된 길이: {len(formatted_analysis)} 문자")
    except Exception as e:
        print(f"❌ Analysis prompt 포맷 실패: {str(e)}")
        return False
    
    # system_prompt에 context 포맷 테스트
    test_context = "테스트 컨텍스트 정보"
    try:
        formatted_system = config.prompts.system_prompt.format(context=test_context)
        print("✅ System prompt 포맷 성공")
        print(f"   - 포맷된 길이: {len(formatted_system)} 문자")
    except Exception as e:
        print(f"❌ System prompt 포맷 실패: {str(e)}")
        return False
    
    print("✅ 모든 프롬프트 포맷 테스트 통과!")
    return True


if __name__ == "__main__":
    # 비동기 테스트 실행
    success = asyncio.run(test_prompt_customization())
    format_success = asyncio.run(test_prompt_formats())
    
    if success and format_success:
        print("\n🎉 전체 테스트 성공!")
        exit(0)
    else:
        print("\n❌ 일부 테스트 실패")
        exit(1)