#!/usr/bin/env python3
"""
Enhanced Shopping Agent 설정 비교 테스트
다양한 설정 모드의 성능과 API 호출 횟수를 비교합니다.
"""

import asyncio
import os
import sys
import pytest
from typing import Dict, Any

# 프로젝트 루트를 Python 패스에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from agent.enhanced_shopping_agent import EnhancedShoppingAgent
from config.agent_config import get_config


class TestConfigComparison:
    """설정 비교 테스트 클래스"""
    
    @pytest.fixture
    def test_query(self):
        """테스트에 사용할 표준 쿼리"""
        return "겨울용 패딩 점퍼 추천해줘. 10만원 이하로"
    
    @pytest.fixture
    def config_modes(self):
        """테스트할 설정 모드들"""
        return {
            "절약 모드": "credit_saving",
            "기본 설정": "default", 
            "성능 테스트": "performance"
        }
    
    async def test_config_loading(self, config_modes):
        """설정 로딩 테스트"""
        print("\n=== 설정 로딩 테스트 ===")
        
        for config_name, config_key in config_modes.items():
            print(f"🔧 {config_name} ({config_key}) 로딩 중...")
            
            config = get_config(config_key)
            assert config is not None, f"{config_key} 설정 로딩 실패"
            
            # 설정 유효성 검증
            assert config.search.max_keywords_to_search > 0
            assert config.search.total_max_search_results > 0
            assert config.scraping.max_urls_to_scrape > 0
            
            print(f"  ✅ 성공 - 검색: {config.search.max_keywords_to_search}키워드, "
                  f"스크래핑: {config.scraping.max_urls_to_scrape}URL")
    
    async def test_query_analysis_comparison(self, test_query, config_modes):
        """질문 분석 비교 테스트"""
        print("\n=== 질문 분석 비교 테스트 ===")
        
        analysis_results = {}
        
        for config_name, config_key in config_modes.items():
            print(f"🧠 {config_name} 질문 분석 중...")
            
            config = get_config(config_key)
            agent = EnhancedShoppingAgent(config)
            
            initial_state = {
                "user_query": test_query,
                "messages": [],
                "processing_status": "시작"
            }
            
            result = await agent.analyze_query(initial_state)
            
            assert result.get('analyzed_query') is not None, f"{config_name} 질문 분석 실패"
            
            analyzed = result['analyzed_query']
            keywords = analyzed.get('search_keywords', [])
            
            analysis_results[config_name] = {
                "keywords": keywords,
                "main_product": analyzed.get('main_product'),
                "price_range": analyzed.get('price_range'),
                "search_intent": analyzed.get('search_intent')
            }
            
            print(f"  ✅ 키워드 {len(keywords)}개: {keywords}")
            print(f"     상품: {analyzed.get('main_product')}")
            print(f"     가격: {analyzed.get('price_range')}")
        
        return analysis_results
    
    async def test_api_call_estimation(self, test_query, config_modes):
        """API 호출 횟수 추정 테스트"""
        print("\n=== API 호출 횟수 추정 테스트 ===")
        
        estimation_results = {}
        
        for config_name, config_key in config_modes.items():
            print(f"📊 {config_name} API 호출 추정 중...")
            
            config = get_config(config_key)
            agent = EnhancedShoppingAgent(config)
            
            # 질문 분석으로 키워드 추출
            initial_state = {
                "user_query": test_query,
                "messages": [],
                "processing_status": "시작"
            }
            
            result = await agent.analyze_query(initial_state)
            keywords = result.get('analyzed_query', {}).get('search_keywords', [])
            
            # API 호출 횟수 계산
            estimated_tavily = min(
                config.search.max_keywords_to_search,
                len(keywords)
            )
            estimated_firecrawl = config.scraping.max_urls_to_scrape
            total_calls = estimated_tavily + estimated_firecrawl
            
            estimation_results[config_name] = {
                "tavily_calls": estimated_tavily,
                "firecrawl_calls": estimated_firecrawl,
                "total_calls": total_calls,
                "config": config
            }
            
            print(f"  📈 Tavily: {estimated_tavily}회, Firecrawl: {estimated_firecrawl}회, 총: {total_calls}회")
        
        return estimation_results
    
    def test_efficiency_comparison(self, estimation_results):
        """효율성 비교 테스트"""
        print("\n=== 효율성 비교 테스트 ===")
        
        # 결과 정렬 (API 호출 횟수 기준)
        sorted_results = sorted(
            estimation_results.items(), 
            key=lambda x: x[1]['total_calls']
        )
        
        print("🏆 효율성 순위:")
        for i, (config_name, result) in enumerate(sorted_results, 1):
            efficiency_score = 100 - (result['total_calls'] * 10)  # 임의의 효율성 점수
            print(f"  {i}. {config_name}: {result['total_calls']}회 호출 (효율성: {efficiency_score}점)")
        
        # 절약 효과 계산
        if len(sorted_results) >= 2:
            most_efficient = sorted_results[0][1]
            least_efficient = sorted_results[-1][1]
            
            savings = least_efficient['total_calls'] - most_efficient['total_calls']
            savings_percent = (savings / least_efficient['total_calls']) * 100
            
            print(f"\n💰 최대 절약 효과:")
            print(f"  최소 호출: {most_efficient['total_calls']}회")
            print(f"  최대 호출: {least_efficient['total_calls']}회")
            print(f"  절약량: {savings}회 ({savings_percent:.1f}%)")
        
        return sorted_results
    
    async def test_config_validation(self, config_modes):
        """설정 유효성 검증 테스트"""
        print("\n=== 설정 유효성 검증 테스트 ===")
        
        for config_name, config_key in config_modes.items():
            print(f"✅ {config_name} 유효성 검증 중...")
            
            config = get_config(config_key)
            
            # 검색 설정 검증
            assert 1 <= config.search.max_keywords_to_search <= 10, "검색 키워드 수가 범위를 벗어남"
            assert 1 <= config.search.total_max_search_results <= 50, "검색 결과 수가 범위를 벗어남"
            assert config.search.search_depth in ["basic", "advanced"], "잘못된 검색 깊이"
            
            # 스크래핑 설정 검증
            assert 1 <= config.scraping.max_urls_to_scrape <= 10, "스크래핑 URL 수가 범위를 벗어남"
            assert config.scraping.content_max_length > 0, "콘텐츠 최대 길이가 0 이하"
            
            print(f"  ✅ {config_name} 유효성 검증 완료")


async def run_comparison_tests():
    """비교 테스트 실행"""
    print("🚀 Enhanced Shopping Agent 설정 비교 테스트 시작\n")
    
    tester = TestConfigComparison()
    
    # 테스트 데이터 준비
    test_query = "겨울용 패딩 점퍼 추천해줘. 10만원 이하로"
    config_modes = {
        "절약 모드": "credit_saving",
        "기본 설정": "default", 
        "성능 테스트": "performance"
    }
    
    try:
        # 1. 설정 로딩 테스트
        await tester.test_config_loading(config_modes)
        
        # 2. 설정 유효성 검증
        await tester.test_config_validation(config_modes)
        
        # 3. 질문 분석 비교
        analysis_results = await tester.test_query_analysis_comparison(test_query, config_modes)
        
        # 4. API 호출 횟수 추정
        estimation_results = await tester.test_api_call_estimation(test_query, config_modes)
        
        # 5. 효율성 비교
        efficiency_ranking = tester.test_efficiency_comparison(estimation_results)
        
        print("\n🎉 모든 테스트 완료!")
        
        return {
            "analysis_results": analysis_results,
            "estimation_results": estimation_results,
            "efficiency_ranking": efficiency_ranking
        }
        
    except Exception as e:
        print(f"\n❌ 테스트 실행 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


async def show_config_details():
    """설정 상세 정보 표시"""
    print("\n=== 📋 설정 상세 정보 ===")
    
    configs = {
        "credit_saving": "절약 모드 (최소한의 API 호출)",
        "default": "기본 모드 (균형잡힌 성능)",
        "performance": "성능 모드 (최대 검색 범위)"
    }
    
    for key, description in configs.items():
        config = get_config(key)
        print(f"\n🔧 {key}:")
        print(f"   📝 {description}")
        print(f"   🔍 검색: {config.search.max_keywords_to_search}개 키워드, 최대 {config.search.total_max_search_results}개 결과")
        print(f"   📄 스크래핑: 최대 {config.scraping.max_urls_to_scrape}개 URL")
        print(f"   📏 콘텐츠 길이: 최대 {config.scraping.content_max_length:,}자")


if __name__ == "__main__":
    async def main():
        """메인 실행 함수"""
        
        # 비교 테스트 실행
        test_results = await run_comparison_tests()
        
        # 설정 상세 정보 표시
        await show_config_details()
        
        # 사용법 안내
        print(f"\n💡 설정 변경 방법:")
        print(f"   1. src/config/agent_config.py에서 직접 수정")
        print(f"   2. build_enhanced_agent('config_name') 함수에 설정명 전달")
        print(f"   3. EnhancedShoppingAgent(config) 생성시 설정 객체 전달")
        
        print(f"\n📖 테스트 실행 방법:")
        print(f"   python tests/test_config_comparison.py")
        print(f"   pytest tests/test_config_comparison.py -v")
    
    asyncio.run(main())