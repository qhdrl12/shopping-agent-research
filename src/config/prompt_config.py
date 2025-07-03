"""
Enhanced Shopping Agent 프롬프트 설정 관리

이 모듈은 Enhanced Shopping Agent에서 사용되는 프롬프트들을 관리합니다.
Streamlit UI에서 실시간으로 수정하고 검증할 수 있는 기능을 제공합니다.
"""

from typing import Dict, Any, List
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class PromptConfig:
    """프롬프트 설정을 관리하는 데이터클래스"""
    
    # 1. 질문 분석 프롬프트 (analyze_query 단계에서 사용)
    analysis_prompt_template: str = """
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
    
    # 2. 시스템 프롬프트 (react_agent 단계에서 사용)
    system_prompt_template: str = """**당신은 전문 쇼핑 컨설턴트입니다.**

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
    
    # 메타데이터 및 사용자 정의 정보
    created_at: str = ""
    updated_at: str = ""
    version: str = "1.0.0"
    title: str = "기본 설정"  # 사용자 정의 제목
    description: str = "Enhanced Shopping Agent 기본 프롬프트 설정"
    author: str = "시스템"  # 작성자
    tags: List[str] = None  # 태그 목록
    is_active: bool = False  # 현재 활성 여부
    
    def __post_init__(self):
        """초기화 후 타임스탬프 설정"""
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptConfig':
        """딕셔너리에서 객체 생성"""
        return cls(**data)
    
    def save_to_file(self, file_path: str):
        """파일에 저장"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'PromptConfig':
        """파일에서 로드"""
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return cls.from_dict(data)
        else:
            # 파일이 없으면 기본값 반환
            return cls()


class PromptManager:
    """프롬프트 관리 클래스"""
    
    def __init__(self, config_dir: str = None):
        """
        프롬프트 매니저 초기화
        
        Args:
            config_dir: 프롬프트 설정 파일들이 저장될 디렉토리
        """
        if config_dir is None:
            # 기본 경로: src/config/prompts/
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.config_dir = os.path.join(current_dir, "prompts")
        else:
            self.config_dir = config_dir
            
        # 디렉토리 생성
        os.makedirs(self.config_dir, exist_ok=True)
        
        # 기본 설정 파일 경로
        self.default_config_path = os.path.join(self.config_dir, "default.json")
        
        # 현재 활성 설정 확인 및 초기화
        self._ensure_default_active()
        
        # 활성 설정을 current_config로 로드
        active_config_name = self._get_active_config_name()
        self.current_config = self.load_config(active_config_name)
    
    def _ensure_default_active(self):
        """
        활성 설정이 없는 경우 기본 설정을 활성화합니다.
        """
        # 활성 설정이 있는지 확인
        has_active = False
        if os.path.exists(self.config_dir):
            for file in os.listdir(self.config_dir):
                if file.endswith('.json'):
                    file_path = os.path.join(self.config_dir, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        if data.get('is_active', False):
                            has_active = True
                            break
                    except Exception:
                        continue
        
        # 활성 설정이 없으면 default.json을 활성화
        if not has_active:
            default_config = self.load_config("default")
            default_config.is_active = True
            self.save_config(default_config, "default")
    
    def _get_active_config_name(self) -> str:
        """
        현재 활성화된 설정의 이름을 반환합니다.
        
        Returns:
            str: 활성 설정 이름 (활성 설정이 없으면 "default")
        """
        if os.path.exists(self.config_dir):
            for file in os.listdir(self.config_dir):
                if file.endswith('.json'):
                    file_path = os.path.join(self.config_dir, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        if data.get('is_active', False):
                            return file[:-5]  # .json 확장자 제거
                    except Exception:
                        continue
        return "default"
    
    def load_config(self, config_name: str = "default") -> PromptConfig:
        """
        지정된 이름의 프롬프트 설정을 로드합니다.
        
        Args:
            config_name: 로드할 설정 이름
            
        Returns:
            PromptConfig: 로드된 프롬프트 설정
        """
        config_path = os.path.join(self.config_dir, f"{config_name}.json")
        return PromptConfig.load_from_file(config_path)
    
    def save_config(self, config: PromptConfig, config_name: str = "default"):
        """
        프롬프트 설정을 파일에 저장합니다.
        
        Args:
            config: 저장할 프롬프트 설정
            config_name: 저장할 설정 이름
        """
        config_path = os.path.join(self.config_dir, f"{config_name}.json")
        config.save_to_file(config_path)
    
    def get_analysis_prompt(self, user_query: str) -> str:
        """
        사용자 쿼리를 포함한 분석 프롬프트를 반환합니다.
        
        Args:
            user_query: 분석할 사용자 질문
            
        Returns:
            str: 완성된 분석 프롬프트
        """
        return self.current_config.analysis_prompt_template.format(user_query=user_query)
    
    def get_system_prompt(self, context: str) -> str:
        """
        컨텍스트를 포함한 시스템 프롬프트를 반환합니다.
        
        Args:
            context: 수집된 컨텍스트 정보
            
        Returns:
            str: 완성된 시스템 프롬프트
        """
        return self.current_config.system_prompt_template.format(context=context)
    
    def update_prompts(self, analysis_prompt: str = None, system_prompt: str = None):
        """
        현재 활성 설정의 프롬프트를 업데이트합니다.
        
        Args:
            analysis_prompt: 새로운 분석 프롬프트 (선택사항)
            system_prompt: 새로운 시스템 프롬프트 (선택사항)
        """
        if analysis_prompt is not None:
            self.current_config.analysis_prompt_template = analysis_prompt
        
        if system_prompt is not None:
            self.current_config.system_prompt_template = system_prompt
        
        # 업데이트 시간 갱신
        self.current_config.updated_at = datetime.now().isoformat()
    
    def get_available_configs(self) -> list:
        """
        사용 가능한 설정 파일 목록을 반환합니다.
        
        Returns:
            list: 설정 파일 이름 목록 (확장자 제외)
        """
        if not os.path.exists(self.config_dir):
            return []
        
        config_files = []
        for file in os.listdir(self.config_dir):
            if file.endswith('.json'):
                config_files.append(file[:-5])  # .json 확장자 제거
        
        return sorted(config_files)
    
    def save_with_metadata(self, config: PromptConfig, title: str, description: str = "", author: str = "사용자", tags: List[str] = None) -> str:
        """
        사용자 정의 메타데이터와 함께 프롬프트 설정을 저장합니다.
        
        Args:
            config: 저장할 프롬프트 설정
            title: 사용자 정의 제목
            description: 설명
            author: 작성자
            tags: 태그 리스트
            
        Returns:
            str: 생성된 파일 이름
        """
        # 메타데이터 업데이트
        config.title = title
        config.description = description
        config.author = author
        config.tags = tags or []
        config.updated_at = datetime.now().isoformat()
        
        # 파일명 생성 (제목 기반 + 고유성 보장)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_title = safe_title.replace(' ', '_')[:20]  # 길이 제한
        
        # 파일명 중복 방지
        base_filename = f"{safe_title}_{timestamp}"
        filename = base_filename
        counter = 1
        
        while os.path.exists(os.path.join(self.config_dir, f"{filename}.json")):
            filename = f"{base_filename}_{counter}"
            counter += 1
        
        self.save_config(config, filename)
        return filename
    
    def get_all_configs(self) -> List[Dict[str, Any]]:
        """
        모든 저장된 설정 목록을 반환합니다.
        
        Returns:
            List[Dict]: 설정 정보 리스트 (메타데이터 포함)
        """
        if not os.path.exists(self.config_dir):
            return []
        
        configs = []
        for file in os.listdir(self.config_dir):
            if file.endswith('.json'):
                config_name = file[:-5]  # .json 제거
                try:
                    config = self.load_config(config_name)
                    configs.append({
                        'filename': config_name,
                        'title': config.title,
                        'description': config.description,
                        'author': config.author,
                        'version': config.version,
                        'created_at': config.created_at,
                        'updated_at': config.updated_at,
                        'tags': config.tags,
                        'is_active': config.is_active,
                        'is_default': config_name == 'default'
                    })
                except Exception as e:
                    # 손상된 파일은 건너뛰기
                    continue
        
        # 수정일시 순으로 정렬 (최신순)
        configs.sort(key=lambda x: x['updated_at'], reverse=True)
        return configs
    
    def set_active_config(self, config_name: str):
        """
        지정된 설정을 활성 설정으로 설정합니다.
        
        Args:
            config_name: 활성화할 설정 이름
        """
        # 1단계: 모든 설정 파일의 is_active를 False로 설정
        if os.path.exists(self.config_dir):
            for file in os.listdir(self.config_dir):
                if file.endswith('.json'):
                    file_path = os.path.join(self.config_dir, file)
                    try:
                        # 각 파일을 개별적으로 로드하고 수정
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # is_active를 False로 설정
                        data['is_active'] = False
                        
                        # 파일에 다시 저장
                        with open(file_path, 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                    except Exception as e:
                        # 손상된 파일은 건너뛰기
                        continue
        
        # 2단계: 선택된 설정만 활성화
        target_config = self.load_config(config_name)
        target_config.is_active = True
        self.save_config(target_config, config_name)
        
        # 3단계: current_config 업데이트
        self.current_config = target_config
        
        # 4단계: default.json 업데이트 (선택된 설정이 default인 경우에만 활성화)
        default_config = PromptConfig(
            analysis_prompt_template=target_config.analysis_prompt_template,
            system_prompt_template=target_config.system_prompt_template,
            title=target_config.title,
            description=target_config.description,
            author=target_config.author,
            tags=target_config.tags[:],  # 복사본 생성
            version=target_config.version,
            created_at=target_config.created_at,
            updated_at=target_config.updated_at,
            is_active=(config_name == "default")  # default 설정을 선택한 경우에만 활성화
        )
        self.save_config(default_config, "default")
    
    def compare_configs(self, config1_name: str, config2_name: str) -> Dict[str, Any]:
        """
        두 설정을 비교합니다.
        
        Args:
            config1_name: 첫 번째 설정 이름
            config2_name: 두 번째 설정 이름
            
        Returns:
            Dict: 비교 결과
        """
        config1 = self.load_config(config1_name)
        config2 = self.load_config(config2_name)
        
        # 길이 차이 계산
        analysis_diff = len(config1.analysis_prompt_template) - len(config2.analysis_prompt_template)
        system_diff = len(config1.system_prompt_template) - len(config2.system_prompt_template)
        
        return {
            'config1': {
                'name': config1_name,
                'title': config1.title,
                'analysis_length': len(config1.analysis_prompt_template),
                'system_length': len(config1.system_prompt_template),
                'updated_at': config1.updated_at
            },
            'config2': {
                'name': config2_name,
                'title': config2.title,
                'analysis_length': len(config2.analysis_prompt_template),
                'system_length': len(config2.system_prompt_template),
                'updated_at': config2.updated_at
            },
            'differences': {
                'analysis_prompt_diff': analysis_diff,
                'system_prompt_diff': system_diff,
                'analysis_same': config1.analysis_prompt_template == config2.analysis_prompt_template,
                'system_same': config1.system_prompt_template == config2.system_prompt_template
            }
        }
    
    def delete_config(self, config_name: str) -> bool:
        """
        설정을 삭제합니다.
        
        Args:
            config_name: 삭제할 설정 이름
            
        Returns:
            bool: 삭제 성공 여부
        """
        if config_name == "default":
            return False  # 기본 설정은 삭제 불가
        
        config_path = os.path.join(self.config_dir, f"{config_name}.json")
        if os.path.exists(config_path):
            os.remove(config_path)
            return True
        return False
    
    def create_backup(self, config_name: str = "default") -> str:
        """
        현재 설정의 백업을 생성합니다.
        
        Args:
            config_name: 백업할 설정 이름
            
        Returns:
            str: 생성된 백업 파일 이름
        """
        config = self.load_config(config_name)
        
        # 백업용 메타데이터 설정
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_title = f"{config.title}_백업_{timestamp}"
        
        return self.save_with_metadata(
            config, 
            title=backup_title,
            description=f"{config.description} (백업)",
            author=config.author,
            tags=config.tags + ["백업"]
        )


# 전역 프롬프트 매니저 인스턴스
prompt_manager = PromptManager()


def get_prompt_manager() -> PromptManager:
    """프롬프트 매니저 인스턴스를 반환합니다."""
    return prompt_manager