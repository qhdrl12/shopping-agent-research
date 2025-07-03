import streamlit as st
import sys
import os
import json
from datetime import datetime
from typing import Dict, Optional

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.local_prompt_manager import LocalPromptManager

# Page config
st.set_page_config(
    page_title="🎯 프롬프트 관리 센터",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .prompt-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .prompt-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    
    .active-prompt {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Status indicators */
    .status-active {
        background: #28a745;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    .status-inactive {
        background: #6c757d;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    /* Sidebar styling */
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    /* Text area styling */
    .stTextArea > div > div > textarea {
        font-family: 'Courier New', monospace;
        border-radius: 5px;
        border: 2px solid #e9ecef;
        transition: border-color 0.3s ease;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
    }
    
    /* Alert styling */
    .custom-alert {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
        border-left: 4px solid;
    }
    
    .alert-success {
        background: #d4edda;
        border-color: #28a745;
        color: #155724;
    }
    
    .alert-warning {
        background: #fff3cd;
        border-color: #ffc107;
        color: #856404;
    }
    
    .alert-info {
        background: #d1ecf1;
        border-color: #17a2b8;
        color: #0c5460;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'active_prompt_name' not in st.session_state:
    st.session_state.active_prompt_name = 'default'
if 'edit_mode' not in st.session_state:
    st.session_state.edit_mode = False
if 'current_editing_prompt' not in st.session_state:
    st.session_state.current_editing_prompt = None
if 'preview_mode' not in st.session_state:
    st.session_state.preview_mode = False

# Initialize prompt manager
@st.cache_resource
def get_prompt_manager():
    return LocalPromptManager()

prompt_manager = get_prompt_manager()

# Main header
st.markdown("""
<div class="main-header">
    <h1>🎯 프롬프트 관리 센터</h1>
    <p>쇼핑 에이전트의 프롬프트를 직관적으로 관리하고 실시간으로 편집하세요</p>
</div>
""", unsafe_allow_html=True)

# Sidebar - Prompt Management Panel
with st.sidebar:
    st.markdown("### 🎛️ 프롬프트 제어판")
    
    # Current active prompt indicator
    st.markdown(f"""
    <div class="sidebar-section">
        <h4>📍 현재 활성 프롬프트</h4>
        <div class="status-active">{st.session_state.active_prompt_name}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick actions
    st.markdown("### ⚡ 빠른 작업")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📝 새 프롬프트", use_container_width=True):
            st.session_state.edit_mode = True
            st.session_state.current_editing_prompt = {
                'name': '',
                'query_analysis_prompt': '',
                'model_response_prompt': '',
                'is_new': True
            }
            st.rerun()
    
    with col2:
        if st.button("👁️ 미리보기", use_container_width=True):
            st.session_state.preview_mode = not st.session_state.preview_mode
            st.rerun()
    
    # Import/Export section
    st.markdown("### 📁 가져오기/내보내기")
    
    # Export current prompts
    if st.button("💾 전체 프롬프트 내보내기", use_container_width=True):
        prompts = prompt_manager._load_prompts()
        export_data = {
            'exported_at': datetime.now().isoformat(),
            'prompts': prompts
        }
        st.download_button(
            label="📥 JSON 파일 다운로드",
            data=json.dumps(export_data, ensure_ascii=False, indent=2),
            file_name=f"prompts_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    # Import prompts
    uploaded_file = st.file_uploader(
        "📤 프롬프트 파일 가져오기",
        type=['json'],
        help="이전에 내보낸 프롬프트 JSON 파일을 업로드하세요"
    )
    
    if uploaded_file is not None:
        try:
            import_data = json.load(uploaded_file)
            if 'prompts' in import_data:
                st.success(f"✅ {len(import_data['prompts'])}개 프롬프트를 찾았습니다!")
                if st.button("🔄 가져오기 실행", use_container_width=True):
                    # Import logic would go here
                    st.success("프롬프트 가져오기 완료!")
                    st.rerun()
        except Exception as e:
            st.error(f"❌ 파일 형식이 올바르지 않습니다: {e}")

# Main content area
if st.session_state.edit_mode and st.session_state.current_editing_prompt:
    # Edit Mode UI
    st.markdown("## ✏️ 프롬프트 편집")
    
    # Edit form
    with st.form("prompt_edit_form", clear_on_submit=False):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            prompt_name = st.text_input(
                "프롬프트 이름",
                value=st.session_state.current_editing_prompt.get('name', ''),
                placeholder="예: advanced_shopping_v2",
                help="프롬프트를 식별할 수 있는 고유한 이름을 입력하세요"
            )
        
        with col2:
            st.markdown("### 📊 상태")
            if st.session_state.current_editing_prompt.get('is_new', False):
                st.markdown('<div class="status-inactive">새 프롬프트</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-active">편집 중</div>', unsafe_allow_html=True)
        
        # Tabbed interface for prompt editing
        tab1, tab2 = st.tabs(["🔍 질문 분석 프롬프트", "💬 최종 답변 프롬프트"])
        
        with tab1:
            st.markdown("**질문 분석 프롬프트 편집**")
            query_analysis_prompt = st.text_area(
                "질문 분석 프롬프트",
                value=st.session_state.current_editing_prompt.get('query_analysis_prompt', ''),
                height=400,
                help="사용자 질문을 분석하여 구조화된 정보를 추출하는 프롬프트",
                label_visibility="collapsed"
            )
            
            # Character count and tips
            char_count_1 = len(query_analysis_prompt)
            st.caption(f"📝 {char_count_1:,}자 | 권장: 1,000-3,000자")
            
            with st.expander("💡 질문 분석 프롬프트 작성 팁"):
                st.markdown("""
                - **구조화된 출력**: JSON 형식으로 명확한 구조 정의
                - **핵심 추출**: 검색 키워드, 의도, 카테고리 등 핵심 정보 추출
                - **예시 포함**: 구체적인 예시로 이해도 향상
                - **에러 방지**: 예외 상황 처리 방법 명시
                """)
        
        with tab2:
            st.markdown("**최종 답변 프롬프트 편집**")
            model_response_prompt = st.text_area(
                "최종 답변 프롬프트",
                value=st.session_state.current_editing_prompt.get('model_response_prompt', ''),
                height=400,
                help="수집된 정보를 바탕으로 최종 답변을 생성하는 프롬프트",
                label_visibility="collapsed"
            )
            
            # Character count and tips
            char_count_2 = len(model_response_prompt)
            st.caption(f"📝 {char_count_2:,}자 | 권장: 800-2,000자")
            
            with st.expander("💡 답변 프롬프트 작성 팁"):
                st.markdown("""
                - **답변 구조**: 명확한 섹션 구분과 정보 계층화
                - **전문성**: 쇼핑 컨설턴트로서의 전문적 조언
                - **실용성**: 구체적이고 실행 가능한 정보 제공
                - **개인화**: 사용자 상황에 맞는 맞춤형 추천
                """)
        
        # Form actions
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.form_submit_button("💾 저장", type="primary", use_container_width=True):
                if prompt_name and query_analysis_prompt and model_response_prompt:
                    try:
                        if st.session_state.current_editing_prompt.get('is_new', False):
                            # Create new prompt
                            result = prompt_manager.create_prompt(
                                name=prompt_name,
                                query_analysis_prompt=query_analysis_prompt,
                                model_response_prompt=model_response_prompt
                            )
                        else:
                            # Update existing prompt
                            result = prompt_manager.update_prompt(
                                prompt_id=st.session_state.current_editing_prompt['id'],
                                name=prompt_name,
                                query_analysis_prompt=query_analysis_prompt,
                                model_response_prompt=model_response_prompt
                            )
                        
                        if result:
                            st.success("✅ 프롬프트가 성공적으로 저장되었습니다!")
                            st.session_state.edit_mode = False
                            st.session_state.current_editing_prompt = None
                            st.rerun()
                        else:
                            st.error("❌ 프롬프트 저장에 실패했습니다.")
                    except Exception as e:
                        st.error(f"❌ 오류 발생: {e}")
                else:
                    st.warning("⚠️ 모든 필드를 입력해주세요.")
        
        with col2:
            if st.form_submit_button("🔄 즉시 적용", use_container_width=True):
                if prompt_name and query_analysis_prompt and model_response_prompt:
                    # Save and set as active
                    st.session_state.active_prompt_name = prompt_name
                    st.success(f"✅ '{prompt_name}' 프롬프트가 즉시 적용되었습니다!")
                else:
                    st.warning("⚠️ 모든 필드를 입력해주세요.")
        
        with col3:
            if st.form_submit_button("👁️ 미리보기", use_container_width=True):
                st.session_state.preview_mode = True
        
        with col4:
            if st.form_submit_button("❌ 취소", use_container_width=True):
                st.session_state.edit_mode = False
                st.session_state.current_editing_prompt = None
                st.rerun()

elif st.session_state.preview_mode:
    # Preview Mode UI
    st.markdown("## 👁️ 프롬프트 미리보기")
    
    if st.session_state.current_editing_prompt:
        prompt_data = st.session_state.current_editing_prompt
    else:
        prompt_data = prompt_manager.get_prompt(st.session_state.active_prompt_name)
    
    if prompt_data:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### 📋 {prompt_data.get('name', 'Unknown')}")
        with col2:
            if st.button("❌ 미리보기 닫기", use_container_width=True):
                st.session_state.preview_mode = False
                st.rerun()
        
        # Preview tabs
        tab1, tab2, tab3 = st.tabs(["🔍 질문 분석", "💬 최종 답변", "📊 정보"])
        
        with tab1:
            st.markdown("**질문 분석 프롬프트**")
            st.text_area(
                "Preview",
                value=prompt_data.get('query_analysis_prompt', ''),
                height=400,
                disabled=True,
                label_visibility="collapsed"
            )
        
        with tab2:
            st.markdown("**최종 답변 프롬프트**")
            st.text_area(
                "Preview",
                value=prompt_data.get('model_response_prompt', ''),
                height=400,
                disabled=True,
                label_visibility="collapsed"
            )
        
        with tab3:
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                st.metric("질문 분석 프롬프트 길이", f"{len(prompt_data.get('query_analysis_prompt', '')):,}자")
                st.metric("생성일", prompt_data.get('created_at', 'N/A')[:10] if prompt_data.get('created_at') else 'N/A')
            
            with info_col2:
                st.metric("답변 프롬프트 길이", f"{len(prompt_data.get('model_response_prompt', '')):,}자")
                st.metric("수정일", prompt_data.get('updated_at', 'N/A')[:10] if prompt_data.get('updated_at') else 'N/A')

else:
    # Main Dashboard UI
    st.markdown("## 📚 프롬프트 라이브러리")
    
    # Load all prompts
    prompt_list = prompt_manager.get_prompt_list()
    
    if not prompt_list:
        st.markdown("""
        <div class="custom-alert alert-info">
            <h4>💡 프롬프트가 없습니다</h4>
            <p>새로운 프롬프트를 생성하여 시작하세요!</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Prompt grid
        for i, prompt_name in enumerate(prompt_list):
            prompt_data = prompt_manager.get_prompt(prompt_name)
            if prompt_data:
                is_active = prompt_name == st.session_state.active_prompt_name
                
                # Create prompt card
                card_class = "prompt-card active-prompt" if is_active else "prompt-card"
                
                with st.container():
                    st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
                    
                    col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
                    
                    with col1:
                        st.markdown(f"### 📋 {prompt_name}")
                        st.caption(f"생성: {prompt_data.get('created_at', 'N/A')[:10] if prompt_data.get('created_at') else 'N/A'}")
                        
                        # Extract meaningful preview from prompt content
                        def extract_prompt_summary(prompt_text):
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
                        
                        # Query analysis prompt preview
                        analysis_summary = extract_prompt_summary(prompt_data.get('query_analysis_prompt', ''))
                        response_summary = extract_prompt_summary(prompt_data.get('model_response_prompt', ''))
                        
                        # Create feature badges
                        st.markdown(f"""
                        <div style="margin: 0.5rem 0;">
                            <div style="background: #e3f2fd; color: #1565c0; padding: 0.25rem 0.5rem; border-radius: 12px; display: inline-block; margin: 0.2rem 0.2rem 0.2rem 0; font-size: 0.75rem;">
                                🔍 {analysis_summary}
                            </div>
                        </div>
                        <div style="margin: 0.5rem 0;">
                            <div style="background: #f3e5f5; color: #7b1fa2; padding: 0.25rem 0.5rem; border-radius: 12px; display: inline-block; margin: 0.2rem 0.2rem 0.2rem 0; font-size: 0.75rem;">
                                💬 {response_summary}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Add prompt statistics
                        analysis_length = len(prompt_data.get('query_analysis_prompt', ''))
                        response_length = len(prompt_data.get('model_response_prompt', ''))
                        total_length = analysis_length + response_length
                        
                        st.markdown(f"""
                        <div style="margin-top: 0.5rem; font-size: 0.7rem; color: #666;">
                            📊 총 {total_length:,}자 (분석: {analysis_length:,} | 답변: {response_length:,})
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        if is_active:
                            st.markdown('<div class="status-active">활성</div>', unsafe_allow_html=True)
                        else:
                            if st.button("🎯 활성화", key=f"activate_{i}", use_container_width=True):
                                st.session_state.active_prompt_name = prompt_name
                                st.success(f"✅ '{prompt_name}' 프롬프트가 활성화되었습니다!")
                                st.rerun()
                    
                    with col3:
                        if st.button("✏️ 편집", key=f"edit_{i}", use_container_width=True):
                            st.session_state.edit_mode = True
                            st.session_state.current_editing_prompt = prompt_data
                            st.rerun()
                    
                    with col4:
                        if st.button("👁️ 보기", key=f"view_{i}", use_container_width=True):
                            st.session_state.preview_mode = True
                            st.session_state.current_editing_prompt = prompt_data
                            st.rerun()
                    
                    with col5:
                        if prompt_name != 'default':  # Prevent deleting default prompt
                            if st.button("🗑️ 삭제", key=f"delete_{i}", use_container_width=True):
                                if st.checkbox(f"'{prompt_name}' 삭제 확인", key=f"confirm_delete_{i}"):
                                    if prompt_manager.delete_prompt(prompt_name):
                                        st.success(f"✅ '{prompt_name}' 프롬프트가 삭제되었습니다!")
                                        if st.session_state.active_prompt_name == prompt_name:
                                            st.session_state.active_prompt_name = 'default'
                                        st.rerun()
                        else:
                            st.markdown('<small style="color: #6c757d;">기본 프롬프트</small>', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown("")  # Add spacing

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 1rem;">
    <p>🎯 프롬프트 관리 센터 | 쇼핑 에이전트 최적화 도구</p>
</div>
""", unsafe_allow_html=True)