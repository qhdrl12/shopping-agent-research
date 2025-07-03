import streamlit as st
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.local_prompt_manager import LocalPromptManager

st.set_page_config(layout="wide")

st.title("프롬프트 편집기")

# 로컬 프롬프트 매니저 초기화
try:
    prompt_manager = LocalPromptManager()
    st.sidebar.success("로컬 프롬프트 매니저가 초기화되었습니다.")
except Exception as e:
    st.error(f"로컬 프롬프트 매니저 초기화 오류: {e}")
    st.stop()

# --- 사이드바 --- #
st.sidebar.header("프롬프트 선택")
prompt_names = prompt_manager.get_prompt_list()
if not prompt_names:
    st.sidebar.warning("로컬 파일에서 프롬프트를 찾을 수 없습니다. 먼저 프롬프트를 생성해주세요.")

selected_prompt_name = st.sidebar.selectbox(
    "편집할 프롬프트를 선택하세요",
    options=prompt_names,
    index=0 if prompt_names else -1,
    key="selected_prompt"
)

# --- 메인 페이지 --- #
if selected_prompt_name:
    prompt_data = prompt_manager.get_prompt(selected_prompt_name)

    if prompt_data:
        st.header(f"`{prompt_data['name']}` 프롬프트 편집")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Query Analysis Prompt")
            query_analysis_prompt = st.text_area(
                "Query Analysis Prompt",
                value=prompt_data['query_analysis_prompt'],
                height=400,
                label_visibility="collapsed"
            )

        with col2:
            st.subheader("Model Response Prompt")
            model_response_prompt = st.text_area(
                "Model Response Prompt",
                value=prompt_data['model_response_prompt'],
                height=400,
                label_visibility="collapsed"
            )

        # --- 버튼 --- #
        st.markdown("---")
        c1, c2, c3, c4 = st.columns([1, 1, 1, 5])

        if c1.button("💾 저장", use_container_width=True):
            prompt_manager.update_prompt(
                prompt_id=prompt_data['id'],
                name=prompt_data['name'],
                query_analysis_prompt=query_analysis_prompt,
                model_response_prompt=model_response_prompt
            )
            st.success(f"'{prompt_data['name']}' 프롬프트가 성공적으로 업데이트되었습니다.")
            st.rerun()

        if c2.button("🗑️ 삭제", use_container_width=True):
            if st.checkbox(f"'{prompt_data['name']}' 프롬프트를 정말로 삭제하시겠습니까?", key="delete_confirm"):
                prompt_manager.delete_prompt(prompt_data['name'])
                st.success(f"'{prompt_data['name']}' 프롬프트가 삭제되었습니다.")
                st.rerun()

# --- 새 프롬프트 생성 --- #
st.sidebar.markdown("--- ")
st.sidebar.header("새 프롬프트 생성")
new_prompt_name = st.sidebar.text_input("새 프롬프트 이름")

if st.sidebar.button("✨ 새로 만들기", use_container_width=True):
    if new_prompt_name:
        prompt_manager.create_prompt(
            name=new_prompt_name,
            query_analysis_prompt="새로운 Query Analysis 프롬프트를 여기에 입력하세요.",
            model_response_prompt="새로운 Model Response 프롬프트를 여기에 입력하세요."
        )
        st.sidebar.success(f"'{new_prompt_name}' 프롬프트가 생성되었습니다.")
        # Immediately select the new prompt
        st.session_state.selected_prompt = new_prompt_name
        st.rerun()

    else:
        st.sidebar.error("새 프롬프트의 이름을 입력해주세요.")
