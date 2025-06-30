import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.tools import tool

@tool
def vector_store_retriever_tool(query: str) -> str:
    """
    FAISS 벡터 스토어에서 사용자의 질문과 관련된 정보를 검색합니다.
    주어진 쿼리에 가장 관련성이 높은 문서를 반환합니다.
    """
    vector_store_path = "vector_store/faiss_index"
    if not os.path.exists(vector_store_path):
        return "벡터 스토어가 초기화되지 않았습니다. 먼저 데이터를 크롤링하고 인덱싱해야 합니다."

    try:
        embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever()
        docs = retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        return f"벡터 스토어 검색 중 오류 발생: {e}"
