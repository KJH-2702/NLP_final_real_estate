"""
Streamlit 데모 앱
------------------------------------------------
실행: streamlit run app.py
계약서 PDF를 업로드하면 조항별 위험도 분류 + 근거 법조문 + AI 설명을 보여준다.
"""

import tempfile

import streamlit as st

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from pipeline import analyze_contract_pdf

st.set_page_config(page_title="임대차 계약서 특약 검토 AI", layout="wide")

st.title("임대차 계약서 특약사항 검토 AI")
st.caption(
    "계약서 PDF를 업로드하면 조항을 정상/주의/위험으로 분류하고, "
    "관련 법 조문을 근거로 삼아 설명해 드립니다. "
    "본 결과는 참고용이며 법적 효력을 갖는 자문이 아닙니다."
)


@st.cache_resource(show_spinner=False)
def get_retriever():
    from rag.retriever import LawRetriever

    return LawRetriever()


@st.cache_resource(show_spinner=False)
def get_explainer():
    from generation.explainer import GeminiExplainer

    return GeminiExplainer()


with st.sidebar:
    st.header("설정")
    use_explanation = st.checkbox("Gemini로 설명 생성", value=True)
    st.caption("GOOGLE_API_KEY 환경변수가 설정되어 있어야 합니다.")

uploaded = st.file_uploader("계약서 PDF 업로드", type=["pdf"])

if uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    retriever = None
    explainer = None

    try:
        with st.spinner("법률 검색기 로딩 중..."):
            retriever = get_retriever()
    except Exception as e:
        st.warning(f"RAG 검색기를 불러오지 못했습니다 (근거 조문 없이 진행): {e}")

    if use_explanation:
        try:
            with st.spinner("Gemini 연결 중..."):
                explainer = get_explainer()
        except Exception as e:
            st.warning(f"Gemini 설명 생성을 사용할 수 없습니다 (설명 없이 진행): {e}")

    with st.spinner("계약서 분석 중..."):
        result = analyze_contract_pdf(
            tmp_path,
            generate_explanations=use_explanation,
            retriever=retriever,
            explainer=explainer,
        )

    summary = result["summary"]
    col1, col2, col3 = st.columns(3)
    col1.metric("정상", summary["정상"])
    col2.metric("주의", summary["주의"])
    col3.metric("위험", summary["위험"])

    st.divider()

    icon_map = {"정상": "🟢", "주의": "🟠", "위험": "🔴"}

    for entry in result["clauses"]:
        label = entry["label"]
        icon = icon_map[label]
        preview = entry["clause"][:45].replace("\n", " ")
        with st.expander(f"{icon} [{label}] ({entry['source']}) {preview}..."):
            st.write(f"**조항 원문**")
            st.write(entry["clause"])
            st.write(f"**분류**: {label}  (신뢰도 {entry['confidence'] * 100:.1f}%)")

            if entry["references"]:
                st.write("**관련 법 조문**")
                for ref in entry["references"]:
                    st.write(f"- {ref['title']} (유사도 {ref['score']})")
                    st.caption(ref["text"])

            if entry["explanation"]:
                st.info(entry["explanation"])
else:
    st.info("좌측에서 옵션을 확인한 뒤, 계약서 PDF 파일을 업로드해주세요.")
