import streamlit as st
from rag import answer_question

st.set_page_config(page_title="Chatbot PDF (RAG)", page_icon="📄")

st.title("📄🤖 Chatbot baseado em PDFs (RAG)")
st.caption("Responde com base nos PDFs indexados em `inputs/pdfs/`.")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("Faça uma pergunta sobre os PDFs:")

if st.button("Perguntar") and question.strip():
    result = answer_question(question)

    st.session_state.history.append(("Você", question))
    st.session_state.history.append(("Bot", result["answer"]))

    with st.expander("🔎 Fontes recuperadas"):
        for s in result["sources"]:
            st.write(f"- {s['source']} (pág. {s['page']})")

st.divider()

for who, msg in st.session_state.history:
    if who == "Você":
        st.markdown(f"**🧑 {who}:** {msg}")
    else:
        st.markdown(f"**🤖 {who}:** {msg}")

