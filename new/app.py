import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import requests
import re
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt_tab')
from tqdm import tqdm
from langchain_core.documents import Document

# ====== 설정 ======
os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')
PDF_DIR = "pdfs"
VECTOR_DIR = "vectorstores/on_demand"
THRESHOLD = 0.75

# ====== 모델 준비 ======
embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.4, max_output_tokens=1536)

# ====== 유틸 함수 ======
def clean_text(text):
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'\n', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def semantic_chunk(text, threshold=THRESHOLD):
    sentences = sent_tokenize(text)
    if len(sentences) < 2:
        return [text]
    embeddings = embedding_model.embed_documents(sentences)
    chunks, chunk = [], [sentences[0]]
    for i in range(1, len(sentences)):
        sim = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0]
        if sim < threshold:
            chunks.append(" ".join(chunk))
            chunk = []
        chunk.append(sentences[i])
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

def process_pdf(file_path, paper_title=""):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    raw_text = "\n".join([d.page_content for d in docs])
    cleaned = clean_text(raw_text)
    chunks = semantic_chunk(cleaned)
    if paper_title:
        chunks.insert(0, f"논문 제목: {paper_title}")
    return chunks

# ====== Streamlit UI ======
st.set_page_config(page_title="논문 검색 및 질의응답", layout="wide")
st.title("📚 논문 검색 + PDF RAG")

query = st.text_input("🔍 찾고 싶은 논문 주제를 입력하세요:")

# ---- Abstract 검색 ----
if query:
    st.subheader("🔎 관련 논문 검색 결과")
    abstracts_store = FAISS.load_local("vectorstores/abstracts_faiss", embedding_model, allow_dangerous_deserialization=True)
    docs = abstracts_store.similarity_search(query, k=5)

    for i, doc in enumerate(docs):
        st.markdown(f"**{i+1}. Title**: {doc.page_content.split('\n')[0].replace('Title: ', '')}")
        st.markdown(f"*Abstract*: {doc.page_content.split('\n')[1].replace('Abstract: ', '')}")
        st.markdown(f"`arXiv ID`: {doc.metadata['source']}")
        st.markdown("---")

# ---- PDF ID 입력 및 자동 처리 ----
st.subheader("📄 arXiv 논문 ID 입력 → PDF 처리")
id_input = st.text_input("arXiv ID를 입력하세요 (예: 1404.0736):")
process_button = st.button("논문 처리하기")

if process_button and id_input:
    with st.spinner("PDF 다운로드 중..."):
        pdf_url = f"https://arxiv.org/pdf/{id_input}.pdf"
        response = requests.get(pdf_url)
        if response.status_code != 200:
            st.error("❌ PDF 다운로드 실패!")
        else:
            os.makedirs(PDF_DIR, exist_ok=True)
            file_path = os.path.join(PDF_DIR, f"{id_input.replace('/', '_')}.pdf")
            with open(file_path, "wb") as f:
                f.write(response.content)

            # metadata에서 논문 제목 가져오기
            title = ""
            try:
                meta = pd.read_csv("metadata/abstracts.csv")
                row = meta[meta["id"] == id_input]
                if not row.empty:
                    title = row["title"].values[0]
            except:
                pass

            with st.spinner("청킹 및 벡터 DB 저장 중..."):
                chunks = process_pdf(file_path, paper_title=title)
                documents = [Document(page_content=chunk, metadata={"source": id_input}) for chunk in chunks]
                faiss_store = FAISS.from_documents(documents, embedding_model)
                faiss_store.save_local(VECTOR_DIR)
                st.success("✅ 처리 완료! 질문해보세요.")

# ---- 질문 UI ----
st.subheader("🤖 논문 내용 질문하기")
user_q = st.text_input("질문을 입력하세요:")
if user_q:
    try:
        local_store = FAISS.load_local(VECTOR_DIR, embedding_model, allow_dangerous_deserialization=True)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=local_store.as_retriever())
        response = qa_chain.invoke(user_q)
        st.markdown(response['result'])
    except:
        st.error("벡터 DB가 아직 없어요! 먼저 논문 PDF를 입력해 주세요.")
