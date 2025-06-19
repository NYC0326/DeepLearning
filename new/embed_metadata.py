import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import pandas as pd
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# 환경변수 설정 (혹시 안 되어 있으면)
os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')

# 파일 경로
INPUT_PATH = "metadata/abstracts.csv"
VECTORSTORE_PATH = "vectorstores/abstracts_faiss"

# 구글 임베딩 모델 로드
embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# CSV 로드 및 Document 리스트 생성
df = pd.read_csv(INPUT_PATH)
documents = []
for _, row in df.iterrows():
    content = f"Title: {row['title']}\nAbstract: {row['abstract']}"
    metadata = {"source": row['id']}
    documents.append(Document(page_content=content, metadata=metadata))

print(f"🔍 {len(documents)}개 논문 abstract 임베딩 중...")

# FAISS 인덱싱
vectorstore = FAISS.from_documents(documents, embedding_model)
os.makedirs("vectorstores", exist_ok=True)
vectorstore.save_local(VECTORSTORE_PATH)

print(f"✅ 저장 완료: {VECTORSTORE_PATH}")
