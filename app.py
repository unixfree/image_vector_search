import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
from PIL import Image
import psycopg2
import io

# --- 1. 초기 설정 및 모델 로드 ---
st.set_page_config(layout="wide", page_title="AI 이미지 검색 엔진")

@st.cache_resource
def load_model():
    # OpenAI의 CLIP 모델 기반 임베딩 모델
    return SentenceTransformer('clip-vit-base-patch32')

model = load_model()

# DB 연결 함수
def get_db_connection():
    return psycopg2.connect(
        dbname="your_db", 
        user="your_user", 
        password="your_password", 
        host="localhost"
    )

# --- 2. 비즈니스 로직 함수 ---
def get_embedding(image):
    """PIL Image 객체를 받아 벡터로 변환"""
    embedding = model.encode(image)
    return embedding.tolist()

# --- 3. UI 레이아웃 ---
st.title("🖼️ Vector DB 기반 이미지 검색 엔진 (pgvector)")
st.markdown("---")

left_col, right_col = st.columns(2)

# --- 좌측: 이미지 등록 (Database Indexing) ---
with left_col:
    st.header("📤 이미지 등록")
    uploaded_file = st.file_uploader("DB에 저장할 이미지를 선택하세요", type=['jpg', 'jpeg', 'png'], key="upload")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="업로드된 이미지", use_container_width=True)
        
        if st.button("벡터 DB에 저장하기"):
            try:
                embedding = get_embedding(image)
                conn = get_db_connection()
                cur = conn.cursor()
                
                # 이미지 데이터를 DB에 저장 (여기서는 편의상 바이너리로 저장하거나 경로를 저장)
                # 실제 서비스에서는 S3 URL 등을 저장하는 것이 좋으나, 여기선 간단히 이미지 자체를 저장하지 않고 
                # 학습용이므로 이미지 이름만 저장한다고 가정합니다.
                cur.execute(
                    "INSERT INTO image_embeddings (image_url, embedding) VALUES (%s, %s)",
                    (uploaded_file.name, embedding)
                )
                conn.commit()
                cur.close()
                conn.close()
                st.success(f"'{uploaded_file.name}' 저장 완료!")
            except Exception as e:
                st.error(f"오류 발생: {e}")

# --- 우측: 유사 이미지 검색 (Vector Search) ---
with right_col:
    st.header("🔍 유사 이미지 검색")
    query_file = st.file_uploader("검색 쿼리로 사용할 이미지", type=['jpg', 'jpeg', 'png'], key="query")
    
    if query_file is not None:
        query_img = Image.open(query_file)
        st.image(query_img, caption="검색 쿼리 이미지", use_container_width=True)
        
        limit = st.slider("검색 결과 개수", 1, 5, 3)
        
        if st.button("유사 이미지 찾기"):
            try:
                query_vec = get_embedding(query_img)
                conn = get_db_connection()
                cur = conn.cursor()
                
                # 코사인 유사도 검색 (pgvector <=> 연산자)
                cur.execute("""
                    SELECT image_url, 1 - (embedding <=> %s) AS similarity
                    FROM image_embeddings
                    ORDER BY embedding <=> %s
                    LIMIT %s;
                """, (query_vec, query_vec, limit))
                
                results = cur.fetchall()
                
                if results:
                    st.subheader("검색 결과")
                    for name, score in results:
                        st.write(f"**파일명:** {name} (유사도: {score:.4f})")
                        # 주의: 실제 환경에선 저장된 경로에서 이미지를 불러와야 합니다.
                else:
                    st.info("검색 결과가 없습니다.")
                
                cur.close()
                conn.close()
            except Exception as e:
                st.error(f"오류 발생: {e}")
