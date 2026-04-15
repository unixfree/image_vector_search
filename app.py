import streamlit as st
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import psycopg2
import os
import io
from dotenv import load_dotenv

# --- 0. 환경 변수 로드 ---
load_dotenv()

# --- 1. 모델 로드 (Transformers 직접 사용) ---
@st.cache_resource
def load_clip_model():
    model_id = "openai/clip-vit-base-patch32"
    # 이미지와 텍스트를 모두 처리할 수 있는 모델과 프로세서 로드
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    return model, processor

clip_model, clip_processor = load_clip_model()

# --- 2. DB 연결 함수 ---
def get_db_connection():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )

# --- 3. 핵심 로직: 이미지 임베딩 추출 ---
def get_image_embedding(image):
    # 1. 이미지 전처리
    inputs = clip_processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        # 2. 이미지 특징 추출
        outputs = clip_model.get_image_features(**inputs)
        
        # 3. 객체(Object)에서 실제 텐서(Tensor) 추출
        # outputs가 텐서가 아닐 경우를 대비해 본체 데이터를 꺼냅니다.
        if hasattr(outputs, "pooler_output"):
            image_features = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state"):
            image_features = outputs.last_hidden_state
        elif isinstance(outputs, (list, tuple)):
            image_features = outputs[0]
        else:
            image_features = outputs # 이미 텐서인 경우

        # 4. 이제 텐서이므로 연산이 가능합니다.
        image_features = image_features.detach().cpu()
        
    # 5. 벡터 정규화 (유사도 정확도 향상)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    # 6. 리스트로 변환 (pgvector 저장용)
    return image_features.squeeze().tolist()

# --- 4. Streamlit UI 구성 ---
st.set_page_config(layout="wide", page_title="AI 이미지 메타데이터 저장소")
st.title("🖼️ CLIP 기반 이미지 벡터 검색 엔진")
st.caption("PostgreSQL pgvector + Transformers CLIP")

left_col, right_col = st.columns(2)

# --- [좌측: 이미지 등록 및 메타데이터 저장] ---
with left_col:
    st.header("📤 이미지 등록")
    uploaded_file = st.file_uploader("DB에 저장할 파일 선택", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        file_name = uploaded_file.name
        file_size = uploaded_file.size
        binary_data = uploaded_file.getvalue()
        image = Image.open(uploaded_file).convert("RGB") # CLIP 모델을 위해 RGB 변환
        
        st.image(image, caption=f"{file_name} ({file_size} bytes)", use_container_width=True)
        
        if st.button("벡터 및 메타데이터 저장"):
            try:
                # 벡터 생성
                with st.spinner("이미지 분석 중..."):
                    embedding = get_image_embedding(image)
                
                # DB 저장
                with get_db_connection() as conn:
                    with conn.cursor() as cur:
                        sql = """
                        INSERT INTO image_embeddings 
                        (file_name, image_data, file_size, embedding) 
                        VALUES (%s, %s, %s, %s::vector)
                        """
                        cur.execute(sql, (
                            file_name, 
                            psycopg2.Binary(binary_data), 
                            file_size, 
                            embedding
                        ))
                        conn.commit()
                st.success(f"Successfully saved: {file_name}")
            except Exception as e:
                st.error(f"저장 실패: {str(e)}")

# --- [우측: 유사 이미지 검색 및 복원] ---
with right_col:
    st.header("🔍 유사 이미지 검색")
    query_file = st.file_uploader("검색용 이미지 업로드", type=['jpg', 'jpeg', 'png'], key="query")
    
    if query_file:
        query_img = Image.open(query_file).convert("RGB")
        st.image(query_img, use_container_width=True, caption="검색 쿼리")
        
        if st.button("가장 유사한 이미지 찾기"):
            try:
                with st.spinner("검색 중..."):
                    query_vec = get_image_embedding(query_img)
            
                    with get_db_connection() as conn:
                        with conn.cursor() as cur:
                            # %s 뒤에 ::vector를 붙여서 명시적으로 타입을 변환합니다.
                            cur.execute("""
                                SELECT file_name, image_data, file_size, 1 - (embedding <=> %s::vector) AS similarity
                                FROM image_embeddings
                                ORDER BY embedding <=> %s::vector LIMIT 1;
                            """, (query_vec, query_vec))
                            result = cur.fetchone()
        
                if result:
                    res_name, res_binary, res_size, res_score = result
                    st.subheader(f"유사도 결과: {res_score:.2%} 일치")
                    st.write(f"📄 파일명: {res_name} ({res_size} bytes)")
                    
                    # 바이너리로부터 이미지 복원
                    res_image = Image.open(io.BytesIO(res_binary))
                    st.image(res_image, caption="DB 복원 이미지", use_container_width=True)
                else:
                    st.warning("DB에 등록된 이미지가 없습니다.")
            except Exception as e:
                st.error(f"검색 실패: {str(e)}")
