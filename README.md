### image_vector_search

#### 1. 테스트용 이미지 데이터셋
테스트용 이미지 데이터셋은 아래 URL에서 다운 받을 수 있습니다. <br>
https://www.robots.ox.ac.uk/~vgg/data/pets/

FastAI Dogs vs Cats: 간단한 분류 테스트용으로 좋습니다. 

#### 2. PostgreSQL (pgvector) 준비
먼저 DB에 접속하여 확장 기능을 설치하고 테이블을 만듭니다. <br>
OpenAI의 CLIP 모델(clip-vit-base-patch32)을 사용할 경우 벡터 차원은 512입니다.

SQL
```
-- 기존 테이블이 있다면 삭제 후 재생성 (또는 ALTER TABLE 사용)
DROP TABLE IF EXISTS image_embeddings;

CREATE TABLE image_embeddings (
    id SERIAL PRIMARY KEY,
    file_name TEXT NOT NULL,         -- 파일명
    image_data BYTEA,               -- 이미지 바이너리 데이터
    file_size INTEGER,              -- 파일 사이즈 (Bytes)
    content_type TEXT,              -- 파일 형식 (image/png 등)
    embedding VECTOR(512),          -- 벡터 데이터 (CLIP 모델 기준 512차원)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 3. Python 환경 준비
Bash
```
pip install streamlit torch torchvision sentence-transformers psycopg2-binary Pillow requests python-dotenv
pip install -U sentence-transformers huggingface_hub
```

#### 4. 수행
OPENAI API Key, DB Conection Info에 대산 env 파일 수정.

```
cp env .env
streamlit run app.py
```

#### 핵심 포인트 설명
Vector 연산자 (<=>): pgvector에서 제공하는 연산자로, 벡터 간의 코사인 거리를 계산합니다. 1 - distance를 하면 유사도가 됩니다. <br>
<br>
CLIP 모델: OpenAI가 공개한 이 모델은 이미지뿐만 아니라 텍스트로도 이미지를 찾을 수 있게 해줍니다 <br>
(model.encode("A cute dog")를 쿼리로 사용 가능).<br>
<br>
성능 최적화: 데이터가 많아지면 아래와 같은 명령으로 HNSW 인덱스를 생성해 검색 속도를 높이세요. <br>
CREATE INDEX ON image_embeddings USING hnsw (embedding vector_cosine_ops);
