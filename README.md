# image_vector_search

### 1. 테스트용 이미지 데이터셋
인터넷상에서 바로 URL 형태로 접근하여 테스트할 수 있는 데이터셋들입니다.

Unsplash Source: https://images.unsplash.com/photo-12345678 형식으로 다양한 고해상도 이미지를 테스트할 수 있습니다.

FastAI Dogs vs Cats: 간단한 분류 테스트용으로 좋습니다. 데이터셋 링크

COCO Dataset: 객체 탐지 및 캡셔닝용 데이터셋으로, 검색 엔진 테스트에 가장 적합합니다. COCO Explorer에서 이미지 URL을 얻을 수 있습니다.

### 2. PostgreSQL (pgvector) 준비
먼저 DB에 접속하여 확장 기능을 설치하고 테이블을 만듭니다. OpenAI의 CLIP 모델(clip-vit-base-patch32)을 사용할 경우 벡터 차원은 512입니다.

SQL
```
-- pgvector 확장 활성화
CREATE EXTENSION IF NOT EXISTS vector;

-- 이미지 정보를 저장할 테이블 생성
CREATE TABLE image_embeddings (
    id SERIAL PRIMARY KEY,
    image_url TEXT,
    embedding VECTOR(512) -- CLIP 모델의 차원에 맞춤
);
```

### 3. Python 코드 구현
OpenAI API를 직접 사용하는 대신, 로컬에서 실행 가능한 OpenAI의 CLIP 모델(이미지 임베딩 전용)을 사용하는 것이 속도와 비용 면에서 효율적입니다.

필수 라이브러리 설치
Bash
```
pip install streamlit torch torchvision sentence-transformers psycopg2-binary Pillow requests python-dotenv
```

### 핵심 포인트 설명
Vector 연산자 (<=>): pgvector에서 제공하는 연산자로, 벡터 간의 코사인 거리를 계산합니다. 1 - distance를 하면 유사도가 됩니다.

CLIP 모델: OpenAI가 공개한 이 모델은 이미지뿐만 아니라 텍스트로도 이미지를 찾을 수 있게 해줍니다 (model.encode("A cute dog")를 쿼리로 사용 가능).

성능 최적화: 데이터가 많아지면 CREATE INDEX ON image_embeddings USING hnsw (embedding vector_cosine_ops); 명령으로 HNSW 인덱스를 생성해 검색 속도를 높이세요.
