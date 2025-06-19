import json
import pandas as pd

# arxiv JSON 파일 경로 (현재 디렉토리 기준)
JSON_PATH = "arxiv-metadata-oai-snapshot.json"
OUTPUT_PATH = "metadata/abstracts.csv"

# 최대 논문 수 (테스트용으로 제한)
MAX_COUNT = 2000
TARGET_CATEGORY = "cs.CV"  # 컴퓨터 비전 전용
START_DATE = "2024-01-01"

# 디렉토리 준비
import os
os.makedirs("metadata", exist_ok=True)

print("📦 arXiv JSON 불러오는 중...")

# JSON을 chunk 단위로 읽고 필터링
data = []
with open(JSON_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        paper = json.loads(line)

        if (
            paper.get("categories") and TARGET_CATEGORY in paper["categories"]
            and len(paper.get("abstract", "")) >= 100
            and paper.get("update_date", "0000") >= START_DATE
        ):
            data.append({
                "id": paper["id"],
                "title": paper["title"].strip().replace("\n", " "),
                "abstract": paper["abstract"].strip().replace("\n", " ")
            })

        if len(data) >= MAX_COUNT:
            break

print(f"✅ {len(data)}개 논문 metadata 추출 완료!")

# 저장
pd.DataFrame(data).to_csv(OUTPUT_PATH, index=False)
print(f"📄 저장 완료: {OUTPUT_PATH}")