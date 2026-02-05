import requests
import os
import json
import pandas as pd
import joblib

# ---------- CONFIG ----------
BATCH_SIZE = 64        # GPU: 32–64 | CPU: 8–16
MAX_CHARS = 1500
OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL_NAME = "bge-m3"
# ---------------------------


def embed_batch(texts):
    r = requests.post(
        OLLAMA_URL,
        json={"model": MODEL_NAME, "input": texts},
        timeout=120
    )
    response = r.json()

    if "error" in response:
        raise RuntimeError(response["error"])

    if "embeddings" in response:
        return response["embeddings"]

    if "data" in response:
        return [item["embedding"] for item in response["data"]]

    raise ValueError("Unknown response format")


def embed_single(text):
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "input": text},
            timeout=60
        )
        response = r.json()

        if "embedding" in response:
            return response["embedding"]

        if "data" in response:
            return response["data"][0]["embedding"]

    except Exception:
        pass

    return None


# ---------- MAIN PIPELINE ----------

json_dir = "jsons"
json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]

rows = []
chunk_id = 0

for json_file in json_files:
    print(f"Creating Embeddings for {json_file}")

    with open(os.path.join(json_dir, json_file), encoding="utf-8") as f:
        content = json.load(f)

    texts = []
    chunks = []

    # Clean input
    for chunk in content.get("chunks", []):
        text = chunk.get("text")

        if not text:
            continue

        text = str(text).strip()
        if not text or text.lower() == "nan":
            continue

        if len(text) > MAX_CHARS:
            continue

        texts.append(text)
        chunks.append(chunk)

    # Batch embedding with fallback
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]
        batch_chunks = chunks[i:i + BATCH_SIZE]

        try:
            embeddings = embed_batch(batch_texts)

            for j, emb in enumerate(embeddings):
                batch_chunks[j]["chunk_id"] = chunk_id
                batch_chunks[j]["embedding"] = emb
                batch_chunks[j]["source_file"] = json_file
                rows.append(batch_chunks[j])
                chunk_id += 1

        except RuntimeError:
            # Fallback to single embedding
            for text, chunk in zip(batch_texts, batch_chunks):
                emb = embed_single(text)
                if emb is None:
                    continue

                chunk["chunk_id"] = chunk_id
                chunk["embedding"] = emb
                chunk["source_file"] = json_file
                rows.append(chunk)
                chunk_id += 1

# Build DataFrame
df = pd.DataFrame.from_records(rows)

print("\n✅ Embedding generation complete")
print(f"Total chunks embedded: {len(df)}")
print(df.head())

joblib.dump(df , "embeddings.joblib")



