import os
import numpy as np

def get_client():
    from openai import OpenAI
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))


def embed_texts(texts: list[str]) -> np.ndarray:
    client = get_client()
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
        dimensions=384,
    )
    embeddings = [item.embedding for item in response.data]
    return np.array(embeddings, dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    return embed_texts([query])[0]
