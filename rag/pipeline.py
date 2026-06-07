import os
from openai import OpenAI
from rag.vector_store import Chunk

SYSTEM_PROMPT = """Você é um assistente especializado em dados espaciais e na nova economia do espaço.
Você responde perguntas sobre:
- Clima espacial e meteorologia
- Satélites e monitoramento orbital
- Agricultura de precisão com imagens de satélite
- Monitoramento ambiental e mudanças climáticas
- Desastres naturais e sensoriamento remoto
- Exploração espacial e astrofísica
- Missões espaciais da NASA, ESA e outras agências

REGRAS:
1. Use SOMENTE as informações dos documentos fornecidos no contexto para responder.
2. Se a resposta não estiver nos documentos, diga: "Não encontrei essa informação nos documentos carregados."
3. Responda SEMPRE em português brasileiro, de forma clara e objetiva.
4. Quando relevante, mencione a fonte do documento.
5. Se o contexto for insuficiente, peça ao usuário que carregue documentos relacionados ao tema."""


class RAGPipeline:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY não configurada. Crie um arquivo .env com sua chave."
            )
        self.client = OpenAI(api_key=api_key)
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def answer(self, query: str, chunks: list[Chunk]) -> tuple[str, list[str]]:
        if not chunks:
            return (
                "Nenhum documento carregado. Faça upload de arquivos na barra lateral antes de perguntar.",
                [],
            )

        context_parts = []
        sources = []
        seen_sources = set()

        for chunk in chunks:
            context_parts.append(f"[Fonte: {chunk.source}]\n{chunk.text}")
            if chunk.source not in seen_sources:
                sources.append(chunk.source)
                seen_sources.add(chunk.source)

        context = "\n\n---\n\n".join(context_parts)

        user_message = f"""Contexto extraído dos documentos:

{context}

---

Pergunta do usuário: {query}

Responda baseando-se apenas no contexto acima."""

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        answer_text = response.choices[0].message.content or "Não foi possível gerar uma resposta."

        return answer_text, sources
