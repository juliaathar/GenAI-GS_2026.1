# Como Rodar o Assistente Espacial RAG

## 1. Instalar dependências

Abra o PowerShell na pasta `rag-espacial` e execute:

```powershell
pip install -r requirements.txt
```

## 2. Configurar a API Key da OpenAI

Copie o arquivo `.env.example` para `.env`:

```powershell
Copy-Item .env.example .env
```

Abra o `.env` e substitua `sua_chave_aqui` pela sua chave da OpenAI:

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

## 3. Rodar o aplicativo

```powershell
streamlit run app.py
```

O navegador abrirá automaticamente em `http://localhost:8501`.

## 4. Usar o assistente

1. Os documentos da pasta `docs/` são carregados automaticamente ao iniciar
2. Para adicionar mais arquivos: use o upload na barra lateral (PDF, TXT ou DOCX)
3. Digite sua pergunta no campo de chat
4. O assistente responde com base nos documentos, exibindo as fontes consultadas

## Documentos incluídos

A pasta `docs/` já contém os seguintes documentos espaciais indexados automaticamente:

- `nasa-2024-cap.pdf` — Plano de Adaptação Climática da NASA (2024)
- `2024_01_infoqueima.pdf` — Boletim INPE de monitoramento de focos de fogo
- `ENLoTIS_Report_2024_v101red.pdf` — Missão ESA/NASA de monitoramento da termosfera
- `GTselioudisEnvSciPSDGlobal*.pdf` — Artigos sobre aceleração do aquecimento global
- `2602.08924v2.pdf` — Artigo acadêmico sobre otimização de constelações de satélites
- `exemplo_dados_espaciais.txt` — Texto de exemplo sobre nova economia espacial

## Arquitetura RAG

```
Documento (PDF/TXT/DOCX)
        ↓
  Chunking (400 palavras, overlap 60)
        ↓
  Embeddings via OpenAI text-embedding-3-small (384 dims)
        ↓
  FAISS IndexFlatIP (cosine similarity)
        ↓
  Pergunta → Embedding → Busca top-k trechos
        ↓
  GPT-4o-mini → Resposta em português com citação de fontes
```
