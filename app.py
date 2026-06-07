import streamlit as st
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

st.set_page_config(
    page_title="Assistente Espacial RAG",
    page_icon="🛸",
    layout="wide",
)

DOCS_DIR = Path(__file__).parent / "docs"
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}


def autoload_docs():
    """Load all documents from the docs/ folder into the vector store."""
    if not DOCS_DIR.exists():
        return

    from rag.document_loader import load_document, chunk_text

    for file_path in sorted(DOCS_DIR.iterdir()):
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if file_path.name in st.session_state.loaded_docs:
            continue
        try:
            text = load_document(file_path.read_bytes(), file_path.name)
            chunks = chunk_text(text)
            st.session_state.vector_store.add(chunks, file_path.name)
            st.session_state.loaded_docs.append(file_path.name)
        except Exception as e:
            st.warning(f"Erro ao carregar {file_path.name}: {e}")


# ── Session state ──────────────────────────────────────────────────────────────

def init_state():
    if "vector_store" not in st.session_state:
        from rag.vector_store import VectorStore
        st.session_state.vector_store = VectorStore()

    if "pipeline" not in st.session_state:
        try:
            from rag.pipeline import RAGPipeline
            st.session_state.pipeline = RAGPipeline()
            st.session_state.api_error = None
        except ValueError as e:
            st.session_state.pipeline = None
            st.session_state.api_error = str(e)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "loaded_docs" not in st.session_state:
        st.session_state.loaded_docs = []

    # Só carrega os docs depois que a chave estiver configurada
    if st.session_state.pipeline is not None and not st.session_state.loaded_docs:
        autoload_docs()


init_state()


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🛸 Assistente Espacial")
    st.caption("RAG para Dados e Documentos Espaciais")
    st.divider()

    # API key alert
    if st.session_state.get("api_error"):
        st.error(f"⚠️ {st.session_state.api_error}")
        api_key_input = st.text_input(
            "Cole sua OPENAI_API_KEY aqui:", type="password"
        )
        if api_key_input:
            os.environ["OPENAI_API_KEY"] = api_key_input
            try:
                from rag.pipeline import RAGPipeline
                st.session_state.pipeline = RAGPipeline()
                st.session_state.api_error = None
                st.success("Chave configurada!")
                st.rerun()
            except Exception as e:
                st.error(f"Chave inválida: {e}")

    st.subheader("📁 Carregar Documentos")
    st.caption("Formatos suportados: PDF, TXT, DOCX")

    uploaded_files = st.file_uploader(
        "Selecione os arquivos",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("➕ Processar Documentos", use_container_width=True):
            from rag.document_loader import load_document, chunk_text

            progress = st.progress(0, text="Processando...")
            new_docs = []

            for i, uf in enumerate(uploaded_files):
                if uf.name in st.session_state.loaded_docs:
                    continue

                try:
                    text = load_document(uf.read(), uf.name)
                    chunks = chunk_text(text)
                    count = st.session_state.vector_store.add(chunks, uf.name)
                    st.session_state.loaded_docs.append(uf.name)
                    new_docs.append(f"✅ {uf.name} ({count} trechos)")
                except Exception as e:
                    new_docs.append(f"❌ {uf.name}: {e}")

                progress.progress((i + 1) / len(uploaded_files))

            progress.empty()
            for msg in new_docs:
                if msg.startswith("✅"):
                    st.success(msg)
                else:
                    st.error(msg)

    if st.session_state.loaded_docs:
        st.divider()
        st.subheader("📚 Documentos Carregados")
        for doc in st.session_state.loaded_docs:
            st.markdown(f"• {doc}")
        st.caption(
            f"Total de trechos no índice: **{st.session_state.vector_store.total}**"
        )

        if st.button("🗑️ Limpar tudo", use_container_width=True):
            st.session_state.vector_store.clear()
            st.session_state.loaded_docs = []
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.subheader("⚙️ Configurações RAG")
    top_k = st.slider("Trechos recuperados (top-k)", 1, 10, 5)

    st.divider()
    st.caption(
        "**Como funciona:**\n"
        "1. Documentos da pasta `docs/` são carregados automaticamente\n"
        "2. Os textos são divididos em trechos\n"
        "3. Embeddings vetoriais são gerados (OpenAI text-embedding-3-small)\n"
        "4. Sua pergunta busca trechos relevantes (FAISS)\n"
        "5. O Claude gera a resposta com base neles"
    )


# ── Main area ─────────────────────────────────────────────────────────────────

st.title("🛸 Assistente Inteligente com RAG")
st.caption("Faça perguntas sobre os documentos espaciais carregados")

# Welcome message
if not st.session_state.messages:
    if st.session_state.vector_store.total > 0:
        st.success(
            f"Documentos carregados automaticamente da pasta `docs/`. "
            f"**{st.session_state.vector_store.total} trechos** indexados e prontos. "
            "Faça sua pergunta abaixo!",
            icon="✅",
        )
    else:
        st.info(
            "**Bem-vindo!** Coloque documentos PDF, TXT ou DOCX na pasta `docs/` "
            "(relatórios da NASA, artigos científicos, dados de satélites etc.) "
            "e reinicie o app — eles serão carregados automaticamente. "
            "Ou faça upload diretamente pela barra lateral.",
            icon="👋",
        )

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Fontes consultadas"):
                for src in msg["sources"]:
                    st.markdown(f"• {src}")

# Input
if prompt := st.chat_input("Digite sua pergunta sobre dados espaciais..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate answer
    with st.chat_message("assistant"):
        if st.session_state.pipeline is None:
            answer = "Configure a OPENAI_API_KEY na barra lateral para usar o assistente."
            sources = []
            st.warning(answer)
        elif st.session_state.vector_store.total == 0:
            answer = "Nenhum documento carregado ainda. Faça upload de arquivos na barra lateral."
            sources = []
            st.warning(answer)
        else:
            with st.spinner("Buscando trechos relevantes e gerando resposta..."):
                chunks = st.session_state.vector_store.search(prompt, k=top_k)
                answer, sources = st.session_state.pipeline.answer(prompt, chunks)

            st.markdown(answer)
            if sources:
                with st.expander("📄 Fontes consultadas"):
                    for src in sources:
                        st.markdown(f"• {src}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
