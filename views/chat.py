import os
import time
import html
import pandas as pd
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI, APIError
from sqlalchemy import create_engine, Column, Integer, String, Float, Text
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session

# Importações do fluxo vetorial (ChromaDB & SentenceTransformers)
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv(override=True)

if not st.session_state.get("autenticado", False):
    st.warning("Por favor, faça login para acessar esta página.")
    st.stop()

# --- VARIÁVEIS DE AMBIENTE ---
endpoint_env = os.getenv("ENDPOINT", "").strip()
api_key_env = os.getenv("API_KEY", "").strip()
api_version_env = os.getenv("API_VERSION", "2024-02-15-preview").strip()
deployment_name = os.getenv("DEPLOYMENT", "").strip()

client = None
if endpoint_env and api_key_env:
    try:
        client = AzureOpenAI(
            azure_endpoint=endpoint_env,
            api_key=api_key_env,
            api_version=api_version_env
        )
    except Exception as e:
        st.error(f"Erro ao inicializar o cliente AzureOpenAI: {e}")

# --- BANCO DE RELACIONAL (Histórico e Vendas) ---
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///chat_history.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    pool_pre_ping=True
)
SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
Base = declarative_base()

class HistoricoChat(Base):
    __tablename__ = "historico"
    id = Column(Integer, primary_key=True, index=True)
    role = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)

class Venda(Base):
    __tablename__ = "vendas"
    id = Column(String(100), primary_key=True, index=True)
    vendedor = Column(String(200), nullable=False)
    venda = Column(Float, nullable=False)

Base.metadata.create_all(bind=engine)

def salvar_mensagem(role: str, content: str):
    session = SessionLocal()
    try:
        content_sanitizado = html.escape(content)
        nova_msg = HistoricoChat(role=role, content=content_sanitizado)
        session.add(nova_msg)
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"Erro ao salvar mensagem: {e}")
    finally:
        session.close()

def carregar_historico():
    session = SessionLocal()
    try:
        mensagens = session.query(HistoricoChat).order_by(HistoricoChat.id.asc()).all()
        return [{"role": m.role, "content": html.unescape(m.content)} for m in mensagens]
    except Exception as e:
        print(f"Erro ao carregar histórico: {e}")
        return []
    finally:
        session.close()

# --- CARREGAMENTO DO MODELO DE EMBEDDINGS E BANCO VETORIAL (Cached) ---
@st.cache_resource
def carregar_recursos_vetoriais():
    # 1. Modelo de Embeddings (SentenceTransformer)[cite: 2]
    modelo_emb = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 2. Banco Vetorial Local (ChromaDB Client)[cite: 2]
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="rag_vetorial_corporativo")
    
    return modelo_emb, collection

modelo_embeddings, collection_vetorial = carregar_recursos_vetoriais()

# --- FUNÇÃO DE CHUNKING (Pré-processamento) ---
def fatiar_texto(texto: str, tamanho_chunk: int = 500, sobreposicao: int = 50) -> list:
    """Divide o texto bruto em pedaços menores (Chunks)[cite: 2]."""
    chunks = []
    for i in range(0, len(texto), tamanho_chunk - sobreposicao):
        chunk = texto[i:i + tamanho_chunk]
        if len(chunk.strip()) > 20:
            chunks.append(chunk)
    return chunks

# --- INGESTÃO E INDEXAÇÃO DE FONTES NO BANCO VETORIAL ---
@st.cache_data
def processar_e_indexar_fontes():
    # Evita re-indexar se a collection já possuir dados
    if collection_vetorial.count() > 0:
        return

    documentos = []
    metadados = []
    ids = []
    contador = 0

    # 1. Scraping do Site[cite: 2]
    url = os.getenv("URL_RASPAGEM", "https://gratuitos.netlify.app/")
    try:
        headers = {"User-Agent": "AgenteVirtual/1.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for script in soup(["script", "style", "iframe", "object"]):
            script.extract()
        texto_site = soup.get_text(separator="\n", strip=True)
        for chunk in fatiar_texto(texto_site):
            documentos.append(chunk)
            metadados.append({"fonte": "site", "origem": url})
            ids.append(f"site_{contador}")
            contador += 1
    except Exception as e:
        print(f"Aviso ao raspar site: {e}")

    # 2. Leitura do PDF[cite: 2]
    caminho_pdf = os.getenv("CAMINHO_PDF", "arquivo.pdf")
    if os.path.exists(caminho_pdf):
        try:
            reader = PdfReader(caminho_pdf)
            texto_pdf = "".join([p.extract_text() or "" for p in reader.pages])
            for chunk in fatiar_texto(texto_pdf):
                documentos.append(chunk)
                metadados.append({"fonte": "pdf", "origem": caminho_pdf})
                ids.append(f"pdf_{contador}")
                contador += 1
        except Exception as e:
            print(f"Aviso ao ler PDF: {e}")

    # 3. Leitura de CSVs locais[cite: 2]
    for arquivo_csv in ['dados.csv', 'tic.csv']:
        if os.path.exists(arquivo_csv):
            try:
                df = pd.read_csv(arquivo_csv).head(50)
                texto_csv = df.to_string()
                for chunk in fatiar_texto(texto_csv):
                    documentos.append(chunk)
                    metadados.append({"fonte": "csv", "origem": arquivo_csv})
                    ids.append(f"csv_{contador}")
                    contador += 1
            except Exception as e:
                print(f"Aviso ao ler CSV {arquivo_csv}: {e}")

    # Inserção no Banco Vetorial ChromaDB se houver documentos
    if documentos:
        embeddings = modelo_embeddings.encode(documentos).tolist()
        collection_vetorial.upsert(
            documents=documentos,
            embeddings=embeddings,
            metadatas=metadados,
            ids=ids
        )

# Executa a indexação inicial das fontes na memória/banco vetorial
processar_e_indexar_fontes()

# --- INTERFACE STREAMLIT ---
st.title("🤖 Agente RAG Vetorial Inteligente & Seguro")
st.caption("Arquitetura de Busca Semântica com ChromaDB e Azure OpenAI")

historico = carregar_historico()
for msg in historico:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if pergunta := st.chat_input("Digite sua dúvida..."):
    if not endpoint_env or not api_key_env or not deployment_name or not client:
        st.error("⚠️ Configuração ausente: Verifique no arquivo `.env` se ENDPOINT, API_KEY e DEPLOYMENT foram preenchidos.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(pergunta)
    salvar_mensagem("user", pergunta)

    t0 = time.time()

    # --- CONSULTA VETORIAL (Busca Semântica / Top-K) ---
    with st.spinner("Executando busca semântica vetorial..."):
        # Gera embedding da pergunta do usuário[cite: 2]
        pergunta_embedding = modelo_embeddings.encode([pergunta]).tolist()
        
        # Realiza a busca no banco vetorial local (Top-K trechos mais relevantes)[cite: 2]
        resultados_busca = collection_vetorial.query(
            query_embeddings=pergunta_embedding,
            n_results=4
        )
        
        trechos_recuperados = resultados_busca.get("documents", [[]])[0]
        fontes_metadados = resultados_busca.get("metadatas", [[]])[0]

    # Montagem do Contexto Aumentado
    contexto_aumentado = "Contexto Selecionado por Busca Vetorial Semântica:\n"
    for idx, trecho in enumerate(trechos_recuperados):
        origem = fontes_metadados[idx].get("origem", "desconhecido") if idx < len(fontes_metadados) else "desconhecido"
        contexto_aumentado += f"\n--- [Trecho {idx+1} | Fonte: {origem}] ---\n{trecho}\n"

    # Exibe telemetria dos trechos recuperados no Expander da interface[cite: 2]
    with st.expander("🔍 Telemetria: Trechos Recuperados (Busca Vetorial)", expanded=False):
        st.write(f"**Total de chunks recuperados:** {len(trechos_recuperados)}")
        st.json(fontes_metadados)
        st.text(contexto_aumentado)

    # Prompt de Sistema / Diretrizes do Modelo LLM[cite: 2]
    prompt_sistema = f"""
Você é um assistente corporativo seguro. Responda à dúvida do usuário utilizando EXCLUSIVAMENTE o contexto vetorial abaixo:

{contexto_aumentado}

Diretrizes:
1. Responda apenas com base nas informações fornecidas acima.
2. Se a informação não estiver disponível, informe educadamente que não possui essa resposta.
"""

    mensagens_api = [{"role": "system", "content": prompt_sistema}]
    for m in historico[-6:]:
        mensagens_api.append({"role": m["role"], "content": m["content"]})
    mensagens_api.append({"role": "user", "content": pergunta})

    resposta_texto = ""
    max_tentativas = 3

    for tentativa in range(max_tentativas):
        try:
            with st.spinner("Modelo LLM (Azure OpenAI) gerando resposta final..."):
                resp = client.chat.completions.create(
                    model=deployment_name,
                    messages=mensagens_api
                )
                resposta_texto = resp.choices[0].message.content #[cite: 2]
                break
        except APIError as e:
            status = getattr(e, 'status_code', None)
            if status == 429 or "429" in str(e):
                if tentativa < max_tentativas - 1:
                    time.sleep(5)
                else:
                    resposta_texto = "⚠️ Limite de requisições (Rate Limit) atingido. Tente novamente em instantes."
            elif status == 401:
                resposta_texto = "⚠️ Erro 401 (Não Autorizado): Verifique se a sua API_KEY do Azure OpenAI está correta."
                break
            elif status == 404:
                resposta_texto = f"⚠️ Erro 404: O nome do modelo/deployment '{deployment_name}' ou ENDPOINT estão incorretos no .env."
                break
            else:
                resposta_texto = f"⚠️ Erro na API do Azure OpenAI: {e}"
                break
        except Exception as e:
            resposta_texto = f"⚠️ Erro ao conectar ao serviço de IA: {e}"
            break

    tempo_total = round(time.time() - t0, 2)

    # Exibição final na interface Streamlit[cite: 2]
    with st.chat_message("assistant"):
        st.markdown(resposta_texto)
        st.caption(f"⏱️ Tempo de processamento vetorial e geração: {tempo_total}s")

    salvar_mensagem("assistant", resposta_texto)