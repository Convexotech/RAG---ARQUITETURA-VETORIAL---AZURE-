# CONTEXT (Contexto)

Você é um Arquiteto de Soluções de Inteligência Artificial sênior apresentando um projeto prático.
Foi desenvolvida uma solução de orquestração centralizada no padrão vetorial que integra dois agentes de ponta:

1. Agente (Azure AI Foundry): Atua como supervisor, executando raciocínio profundo, análise de causa-raiz e síntese executiva.
2. Agente Spoke (azure / apenai): Atua como especialista em inferência de altíssima velocidade, realizando triagem inicial e extração de tags em milissegundos.

# OBJECTIVE (Objetivo)

Crie a aplicação de Arquitetura da Solução multi agente vetorial:

Apresentar uma demonstração técnica e fluida da arquitetura funcionando em tempo real. A apresentação deve:

1. Explicar como a requisição flui da interface streamlit para o Hub e é delegada aos Spokes.
2. Demonstrar o desempenho em produção testando entradas reais (ex: incidentes críticos e feedbacks de usuários).
3. Evidenciar a complementariedade entre os dois provedores: a baixa latência do Groq combinada com a capacidade analítica da Azure AI Foundry.

# STYLE (Estilo)

Profissional, direto, focado em arquitetura de software e engenharia de IA. Utilize tópicos concisos, comparações de performance/latência e trechos de JSON estruturados.

# TONE (Tom)

Técnico, seguro, inovador e persuasivo. Deve transmitir maturidade técnica na escolha de combinar diferentes provedores de LLM.

# AUDIENCE (Público-Alvo)

Tech leads, desenvolvedores sêniores, diretores de tecnologia ou clientes que buscam entender o valor prático de orquestrar múltiplos modelos de IA (Multi-Model / Multi-Agent Systems).

# RESPONSE FORMAT (Formato de Resposta)

Apresente a demonstração na seguinte estrutura:

- 🏗️ Arquitetura da Solução multi agente vetorial (Apresentação do vetorial entre dois agenteste conforme citado acima)
- ⚡ O Valor da Combinação (Por que usar azure para velocidade + Azure para profundidade)
- 🎬 Roteiro da Demonstração (Casos de teste em tempo real com entradas e saídas JSON esperadas)
- 📊 Análise de Resultados (Ganhos de latência, precisão do diagnóstico e resposta consolidada)
- testes local 1 maquina 
- interface grafica streamlit
- API  azure 
- Agente RAG, base do projeto: 

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

load_dotenv(override=True)

# Proteção de acesso: redireciona se o usuário não estiver autenticado
if not st.session_state.get("autenticado", False):
    st.warning("Por favor, faça login para acessar esta página.")
    st.stop()

# --- FORMATAÇÃO E VALIDAÇÃO DAS VARIÁVEIS DE AMBIENTE ---
endpoint_env = os.getenv("ENDPOINT", "").strip()
api_key_env = os.getenv("API_KEY", "").strip()
api_version_env = os.getenv("API_VERSION", "2024-02-15-preview").strip()
deployment_name = os.getenv("DEPLOYMENT", "").strip()

# Inicializa o cliente AzureOpenAI com tratamento defensivo
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

# --- CONFIGURAÇÃO DO BANCO DE DADOS E ORM ---
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

# --- GERENCIAMENTO DE MEMÓRIA DO CHAT ---
def salvar_mensagem(role: str, content: str):
    session = SessionLocal()
    try:
        content_sanitizado = html.escape(content)
        nova_msg = HistoricoChat(role=role, content=content_sanitizado)
        session.add(nova_msg)
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"Erro ao salvar mensagem no histórico: {e}")
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

# --- FUNÇÕES DE RETRIEVAL (RAG & SCRAPING) ---
@st.cache_data
def raspar_site():
    url = os.getenv("URL_RASPAGEM", "https://gratuitos.netlify.app/")
    try:
        headers = {"User-Agent": "AgenteVirtual/1.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for script in soup(["script", "style", "iframe", "object"]):
            script.extract()
        texto = soup.get_text(separator="\n", strip=True)
        return texto[:3000] if texto else "[Site sem conteúdo texto]"
    except Exception as e:
        return f"[Aviso: Site indisponível ({e})]"

@st.cache_data
def ler_pdf():
    caminho_pdf = os.getenv("CAMINHO_PDF", "arquivo.pdf")
    if not os.path.exists(caminho_pdf):
        return "[Aviso: PDF não encontrado]"
    try:
        reader = PdfReader(caminho_pdf)
        texto = "".join([p.extract_text() or "" for p in reader.pages])
        return texto[:3000] if texto.strip() else "[PDF sem texto]"
    except Exception as e:
        return f"[Aviso: Falha ao ler PDF ({e})]"

def carregar_dados_locais():
    try:
        dados_csv = pd.read_csv('dados.csv').head(50).to_string()
    except Exception:
        dados_csv = "[dados.csv não encontrado]"

    try:
        t_csv = pd.read_csv('tic.csv').head(50).to_string()
    except Exception:
        t_csv = "[tic.csv não encontrado]"

    session = SessionLocal()
    try:
        vendas = session.query(Venda).limit(50).all()
        dados_db = str([{"id": v.id, "vendedor": v.vendedor, "venda": v.venda} for v in vendas]) if vendas else "Sem registros de vendas."
    except Exception as e:
        dados_db = f"[Erro ao ler banco de vendas: {e}]"
    finally:
        session.close()

    return dados_csv, t_csv, dados_db

# --- ORQUESTRADOR DE CONTEXTO ---
def orquestrar_fontes(pergunta: str):
    p = pergunta.lower()
    precisa_site = any(k in p for k in ["site", "curso", "geral", "sobre", "gratuitos", "institucional"])
    precisa_pdf = any(k in p for k in ["pdf", "regulamento", "documento", "regra", "edital"])
    precisa_csv = any(k in p for k in ["dado", "tabela", "informação", "tic"])
    precisa_vendas = any(k in p for k in ["venda", "vendedor", "valor", "preço", "comercial", "id"])

    if not (precisa_site or precisa_pdf or precisa_csv or precisa_vendas):
        precisa_site = precisa_pdf = precisa_csv = precisa_vendas = True

    dados_csv, t_csv, dados_db = carregar_dados_locais() if (precisa_csv or precisa_vendas) else ("", "", "")
    conteudo_site = raspar_site() if precisa_site else ""
    conteudo_pdf = ler_pdf() if precisa_pdf else ""

    contexto = "Contexto Selecionado pela Orquestração:\n"
    if precisa_site and conteudo_site: contexto += f"\n----[SITE]----\n{conteudo_site}\n"
    if precisa_pdf and conteudo_pdf: contexto += f"\n----[PDF]----\n{conteudo_pdf}\n"
    if precisa_csv and dados_csv: contexto += f"\n----[CSV 1]----\n{dados_csv}\n"
    if precisa_csv and t_csv: contexto += f"\n----[CSV 2]----\n{t_csv}\n"
    if precisa_vendas and dados_db: contexto += f"\n----[VENDAS]----\n{dados_db}\n"

    return contexto

# --- INTERFACE DO CHAT STREAMLIT ---
st.title("🤖 Agente Virtual Inteligente & Seguro")
st.write("Interface principal de conversa com o assistente orquestrado.")

# Renderiza histórico armazenado no banco
historico = carregar_historico()
for msg in historico:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Entrada de pergunta do usuário
if pergunta := st.chat_input("Digite sua mensagem..."):
    # Validação prévia das credenciais
    if not endpoint_env or not api_key_env or not deployment_name or not client:
        st.error("⚠️ Configuração ausente: Verifique no arquivo .env se ENDPOINT, API_KEY e GPT5_MODEL foram preenchidos corretamente.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(pergunta)
    salvar_mensagem("user", pergunta)

    contexto = orquestrar_fontes(pergunta)

    prompt_sistema = f"""
Você é um assistente corporativo seguro. Responda à dúvida do usuário utilizando EXCLUSIVAMENTE o contexto abaixo:

{contexto}

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
            with st.spinner("Agente consultando as fontes de dados..."):
                resp = client.chat.completions.create(
                    model=deployment_name,
                    messages=mensagens_api,
                    
                )
                resposta_texto = resp.choices[0].message.content
                break
        except APIError as e:
            status = getattr(e, 'status_code', None)
            if status == 429 or "429" in str(e):
                if tentativa < max_tentativas - 1:
                    time.sleep(5)
                else:
                    resposta_texto = "⚠️ Limite de requisições (Rate Limit) atingido. Tente novamente em instantes."
            elif status == 401:
                resposta_texto = f"⚠️ Erro 401 (Não Autorizado): Verifique se a sua API_KEY do Azure OpenAI está correta."
                break
            elif status == 404:
                resposta_texto = f"⚠️ Erro 404 (Não Encontrado): O nome do modelo/deployment '{deployment_name}' ou o ENDPOINT estão incorretos no .env."
                break
            else:
                resposta_texto = f"⚠️ Erro na API do Azure OpenAI ({e.status_code if status else 'APIError'}): {e.message if hasattr(e, 'message') else e}"
                break
        except Exception as e:
            # Exibe o erro detalhado diretamente na tela para facilitar o diagnóstico
            resposta_texto = f"⚠️ Erro ao conectar ao serviço de IA: {e}"
            break

    with st.chat_message("assistant"):
        st.markdown(resposta_texto)
    salvar_mensagem("assistant", resposta_texto)


.env
# ENDPOINT=
# API_KEY=
# DEPLOYMENT=
# API_VERSION=
# DEFAULT_ADMIN_PASSWORD=


login.py

import os
import hashlib
import streamlit as st
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session

# --- CONFIGURAÇÃO DO ORM ---
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

class Usuario(Base):
    __tablename__ = "usuarios"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(128), nullable=False)

def hash_senha(senha: str) -> str:
    return hashlib.sha256(senha.encode()).hexdigest()

def inicializar_banco():
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    try:
        if not session.query(Usuario).filter_by(username="admin").first():
            senha_padrao = os.getenv("DEFAULT_ADMIN_PASSWORD", "admin123")
            admin = Usuario(username="admin", password_hash=hash_senha(senha_padrao))
            session.add(admin)
            session.commit()
    except Exception as e:
        session.rollback()
        print(f"Erro ao criar usuario padrão: {e}")
    finally:
        session.close()

# Inicializa banco ao carregar a página
inicializar_banco()

st.title("🔒 Acesso Restrito ao Sistema")
st.write("Insira suas credenciais para liberar o acesso ao Agente Virtual.")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    with st.form("form_login"):
        usuario_input = st.text_input("Usuário")
        senha_input = st.text_input("Senha", type="password")
        submit_button = st.form_submit_button("Entrar", use_container_width=True)

        if submit_button:
            session = SessionLocal()
            try:
                user = session.query(Usuario).filter_by(username=usuario_input).first()
                if user and user.password_hash == hash_senha(senha_input):
                    st.session_state.autenticado = True
                    st.session_state.usuario = usuario_input
                    st.success("Login realizado com sucesso!")
                    st.rerun()
                else:
                    st.error("Usuário ou senha inválidos.")
            except Exception as e:
                st.error(f"Erro no servidor: {e}")
            finally:
                session.close()


app.py:

import streamlit as st

st.set_page_config(page_title="Agente IA Seguro", page_icon="🔒", layout="centered")

# Inicializa o estado de autenticação na sessão
if "autenticado" not in st.session_state:
    st.session_state.autenticado = False

if "usuario" not in st.session_state:
    st.session_state.usuario = None

# Função de logout
def logout():
    st.session_state.autenticado = False
    st.session_state.usuario = None
    st.rerun()

# Definição das Páginas
pagina_login = st.Page("views/login.py", title="Login", icon="🔑", default=True)
pagina_chat = st.Page("views/chat.py", title="Agente Virtual", icon="🤖")

# Controle de Acesso e Navegação
if not st.session_state.autenticado:
    # Quando NÃO autenticado: apenas a página de Login fica acessível
    pg = st.navigation([pagina_login])
else:
    # Quando AUTENTICADO: libera o Chat e adiciona o botão de Sair na barra lateral
    pg = st.navigation([pagina_chat])
    st.sidebar.write(f"Conectado como: **{st.session_state.usuario}**")
    st.sidebar.button("Sair (Logout)", on_click=logout, use_container_width=True)

# Executa a página selecionada
pg.run()

