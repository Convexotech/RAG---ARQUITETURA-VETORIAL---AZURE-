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