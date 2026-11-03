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