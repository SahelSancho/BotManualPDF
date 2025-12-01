# Bot Manual PDF -- Guia de Instalação e Execução

Este projeto é um bot em Python que utiliza **LangChain**, **FAISS**,
**Google Gemini** e **Python Telegram Bot** para responder perguntas com
base no conteúdo de arquivos PDF enviados pelo usuário.

------------------------------------------------------------------------

## Funcionalidades do Projeto

-   Recebe arquivos PDF enviados pelo Telegram.
-   Lê, extrai e indexa o conteúdo usando **FAISS**.
-   Utiliza **embeddings HuggingFace** para vetorização.
-   Usa modelo **Google Gemini 2.0-flash-exp para gerar respostas.
-   Executa um pipeline completo de **RAG (Retrieval-Augmented
    Generation)**.
-   Arquitetura limpa e pronta para produção.

------------------------------------------------------------------------

# Configuração do Ambiente

## 1. Criar o Ambiente Virtual

Abra o terminal do VS Code e execute:

    python -m venv venv

## 2. Habilitar Scripts no PowerShell (necessário apenas 1 vez)

    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

## 3. Ativar o Ambiente Virtual

    venv\Scripts\activate.bat

------------------------------------------------------------------------

# Instalação das Dependências

Com o ambiente ativado, instale tudo com:

    pip install python-telegram-bot python-dotenv langchain langchain-community langchain-core langchain-text-splitters langchain-huggingface sentence-transformers faiss-cpu google-genai pypdf

------------------------------------------------------------------------

# Arquivo `.env`

Nesse arquivo coloque o token do telegram e gemini nos lugares determinados:

    TELEGRAM_TOKEN=SEU_TOKEN_AQUI
    GOOGLE_API_KEY=SUA_CHAVE_DO_GEMINI

------------------------------------------------------------------------

# Executar o Bot

Com o ambiente ativado:

    python main.py

Se tudo estiver correto, aparecerá:

    Bot iniciado! (Ctrl+C para parar)
