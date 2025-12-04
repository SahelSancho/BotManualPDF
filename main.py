import os
import tempfile
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters
)
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from google import genai

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

client = genai.Client(api_key=GOOGLE_API_KEY)

user_data = {} 

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Olá! Sou seu Assistente Técnico Inteligente.\n"
        "Envie um manual em PDF (máx 20MB) para começarmos."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Envie um arquivo PDF para eu analisar.")

async def process_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    document = update.message.document

    if document.file_size > 20 * 1024 * 1024:
        await update.message.reply_text(
            "O arquivo é muito grande (maior que 20MB). "
            "A API do Telegram não permite que bots baixem arquivos desse tamanho diretamente."
        )
        return

    if document.mime_type != "application/pdf":
        await update.message.reply_text("Apenas arquivos PDF são aceitos.")
        return

    await update.message.reply_text("Baixando e processando PDF... Isso pode levar alguns segundos.")

    try:
        file = await context.bot.get_file(document.file_id)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            await file.download_to_drive(temp_pdf.name)
            temp_path = temp_pdf.name

        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
        chunks = splitter.split_documents(docs)

        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        embedding_model = HuggingFaceEmbeddings(model_name=model_name)

        vectorstore = FAISS.from_documents(chunks, embedding=embedding_model)

        user_data[user_id] = {
            "vectorstore": vectorstore,
            "history": []
        }

        os.remove(temp_path)

        await update.message.reply_text(
            f"PDF Processado com sucesso!\n"
            f"{len(chunks)} trechos indexados.\n"
            f"Pode fazer perguntas!"
        )

    except Exception as e:
        print(f"Erro ao processar PDF: {e}")
        await update.message.reply_text("Erro ao ler o PDF. Tente um arquivo mais simples ou sem proteção.")

async def handle_questions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    question = update.message.text

    if user_id not in user_data:
        await update.message.reply_text("📂 Por favor, envie um PDF primeiro!")
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        data = user_data[user_id]
        vectorstore = data["vectorstore"]
        history = data["history"]

        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        history_text = ""
        for q, a in history[-3:]:
            history_text += f"Human: {q}\nAI: {a}\n"

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        prompt_text = (
            "Você é um especialista técnico analisando um manual.\n"
            "Responda à pergunta baseando-se no contexto abaixo e no histórico da conversa.\n"
            "Se a resposta não estiver no contexto, diga que não encontrou, mas tente inferir com base no histórico se fizer sentido.\n\n"
            "--- HISTÓRICO DA CONVERSA ---\n"
            "{history}\n"
            "------------------------------\n\n"
            "--- CONTEXTO DO MANUAL ---\n"
            "{context}\n"
            "--------------------------\n\n"
            "Pergunta Atual: {question}"
        )
        
        prompt = PromptTemplate.from_template(prompt_text)

        docs = retriever.invoke(question)
        context_str = format_docs(docs)

        full_prompt = prompt.format(
            history=history_text,
            context=context_str,
            question=question
        )

        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=full_prompt
        )
        
        answer = response.text

        user_data[user_id]["history"].append((question, answer))

        await update.message.reply_text(answer)

    except Exception as e:
        print(f"Erro RAG: {e}")
        await update.message.reply_text("Erro ao gerar resposta. Tente reformular.")

if __name__ == "__main__":
    if not TELEGRAM_TOKEN:
        print("Erro: TELEGRAM_TOKEN ausente no .env")
        exit(1)

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.Document.PDF, process_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_questions))

    print("Bot iniciado! (Ctrl+C para parar)")
    app.run_polling()
