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
user_vectorstores = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Olá! Sou seu Assistente de Manuais.\n"
        "Envie um PDF para eu ler e depois faça perguntas!"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Envie um PDF para começar.")

async def process_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    document = update.message.document

    if document.mime_type != "application/pdf":
        await update.message.reply_text("Apenas arquivos PDF são aceitos.")
        return

    await update.message.reply_text("Processando PDF...")

    try:
        file = await context.bot.get_file(document.file_id)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            await file.download_to_drive(temp_pdf.name)
            temp_path = temp_pdf.name

        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        vectorstore = FAISS.from_documents(chunks, embedding=embedding_model)

        user_vectorstores[user_id] = vectorstore

        os.remove(temp_path)

        await update.message.reply_text(
            f"PDF processado! {len(chunks)} trechos indexados. Pode perguntar!"
        )

    except Exception as e:
        print("Erro ao processar PDF:", e)
        await update.message.reply_text("Erro ao ler o PDF.")

async def handle_questions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    question = update.message.text

    if user_id not in user_vectorstores:
        await update.message.reply_text("Envie um PDF primeiro!")
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        vectorstore = user_vectorstores[user_id]
        retriever = vectorstore.as_retriever()

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        prompt_text = (
            "Você é um assistente técnico.\n"
            "Use SOMENTE o contexto extraído do PDF para responder e responda em qualquer idioma solicitado.\n"
            "Se algo não estiver no manual, diga: 'Não encontrei isso no manual.'\n\n"
            "Contexto:\n{context}\n\n"
            "Pergunta: {question}"
        )
        prompt = PromptTemplate.from_template(prompt_text)

        rag_chain = (
            {"context": retriever | format_docs,
             "question": RunnablePassthrough()}
            | prompt
        )

        full_prompt = rag_chain.invoke(question)

        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=full_prompt
        )

        await update.message.reply_text(response.text)

    except Exception as e:
        print("Erro RAG:", e)
        await update.message.reply_text("Erro ao gerar resposta.")

if __name__ == "__main__":
    if not TELEGRAM_TOKEN:
        print("Erro: TELEGRAM_TOKEN ausente no .env")
        exit(1)

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.Document.PDF, process_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_questions))

    print("🤖 Bot iniciado! (Ctrl+C para parar)")
    app.run_polling()
