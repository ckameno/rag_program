import os
import streamlit as st
import requests
import glob
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

# Ollama APIのエンドポイント
ollama_url = "http://127.0.0.1:11434/v1/chat/completions"

# text splitterの定義
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=100,
    length_function=len,
    separators=[
        "\n", " ", ".", ",", ";", ":", "(", ")", "[", "]", "{", "}", "<", ">", '"', "'", 
        "、", "。", "，", "；", "：", "（", "）", "【", "】", "「", "」", "『", "』", 
        "〈", "〉", "《", "》", "“", "”"],
)

# Embessingsの定義
embeddings = OllamaEmbeddings(model="gemma2:9b")

# ChromaDBの初期化
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db"
)

# Retrieverの設定
retriever = vector_store.as_retriever(
    search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10}
)

# 通常のチャット関数
def default_chat(prompt):
    headers = {"Content-Type": "application/json"}      
    data = {
        "model": "gemma2:9b",
        "messages": [
            {"role": "system", "content": "You are AI assistant."},
            {"role": "user", "content": prompt}
        ]
    }         
    response = requests.post(ollama_url, json=data, headers=headers)           
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return "Error: " + response.text

# RAGチャット処理
def chat_with_retriever(prompt):
    retriever_response = retriever.invoke(prompt)
    context = retriever_response[0].page_content if retriever_response else "No relevant document found."

    headers = {"Content-Type": "application/json"}
    data = {
        "model": "gemma2:9b",
        "messages": [
            {"role": "system", "content": "You are AI assistant. Use the provided document to answer."},
            {"role": "user", "content": f"Document:\n{context}"},
            {"role": "assistant", "content": "I will answer using the document."},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(ollama_url, json=data, headers=headers)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return "Error: " + response.text

# Streamlitのチャット履歴管理
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# PDFファイルをベクトル化し、ChromaDBに追加
def pdf_to_vector():
    path = "./pdf_docs"
    pdf_list = []
    if os.path.exists("pdf_list_stored.txt"):
        with open("pdf_list_stored.txt", "r") as f:
            for line in f:
                pdf_list.append(line.strip())
    else:
        with open("pdf_list_stored.txt", "w") as f:
            pass

    pdf_list_current = glob.glob(path + "/*.pdf")
    pdf_list_new = list(set(pdf_list_current) - set(pdf_list))

    for pdf in pdf_list_new:
        loader = PyMuPDFLoader(pdf)
        texts = loader.load_and_split(text_splitter)
        vector_store.add_documents(texts)
        pdf_list.append(pdf)

    with open("pdf_list_stored.txt", "w") as f:
        for pdf in pdf_list:
            f.write(pdf + "\n")

# Markdownファイルをベクトル化し、ChromaDBに追加
def md_to_vector():
    path = "./md_docs"
    md_list = []
    if os.path.exists("md_list_stored.txt"):
        with open("md_list_stored.txt", "r") as f:
            for line in f:
                md_list.append(line.strip())
    else:
        with open("md_list_stored.txt", "w") as f:
            pass

    md_list_current = glob.glob(path + "/*.md")
    md_list_new = list(set(md_list_current) - set(md_list))

    for md in md_list_new:
        loader = UnstructuredMarkdownLoader(md)
        texts = loader.load_and_split(text_splitter)
        vector_store.add_documents(texts)
        md_list.append(md)

    with open("md_list_stored.txt", "w") as f:
        for md in md_list:
            f.write(md + "\n")

# チャット履歴を画面に表示
def display_chat_history():
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.write("User: " + chat["content"])
        else:
            st.write("Bot: " + chat["content"])

# Streamlitアプリのタイトル
st.title('RAG機能付きチャットボット')

# 履歴表示
display_chat_history()

# ユーザのプロンプト入力欄
prompt = st.text_area('プロンプト入力欄', )

# ボタンの配置
button1, button2, button3, button4 = st.columns(4)

# 通常チャット実行
if button1.button('チャット'):
    chat_response = default_chat(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.session_state.chat_history.append({"role": "system", "content": chat_response})
    st.rerun()

# RAGチャット実行
if button2.button('RAG'):
    rag_response = chat_with_retriever(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.session_state.chat_history.append({"role": "system", "content": rag_response})
    st.write(rag_response)
    st.rerun()

# PDFベクトル化実行
if button3.button('PDFをベクトル化'):
    pdf_to_vector()
    st.success("PDFベクトル化完了")
    st.rerun()

# Markdownベクトル化実行
if button4.button('Markdownをベクトル化'):
    md_to_vector()
    st.success("Markdownベクトル化完了")
    st.rerun()
