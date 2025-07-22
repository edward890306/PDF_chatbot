from langchain.chains import ConversationalRetrievalChain ## 對話檢索鏈
from langchain_community.document_loaders import PyPDFLoader ## PDF載入器
from langchain_community.vectorstores import FAISS # 要儲存的向量資料庫
from langchain_openai import OpenAIEmbeddings # 嵌入向量
from langchain_openai import ChatOpenAI # 聊天機器人
from langchain_text_splitters import RecursiveCharacterTextSplitter # 遞歸字元文字切割器


def qa_agent(openai_api_key, memory, uploaded_file, question):
    model = ChatOpenAI(model="gpt-3.5-turbo", api_key=openai_api_key) # 原openai_api_key = openai_api_key 會報錯

    ## pdf 為二進位資料
    file_content = uploaded_file.read() # 這行程式碼會讀取 uploaded_file（假設這是一個上傳的 PDF 檔案，類型可能是 werkzeug.datastructures.FileStorage 或其他類型的檔案物件），並將檔案的內容讀取為 二進位格式(bytes)
    temp_file_path = "temp.pdf"
    with open(temp_file_path, "wb") as temp_file: # 這部分的目的是將剛剛讀取的 PDF 內容寫入一個名為 temp.pdf 的本地臨時檔案,以寫入二進制模式(wb)開啟temp.pdf
        temp_file.write(file_content)  # 將file_content 寫入這個檔案中

    loader = PyPDFLoader(temp_file_path) # 載入PDF
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter( #根據下面符號進行切割 索引排前優先切割 切割上下文覆蓋50個字元
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n", "。", "！", "？", "，", "、", ""]
    )
    texts = text_splitter.split_documents(docs) # 根據字元切割器進行切割
    embeddings_model = OpenAIEmbeddings() # 設置嵌入向量
    db = FAISS.from_documents(texts, embeddings_model) # 將切割完的文字進行嵌入向量 (等同於把文字變成向量)
    retriever = db.as_retriever() # 設置檢索器
    qa = ConversationalRetrievalChain.from_llm( #設置對話檢索鏈
        llm=model,
        retriever=retriever,
        memory=memory
    )
    response = qa.invoke({"chat_history": memory, "question": question})
    return response
