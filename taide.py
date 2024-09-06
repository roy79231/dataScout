from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
import pdfplumber
import os
from sqlalchemy import create_engine, Column, Integer, String, Text,inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import torch
from sentence_transformers import SentenceTransformer, util
import warnings
import pandas as pd
import numpy as np


app = Flask(__name__)
CORS(app)

@app.route('/test', methods=['GET'])
def test():
    return "hello"

@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    user_input = request.form.get('user_input')  # 獲取使用者輸入
    try:
        response =TAIDEchat(user_input)
        return jsonify({'response': response})
    
    except Exception as e:
        # 打印錯誤訊息便於除錯
        print(f"處理 PDF 時發生錯誤: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/enter_url', methods=['POST'])
def enter_url():
    # 取得 download 資料夾中的所有文件
    download_dir = r"C:\Users\user\Desktop\專題\use_flask\download"
    files = os.listdir(download_dir)

    for file_name in files:
        file_path = os.path.join(download_dir, file_name)  # 使用 os.path.join 拼接文件路徑
        print(file_path)
        index_pdf(file_path)  # 處理 PDF 文件
    
    return  jsonify({"message": "已讀取完該網頁"}), 200 

#roy用的資料庫
embedder = SentenceTransformer('BAAI/bge-m3',device='cuda' if torch.cuda.is_available() else 'cpu')

# 建立資料庫引擎和會話
engine = create_engine('sqlite:///embeddings_bge_m3.db')
Base = declarative_base()
Session = sessionmaker(bind=engine)
session = Session()

# 定義資料庫表格模型
class Embedding(Base):
    __tablename__ = 'embeddings'
    id = Column(Integer, primary_key=True)
    text = Column(Text, nullable=False)
    embedding = Column(Text, nullable=False)

# 檢查資料庫表格是否存在
inspector = inspect(engine)
if 'embeddings' not in inspector.get_table_names():
    Base.metadata.create_all(engine)
    
def save_embedding_to_db(text, embedding):
    """將嵌入向量存入資料庫"""
    exists = session.query(Embedding).filter_by(text=text).first()
    if exists:
        return
    
    # 確保嵌入從 GPU 轉移到 CPU
    embedding = embedding.cpu().numpy()
    embedding_str = ','.join([str(x) for x in embedding])
    new_entry = Embedding(text=text, embedding=embedding_str)
    session.add(new_entry)
    session.commit()

def load_embeddings_from_db():
    """從資料庫中載入所有嵌入向量和對應的文本"""
    results = session.query(Embedding).all()
    embeddings = []
    for result in results:
        embedding = np.fromstring(result.embedding, sep=',', dtype=np.float32)
        embeddings.append((embedding, result.text))
    return embeddings


# 解析 PDF 並嵌入內容
def index_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    df = pd.DataFrame(table)

                    # 初始化一個字典來保存每一列的上一次有效值
                    last_valid_values = {}
                    
                    for _, row in df.iterrows():
                        # 遍歷每一個單元格，如果是 None，替換為上一行的相應值
                        for col_index, cell in enumerate(row):
                            if cell is None or str(cell).strip() == '':  # 如果該單元格為 None 或空
                                if col_index in last_valid_values:  # 如果該列有上一次的有效值
                                    row[col_index] = last_valid_values[col_index]  # 用上一行的有效值填充
                            else:
                                # 更新該列的上一次有效值
                                last_valid_values[col_index] = cell
                        
                        # 將表格行轉換為文字，並處理空值為空字符串
                        row_text = ' | '.join([str(cell).replace('\n', '').strip() for cell in row])

                        if row_text.strip():  # 確保行不為空
                            embedding = embedder.encode(row_text, convert_to_tensor=True)
                            save_embedding_to_db(row_text, embedding)
            else:
                text = page.extract_text()
                if text : 
                    sentences = text.split('\n')  
                    for sentence in sentences:
                        if sentence.strip():  
                            embedding = embedder.encode(sentence, convert_to_tensor=True)
                            save_embedding_to_db(sentence, embedding)

def retrieve_context(query_embedding, top_k=10):
    """根據嵌入向量檢索最相似的內容"""
    # 檢查 query_embedding 所在設備
    device = query_embedding.device  # 獲取 query_embedding 的設備（cpu 或 cuda:0）

    loaded_index = load_embeddings_from_db()
    hits = sorted(
        loaded_index, 
        key=lambda x: util.cos_sim(
            query_embedding.to(device),  # 確保 query_embedding 在相同設備
            torch.tensor(x[0], dtype=torch.float32).to(device)  # 將嵌入轉移到相同設備
        ), 
        reverse=True
    )

    print([hit[1] for hit in hits[:top_k]])
    return [hit[1] for hit in hits[:top_k]]

def get_embedding(text):
    """使用 bge-m3 模型獲取文本的嵌入向量"""
    text = text.replace("\n", " ")
    embedding = embedder.encode(text, convert_to_tensor=True).to(embedder.device)
    return embedding


# 调用 TAIDE 模型生成响应
def TAIDEchat(sInput):

    generation_args = {
        "max_new_tokens": 1000,
        "return_full_text": False,
        "temperature": 0.3,
        "do_sample": True,
    }
        
    # 將用戶輸入轉換為嵌入向量
    query_embedding = get_embedding(sInput)

    # 檢索最相似的內容
    context = retrieve_context(query_embedding)
    context_text = "\n".join(context)
        
    # 將檢索到的內容添加為系統訊息    
    messages = [{"role": "system", "content": context_text}, {"role": "user", "content": sInput}]
        
    # 生成模型回應
    output = pipe(messages, **generation_args)
    model_response = output[0]['generated_text']

    return model_response

warnings.filterwarnings("ignore")
# 設定 Hugging Face API 令牌
os.environ["HUGGINGFACE_HUB_TOKEN"] = "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"  # 用你的 Hugging Face API 令牌替換這個值

# 或者直接在代碼中使用 login 函數
from huggingface_hub import login
login("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")

# 初始化 LLM 模型
device = 0 if torch.cuda.is_available() else -1  # 0 代表 GPU，-1 代表 CPU
model = AutoModelForCausalLM.from_pretrained("taide/TAIDE-LX-7B-Chat", load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained("taide/TAIDE-LX-7B-Chat", use_fast=False)
# 創建生成管道，並設置設備
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

if __name__ == '__main__':
    app.run(port=5000)
