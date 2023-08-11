# *_*coding:utf-8 *_*
# @Author : YueMengRui
import os

OCR_URL = 'http://127.0.0.1:6000/ai/ocr/general'

EMBEDDING_MODEL_NAME_LIST_URL = 'http://127.0.0.1:6000/ai/embedding/model/list'
TEXT_EMBEDDING_URL = 'http://127.0.0.1:6000/ai/embedding/text'

LLM_MODEL_NAME_LIST_URL = 'http://127.0.0.1:6000/ai/llm/list'
LLM_CHAT_URL = 'http://127.0.0.1:6000/ai/llm/chat'

VECTOR_STORE_ROOT_DIR = './VectorStores'

if not os.path.exists(VECTOR_STORE_ROOT_DIR):
    os.makedirs(VECTOR_STORE_ROOT_DIR)

PROMPT_TEMPLATE = """你是一个出色的文档问答助手，根据给定的文本片段和问题进行回答，仔细分析和思考每个文本片段的内容，回答要合理、简洁，直接回复答案，回复语言采用中文。
使用下面的文本片段列表，回答问题：{query}

{context}
"""

# 若能找到对应答案，答案以'根据文档知识'开头。若无法找到对应答案，则使用外部知识进行回答，答案以'根据外部知识'开头。
