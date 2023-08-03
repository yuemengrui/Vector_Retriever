# *_*coding:utf-8 *_*
# @Author : YueMengRui
import json
import gradio as gr
import shutil
import requests
from config import *
from copy import deepcopy
import os
from utils import get_llm_name_list, get_embedding_model_name_list, get_llm_answer, md5hex
from local_knowledge_handlers.local_knowledge_vector_store import KnowledgeVectorStore
import sseclient

# init
knowledge_vector_store = KnowledgeVectorStore()
embedding_model_name_list = get_embedding_model_name_list()
llm_name_list = get_llm_name_list()

llm_history = []


def get_vs_list(embedding_model):
    print('get_vs_list: ', embedding_model)
    vector_dir = os.path.join(VECTOR_STORE_ROOT_DIR, embedding_model)
    if not os.path.exists(vector_dir):
        os.makedirs(vector_dir)
        return []

    file_list = [x.split('.')[0] for x in os.listdir(vector_dir) if x != 'name_hash.json']

    if len(file_list) == 0:
        return []

    with open(os.path.join(vector_dir, 'name_hash.json'), 'r') as f:
        name_hash_data = json.load(f)

    return [name_hash_data[x] for x in file_list]


def get_answer(query, knowledge_files, history, embedding_model, llm, **kwargs):
    if len(llm_history) == 0:
        for h in history:
            if h[0] is not None:
                llm_history.append(h)

    print('get_answer_knowledge_files: ', knowledge_files)
    prompt = query
    sources = []
    if knowledge_files:
        with open(os.path.join(os.path.join(VECTOR_STORE_ROOT_DIR, embedding_model), 'name_hash.json'), 'r') as f:
            name_hash_data = json.load(f)

        vector_store_hash_list = []
        for k_file in knowledge_files:
            for h, n in name_hash_data.items():
                if n == k_file:
                    vector_store_hash_list.append(h)

        vector_store_hash_list = list(set(vector_store_hash_list))
        vector_store_dir_list = [os.path.join(os.path.join(VECTOR_STORE_ROOT_DIR, embedding_model), x) for x in
                                 vector_store_hash_list]
        print(vector_store_dir_list)

        prompt, related_docs = knowledge_vector_store.generate_knowledge_based_prompt(
            embedding_model_name=embedding_model,
            query=query,
            vector_store_dir_list=vector_store_dir_list,
            prompt_template=PROMPT_TEMPLATE,
            **kwargs)
        print(prompt)
        print(related_docs)
        for doc in related_docs:
            sources.append({'file_name': name_hash_data[doc.metadata['file_hash']], 'score': doc.metadata['score'],
                            'text': doc.page_content})

    resp = get_llm_answer(model_name=llm, prompt=prompt, history=llm_history)
    resp[-1][0] = query
    llm_history.append([prompt, deepcopy(resp[-1][1])])
    if sources:
        source = "\n\n"
        source += "".join([
            f"""<details><summary style="font-weight:bold;white-space:no-wrap;">出处 [{doc['file_name']}]{"  "}[score:{doc['score']}]</summary>\n{doc['text']}\n</details>"""
            for doc in sources])

        resp[-1][1] += source

    print('llm_history: ', llm_history)
    history.append(resp[-1])

    return history, ""


def change_embedding_model(model_name, history):
    return gr.update(visible=True), gr.update(choices=get_vs_list(model_name), visible=True), history


def upload_knowledge_file(embedding_model, files, chatbot):
    print('upload_knowledge_file: ', embedding_model)
    if isinstance(files, list):
        for file in files:
            filename = os.path.split(file.name)[-1]
            vs_status = f"""新增向量知识库 {filename} 失败，请重新上传"""
            with open(file.name, 'rb') as f:
                file_data = f.read()
            file_hash = md5hex(file_data)
            if file_hash != '':
                if knowledge_vector_store.build_vector_store(filepath=file.name, file_hash=file_hash,
                                                             vector_dir=os.path.join(VECTOR_STORE_ROOT_DIR,
                                                                                     embedding_model),
                                                             embedding_model_name=embedding_model):
                    name_hash_map_file = os.path.join(os.path.join(VECTOR_STORE_ROOT_DIR, embedding_model),
                                                      'name_hash.json')
                    if os.path.exists(name_hash_map_file):
                        with open(name_hash_map_file, 'r') as ff:
                            name_hash_data = json.load(ff)

                        name_hash_data.update({file_hash: filename})
                    else:
                        name_hash_data = {file_hash: filename}

                    with open(name_hash_map_file, 'w') as fff:
                        json.dump(name_hash_data, fff, ensure_ascii=False, indent=2)

                    vs_status = f"""新增向量知识库 {filename} 成功"""

            chatbot = chatbot + [[None, vs_status]]
    return gr.update(choices=get_vs_list(embedding_model), visible=True), chatbot


def delete_knowledge_file(embedding_model, knowledge_file_list, chatbot):
    print('delete_knowledge_file: ', knowledge_file_list)
    with open(os.path.join(os.path.join(VECTOR_STORE_ROOT_DIR, embedding_model), 'name_hash.json'), 'r') as f:
        name_hash_data = json.load(f)

    for k_file in knowledge_file_list:
        for h, n in name_hash_data.items():
            if n == k_file:
                break
        try:
            shutil.rmtree(os.path.join(os.path.join(VECTOR_STORE_ROOT_DIR, embedding_model), h))
            del name_hash_data[h]
            status = f"删除知识库 {k_file} 成功"
        except Exception as e:
            print(e)
            status = f"删除知识库 {k_file} 失败"
        chatbot = chatbot + [[None, status]]

    with open(os.path.join(os.path.join(VECTOR_STORE_ROOT_DIR, embedding_model), 'name_hash.json'), 'w') as ff:
        json.dump(name_hash_data, ff, ensure_ascii=False, indent=2)

    return gr.update(choices=get_vs_list(embedding_model), visible=True), chatbot


########################################################################################

block_css = """.importantButton {
    background: linear-gradient(45deg, #7e0570,#5d1c99, #6e00ff) !important;
    border: none !important;
}
.message{padding : 8px !important;}
.importantButton:hover {
    background: linear-gradient(45deg, #ff00e0,#8500ff, #6e00ff) !important;
    border: none !important;
}"""

webui_title = """
# 🎉Vector Retriever Web UI🎉 👍 
"""
init_message = f"""欢迎使用 Vector Retriever Web UI！

请在右侧选择LLM模型和embedding模型，目前支持直接与 LLM 模型对话或基于本地知识库问答。

"""

default_theme_args = dict(
    font=["Source Sans Pro", 'ui-sans-serif', 'system-ui', 'sans-serif'],
    font_mono=['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace'],
)

with gr.Blocks(css=block_css, theme=gr.themes.Default(**default_theme_args)) as demo:
    gr.Markdown(webui_title)
    with gr.Tab("向量检索"):
        with gr.Row():
            with gr.Column(scale=10):
                chatbot = gr.Chatbot([[None, init_message]],
                                     elem_id="chat-box",
                                     show_label=False).style(height=700)
                query = gr.Textbox(show_label=False,
                                   placeholder="请输入提问内容，按回车进行提交").style(container=False)

            with gr.Column(scale=3):
                llm = gr.Radio(llm_name_list,
                               label="LLMs",
                               value=llm_name_list[0])
                embedding_model = gr.Radio(embedding_model_name_list,
                                           label="Embedding Models",
                                           value="")

                vs_list = gr.Accordion("知识文件列表", visible=False)
                with vs_list:
                    knowledge_file_list = gr.CheckboxGroup([], show_label=False, visible=True)
                    knowledge_file_delete = gr.Button("删除知识文件", visible=True)

                embedding_model.change(fn=change_embedding_model,
                                       inputs=[embedding_model, chatbot],
                                       outputs=[vs_list, knowledge_file_list, chatbot])

                with vs_list:
                    file2vs = gr.Column(visible=True)

                    with file2vs:
                        sentence_size = gr.Number(value=512, precision=0,
                                                  label="文本入库分句长度限制",
                                                  interactive=True, visible=False)
                        with gr.Tab("上传知识文件"):
                            files = gr.File(label="添加文件",
                                            file_types=['.txt', '.md', '.docx', '.pdf', '.png', '.jpg', ".csv"],
                                            file_count="multiple",
                                            show_label=False)
                            knowledge_file_upload = gr.Button(value="上传知识文件", visible=True)

                            knowledge_file_upload.click(fn=upload_knowledge_file,
                                                        inputs=[embedding_model, files, chatbot],
                                                        outputs=[knowledge_file_list, chatbot])
                    knowledge_file_delete.click(fn=delete_knowledge_file,
                                                inputs=[embedding_model, knowledge_file_list, chatbot],
                                                outputs=[knowledge_file_list, chatbot])

                query.submit(get_answer,
                             [query, knowledge_file_list, chatbot, embedding_model, llm],
                             [chatbot, query])

(demo
 .queue(concurrency_count=1)
 .launch(server_name='0.0.0.0',
         server_port=7860,
         show_api=False,
         share=False,
         inbrowser=False))
