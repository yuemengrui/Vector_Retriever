# *_*coding:utf-8 *_*
# @Author : YueMengRui
import json
import gradio as gr
import shutil
import requests
from config import *
from copy import deepcopy
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils import get_llm_name_list, get_embedding_model_name_list, get_llm_answer, md5hex, get_embeddings
from local_knowledge_handlers.local_knowledge_vector_store import KnowledgeVectorStore

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

    file_list = [x.split('.')[0] for x in os.listdir(vector_dir) if x != 'name_hash.json' and len(x) == 32]

    if len(file_list) == 0:
        return []

    with open(os.path.join(vector_dir, 'name_hash.json'), 'r') as f:
        name_hash_data = json.load(f)

    return [name_hash_data[x] for x in file_list]


def get_answer(query, knowledge_files, history, embedding_model, llm, prompt_template, max_length, max_prompt_length,
               top_p, temperature, llm_history_len, custom_configs, knowledge_chunk_size, knowledge_chunk_connect,
               vector_search_top_k, knowledge_score_threshold, stream=True):
    print({'custom_configs': custom_configs})
    try:
        custom_generation_configs = json.loads(custom_configs)
    except Exception as e:
        print(e)
        custom_generation_configs = {}

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

        prompt, related_docs = knowledge_vector_store.generate_knowledge_based_prompt(
            embedding_model_name=embedding_model,
            query=query,
            vector_store_dir_list=vector_store_dir_list,
            prompt_template=prompt_template if prompt_template else PROMPT_TEMPLATE,
            max_prompt_length=max_prompt_length,
            knowledge_chunk_size=knowledge_chunk_size,
            knowledge_chunk_connect=True if knowledge_chunk_connect == '启用' else False,
            vector_search_top_k=vector_search_top_k,
            score_threshold=knowledge_score_threshold)
        print(prompt)
        for doc in related_docs:
            sources.append({'file_name': name_hash_data[doc.metadata['file_hash']], 'score': doc.metadata['score'],
                            'text': doc.page_content})

    generation_configs = {
        "max_length": max_length,
        "max_prompt_length": max_prompt_length,
        "top_p": top_p,
        "temperature": temperature
    }

    generation_configs.update(custom_generation_configs)

    print('generation_configs: ', generation_configs)

    if stream:
        history.append("")
        for resp, usage in get_llm_answer(model_name=llm, prompt=prompt,
                                          history=[] if llm_history_len == 0 else llm_history[-llm_history_len:],
                                          generation_configs=generation_configs, stream=stream):
            resp[-1][0] = query
            llm_history.append([prompt, deepcopy(resp[-1][1])])
            source = f"""\n\n<summary style="font-weight:bold">{usage}</summary>\n\n"""
            if sources:
                source += "".join([
                    f"""<details><summary style="font-weight:bold;white-space:no-wrap;">出处 [{doc['file_name']}]---[score:{doc['score']}]</summary>\n{doc['text']}\n</details>"""
                    for doc in sources])

            resp[-1][1] += source

            # print('llm_history: ', llm_history)
            history[-1] = resp[-1]

            yield history, ""
    else:
        resp, usage = get_llm_answer(model_name=llm, prompt=prompt,
                                     history=[] if llm_history_len == 0 else llm_history[-llm_history_len:],
                                     generation_configs=generation_configs, stream=stream)
        resp[-1][0] = query
        llm_history.append([prompt, deepcopy(resp[-1][1])])
        source = f"""\n\n<summary style="font-weight:bold">{usage}</summary>\n\n"""
        if sources:
            source += "".join([
                f"""<details><summary style="font-weight:bold;white-space:no-wrap;">出处 [{doc['file_name']}]---[score:{doc['score']}]</summary>\n{doc['text']}\n</details>"""
                for doc in sources])

        resp[-1][1] += source

        print('llm_history: ', llm_history)
        history.append(resp[-1])

        return history, ""


def change_embedding_model(model_name, history):
    print(model_name)
    return gr.update(visible=False if model_name == "无" else True), \
        gr.update(choices=get_vs_list(model_name), value=[], visible=False if model_name == "无" else True), history


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


def clear_history():
    global llm_history
    llm_history = []
    return []


def calculate_similarity(sent1, sent2, embedding_model, chatbot):
    embeddings = get_embeddings(embedding_model, [sent1, sent2])

    score = \
        cosine_similarity(np.array([embeddings[0]], dtype=np.float32), np.array([embeddings[1]], dtype=np.float32))[0][
            0]
    answer = f"""
    <span style="font-weight:bold">文本1: </span>{sent1}\n\n
    <span style="font-weight:bold">文本2: </span>{sent2}\n\n
    <span style="font-weight:bold">{embedding_model}</span> ------ <span style="color: #f00;">[{score}]</span>\n\n
    """
    chatbot.append([None, answer])
    return chatbot


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
                                     show_label=False).style(height=666)
                query = gr.Textbox(show_label=False,
                                   placeholder="请输入提问内容，按回车进行提交").style(container=False)
                history_clear = gr.Button("清除历史记录", visible=True)

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
    with gr.Tab("hyper-parameters"):
        prompt_template = gr.Textbox(label="Prompt Template")
        with gr.Row():
            with gr.Column(scale=5):
                gr.Markdown("Model configs")
                max_length = gr.Number(value=4096, precision=0,
                                       label="模型生成token的最大长度",
                                       interactive=True)
                max_prompt_length = gr.Number(value=3096, precision=0,
                                              label="prompt的最大长度",
                                              interactive=True)
                top_p = gr.Slider(0.01, 1,
                                  value=0.8,
                                  step=0.01,
                                  label="top_p",
                                  interactive=True)
                temperature = gr.Slider(0.01, 1,
                                        value=0.8,
                                        step=0.01,
                                        label="temperature",
                                        interactive=True)
                llm_history_len = gr.Slider(0, 10,
                                            value=10,
                                            step=1,
                                            label="LLM 对话轮数",
                                            interactive=True)
                custom_configs = gr.Code(value=str({}), label="more_configs with json format", language="json",
                                         interactive=True)
            with gr.Column(scale=5):
                gr.Markdown("Knowledge configs")
                vector_search_top_k = gr.Slider(1, 20, value=10, step=1,
                                                label="向量匹配 top k", interactive=True)
                knowledge_score_threshold = gr.Slider(0, 1, value=0.5, step=0.01,
                                                      label="向量匹配过滤阈值", interactive=True)
                knowledge_chunk_connect = gr.Radio(["启用", "不启用"],
                                                   value="启用",
                                                   label="是否启用上下文关联")
                knowledge_chunk_size = gr.Number(value=512, precision=0,
                                                 label="关联文本上下文最大长度",
                                                 interactive=True)

    query.submit(get_answer,
                 [query, knowledge_file_list, chatbot, embedding_model, llm, prompt_template, max_length,
                  max_prompt_length, top_p, temperature, llm_history_len, custom_configs, knowledge_chunk_size,
                  knowledge_chunk_connect, vector_search_top_k, knowledge_score_threshold],
                 [chatbot, query])
    history_clear.click(fn=clear_history,
                        outputs=[chatbot])

    with gr.Tab("文本相似度测试"):
        with gr.Row():
            with gr.Column(scale=10):
                chatbot = gr.Chatbot([[None,
                                       f"""欢迎使用 文本相似度测试！\n\n请在右侧选择embedding模型，然后在下方输入需要计算相似度的文本对"""]],
                                     elem_id="chat-box",
                                     show_label=False).style(height=500)
                sent1 = gr.Textbox(show_label=False,
                                   placeholder="文本1")
                sent2 = gr.Textbox(show_label=False,
                                   placeholder="文本2")
                calculate = gr.Button("计算相似度", visible=True)
                history_clear = gr.Button("清除历史记录", visible=True)

            with gr.Column(scale=3):
                embedding_model = gr.Radio(embedding_model_name_list,
                                           label="Embedding Models",
                                           value=embedding_model_name_list[1])

            calculate.click(fn=calculate_similarity,
                            inputs=[sent1, sent2, embedding_model, chatbot],
                            outputs=[chatbot])
            history_clear.click(fn=lambda: [],
                                outputs=[chatbot])

(demo
 .queue(concurrency_count=1)
 .launch(server_name='0.0.0.0',
         server_port=7860,
         show_api=False,
         share=False,
         inbrowser=False))
