# *_*coding:utf-8 *_*
# @Author : YueMengRui
import hashlib
import requests
from config import *


def md5hex(data):
    try:
        m = hashlib.md5()
        m.update(data)
        return str(m.hexdigest())
    except Exception as e:
        print(str({'EXCEPTION': e}) + '\n')
        return ''


def get_embedding_model_name_list():
    return requests.get(url=EMBEDDING_MODEL_NAME_LIST_URL).json()['data']['embedding_model_list']


def get_llm_name_list():
    return requests.get(url=LLM_MODEL_NAME_LIST_URL).json()['data']['llm_list']


def get_embeddings(model_name, sentences):
    print('get_embeddings: ', model_name)
    print('get_embeddings: ', sentences)
    data = {
        "model_name": model_name,
        "sentences": sentences
    }

    resp = requests.post(url=TEXT_EMBEDDING_URL, json=data)

    return resp.json()['data'][0]['embeddings']


def get_llm_answer(model_name, prompt, history, generation_configs={}):
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model_name": model_name,
        "queries": [
            {
                "prompt": prompt,
                "history": history,
                "generation_configs": generation_configs
            }
        ]
    }

    resp = requests.post(url=LLM_CHAT_URL, json=data, headers=headers, stream=False)
    print(resp.json())

    return resp.json()['data']['answers'][0]['history']
