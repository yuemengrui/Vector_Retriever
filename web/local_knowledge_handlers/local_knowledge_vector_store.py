# *_*coding:utf-8 *_*
# @Author : YueMengRui
import os
import hashlib
import numpy as np
from copy import deepcopy
from typing import List, Tuple
from loader.file_loader import load_file
from vector_stores.my_faiss import MyFAISS
from utils import get_embeddings


class KnowledgeVectorStore:

    def __init__(self, logger=None):
        self.logger = logger

    def write_log(self, msg):
        if self.logger:
            self.logger.info(str(msg) + '\n')
        else:
            print(str(msg) + '\n')

    def build_vector_store(self, filepath: str, file_hash: str, vector_dir: str, embedding_model_name: str):

        if os.path.exists(os.path.join(vector_dir, file_hash)):
            self.write_log({"file hash exist": file_hash})
            return True

        if not os.path.exists(filepath):
            self.write_log({"file load error": "路径不存在"})
            return False
        elif os.path.isfile(filepath):
            file = os.path.split(filepath)[-1]
            # try:
            docs = load_file(filepath)
            for doc in docs:
                doc.metadata.update({'file_hash': file_hash})
            self.write_log({'load_doc': docs})
            sentences = [d.page_content for d in docs]
            embeddings = get_embeddings(embedding_model_name, sentences)
            vector_store_dir = os.path.join(vector_dir, file_hash)
            vector_store = MyFAISS.from_documents(docs, embeddings=embeddings)
            vector_store.save_local(vector_store_dir)
            self.write_log({"file load": "{}已成功加载".format(file)})
            return True
            # except Exception as e:
            #     self.write_log({'file load error': '{}未能成功加载: {}'.format(file, str(e))})
            #     return False
        else:
            return False

    def get_related_docs(self, embedding, vector_store_dir_list, knowledge_chunk_size=512, knowledge_chunk_connect=True,
                         vector_search_top_k=10, **kwargs):

        if len(vector_store_dir_list) == 0:
            return []

        for i, vector_store_dir in enumerate(vector_store_dir_list):
            if i == 0:
                vector_store = MyFAISS.load_local(vector_store_dir)
            else:
                vector_store.merge_from(MyFAISS.load_local(vector_store_dir))

        if not isinstance(knowledge_chunk_size, (int, float)):
            knowledge_chunk_size = 512

        if not isinstance(knowledge_chunk_connect, bool):
            knowledge_chunk_connect = True

        if not isinstance(vector_search_top_k, int):
            vector_search_top_k = 10

        self.write_log(
            {'knowledge_chunk_size': knowledge_chunk_size, 'knowledge_chunk_connect': knowledge_chunk_connect,
             'vector_search_top_k': vector_search_top_k})

        related_docs = vector_store.similarity_search_with_score_by_vector(embedding, k=vector_search_top_k,
                                                                           chunk_size=knowledge_chunk_size,
                                                                           chunk_connect=knowledge_chunk_connect,
                                                                           **kwargs)

        return related_docs

    def generate_prompt(self, related_docs: List[str],
                        query: str,
                        max_prompt_len: int,
                        prompt_template=None) -> str:
        if not related_docs:
            return query, []

        self.write_log({'related_docs': related_docs})
        base_prompt_len = len(prompt_template.format(context='', query=query))
        true_related_docs = []

        for i in related_docs:
            if base_prompt_len + len(i.page_content) > max_prompt_len:
                break

            true_related_docs.append(i)
            base_prompt_len += len(i.page_content)

        if not true_related_docs:
            return query, []

        context = ''
        for ind, doc in enumerate(true_related_docs):
            context += '文本片段{}: {}\n'.format(str(ind + 1), doc.page_content)
            # context = "\n".join([doc.page_content for doc in true_related_docs])
        self.write_log({'context_len': len(context), 'context': context})
        prompt = prompt_template.format(context=context, query=query)

        return prompt, true_related_docs

    def generate_knowledge_based_prompt(self, embedding_model_name, query, vector_store_dir_list,
                                        max_prompt_length=3096,
                                        prompt_template=None,
                                        **kwargs):
        self.write_log({'max_prompt_length': max_prompt_length, 'prompt_template': prompt_template})
        self.write_log(kwargs)
        embedding = get_embeddings(embedding_model_name, [query])[0]
        related_docs = self.get_related_docs(embedding, vector_store_dir_list, **kwargs)

        knowledge_based_prompt, docs = self.generate_prompt(related_docs, query, max_prompt_length, prompt_template)

        return knowledge_based_prompt, docs
