# *_*coding:utf-8 *_*
# @Author : YueMengRui
"""Wrapper around FAISS vector database."""
from __future__ import annotations

import math
import os
import pickle
import uuid
import warnings
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Callable,
)

import numpy as np
from copy import deepcopy
from langchain.docstore.base import AddableMixin, Docstore
from langchain.docstore.document import Document
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.vectorstores.utils import maximal_marginal_relevance


def dependable_faiss_import(no_avx2: Optional[bool] = None) -> Any:
    """
    Import faiss if available, otherwise raise error.
    If FAISS_NO_AVX2 environment variable is set, it will be considered
    to load FAISS with no AVX2 optimization.

    Args:
        no_avx2: Load FAISS strictly with no AVX2 optimization
            so that the vectorstore is portable and compatible with other devices.
    """
    if no_avx2 is None and "FAISS_NO_AVX2" in os.environ:
        no_avx2 = bool(os.getenv("FAISS_NO_AVX2"))

    try:
        if no_avx2:
            from faiss import swigfaiss as faiss
        else:
            import faiss
    except ImportError:
        raise ValueError(
            "Could not import faiss python package. "
            "Please install it with `pip install faiss` "
            "or `pip install faiss-cpu` (depending on Python version)."
        )
    return faiss


def _default_relevance_score_fn(score: float) -> float:
    """Return a similarity score on a scale [0, 1]."""
    # The 'correct' relevance function
    # may differ depending on a few things, including:
    # - the distance / similarity metric used by the VectorStore
    # - the scale of your embeddings (OpenAI's are unit normed. Many others are not!)
    # - embedding dimensionality
    # - etc.
    # This function converts the euclidean norm of normalized embeddings
    # (0 is most similar, sqrt(2) most dissimilar)
    # to a similarity function (0 to 1)
    return 1.0 - score / math.sqrt(2)


class MyFAISS:
    """Wrapper around FAISS vector database.

    To use, you should have the ``faiss`` python package installed.

    """

    def __init__(
            self,
            index: Any,
            docstore: Docstore,
            index_to_docstore_id: Dict[int, str],
            relevance_score_fn: Optional[
                Callable[[float], float]
            ] = _default_relevance_score_fn,
            normalize_L2: bool = False,
    ):
        """Initialize with necessary components."""
        self.index = index
        self.docstore = docstore
        self.index_to_docstore_id = index_to_docstore_id
        self.relevance_score_fn = relevance_score_fn
        self._normalize_L2 = normalize_L2

    def similarity_search_with_score_by_vector(
            self,
            embedding: List[float],
            k: int = 4,
            chunk_size=512,
            chunk_conent=True,
            **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            embedding: Embedding vector to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, Any]]): Filter by metadata. Defaults to None.
            fetch_k: (Optional[int]) Number of Documents to fetch before filtering.
                      Defaults to 20.
            **kwargs: kwargs to be passed to similarity search. Can include:
                score_threshold: Optional, a floating point value between 0 to 1 to
                    filter the resulting set of retrieved docs

        Returns:
            List of documents most similar to the query text and L2 distance
            in float for each. Lower score represents more similarity.
        """
        faiss = dependable_faiss_import()
        vector = np.array([embedding], dtype=np.float32)
        if self._normalize_L2:
            faiss.normalize_L2(vector)
        scores, indices = self.index.search(vector, k)
        print(scores)
        docs = []
        id_set = set()
        store_len = len(self.index_to_docstore_id)
        for j, i in enumerate(indices[0]):
            if i == -1:
                # This happens when not enough docs are returned.
                continue
            _id = self.index_to_docstore_id[i]
            doc = self.docstore.search(_id)
            if not chunk_conent:
                if not isinstance(doc, Document):
                    raise ValueError(f"Could not find document for id {_id}, got {doc}")
                doc.metadata["score"] = 1 - scores[0][j]
                docs.append(doc)
                continue
            id_set.add(i)
            docs_len = len(doc.page_content)
            # chunk_size = max(0.0, knowledge_chunk_size * (
            #         (knowledge_score_threshold - max(knowledge_good_score, int(scores[0][j]))) / (
            #         knowledge_score_threshold - knowledge_good_score)))
            #
            # if logger:
            #     logger.info(str({'knowledge_chunk_size': knowledge_chunk_size,
            #                      'knowledge_score_threshold': knowledge_score_threshold,
            #                      'knowledge_good_score': knowledge_good_score, 'score': int(scores[0][j]),
            #                      'chunk_size': chunk_size}) + '\n')

            for n in range(1, max(i, store_len - i)):
                break_flag = False
                for l in [i + n, i - n]:
                    if 0 <= l < len(self.index_to_docstore_id):
                        _id0 = self.index_to_docstore_id[l]
                        doc0 = self.docstore.search(_id0)
                        if docs_len + len(doc0.page_content) > chunk_size:
                            break_flag = True
                            break
                        elif doc0.metadata["source"] == doc.metadata["source"]:
                            docs_len += len(doc0.page_content)
                            id_set.add(l)
                if break_flag:
                    break
        if not chunk_conent:
            return docs
        if len(id_set) == 0:
            return []
        id_list = sorted(list(id_set))
        id_lists = self.seperate_list(id_list)
        for id_seq in id_lists:
            for id in id_seq:
                if id == id_seq[0]:
                    _id = self.index_to_docstore_id[id]
                    doc = self.docstore.search(_id)
                else:
                    _id0 = self.index_to_docstore_id[id]
                    doc0 = self.docstore.search(_id0)
                    doc.page_content += " " + doc0.page_content
            if not isinstance(doc, Document):
                raise ValueError(f"Could not find document for id {_id}, got {doc}")
            doc_score = min([scores[0][id] for id in [indices[0].tolist().index(i) for i in id_seq if i in indices[0]]])
            doc.metadata["score"] = 1 - doc_score
            docs.append(doc)

        print(docs)
        score_threshold = kwargs.get("score_threshold", 0.45)
        if score_threshold is not None:
            docs = [doc for doc in docs if doc.metadata['score'] >= score_threshold]

        docs.sort(key=lambda x:x.metadata['score'], reverse=True)
        return docs

    def max_marginal_relevance_search_by_vector(
            self,
            embedding: List[float],
            k: int = 4,
            fetch_k: int = 20,
            lambda_mult: float = 0.5,
            filter: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch before filtering to
                     pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        _, indices = self.index.search(
            np.array([embedding], dtype=np.float32),
            fetch_k if filter is None else fetch_k * 2,
        )
        if filter is not None:
            filtered_indices = []
            for i in indices[0]:
                if i == -1:
                    # This happens when not enough docs are returned.
                    continue
                _id = self.index_to_docstore_id[i]
                doc = self.docstore.search(_id)
                if not isinstance(doc, Document):
                    raise ValueError(f"Could not find document for id {_id}, got {doc}")
                if all(doc.metadata.get(key) == value for key, value in filter.items()):
                    filtered_indices.append(i)
            indices = np.array([filtered_indices])
        # -1 happens when not enough docs are returned.
        embeddings = [self.index.reconstruct(int(i)) for i in indices[0] if i != -1]
        mmr_selected = maximal_marginal_relevance(
            np.array([embedding], dtype=np.float32),
            embeddings,
            k=k,
            lambda_mult=lambda_mult,
        )
        selected_indices = [indices[0][i] for i in mmr_selected]
        docs = []
        for i in selected_indices:
            if i == -1:
                # This happens when not enough docs are returned.
                continue
            _id = self.index_to_docstore_id[i]
            doc = self.docstore.search(_id)
            if not isinstance(doc, Document):
                raise ValueError(f"Could not find document for id {_id}, got {doc}")
            docs.append(doc)
        return docs

    def merge_from(self, target: MyFAISS) -> None:
        """Merge another FAISS object with the current one.

        Add the target FAISS to the current one.

        Args:
            target: FAISS object you wish to merge into the current one

        Returns:
            None.
        """
        if not isinstance(self.docstore, AddableMixin):
            raise ValueError("Cannot merge with this type of docstore")
        # Numerical index for target docs are incremental on existing ones
        starting_len = len(self.index_to_docstore_id)

        # Merge two IndexFlatL2
        self.index.merge_from(target.index)

        # Get id and docs from target FAISS object
        full_info = []
        for i, target_id in target.index_to_docstore_id.items():
            doc = target.docstore.search(target_id)
            if not isinstance(doc, Document):
                raise ValueError("Document should be returned")
            full_info.append((starting_len + i, target_id, doc))

        # Add information to docstore and index_to_docstore_id.
        self.docstore.add({_id: doc for _, _id, doc in full_info})
        index_to_id = {index: _id for index, _id, _ in full_info}
        self.index_to_docstore_id.update(index_to_id)

    @classmethod
    def __from(
            cls,
            texts: List[str],
            embeddings: List[List[float]],
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            normalize_L2: bool = False,
            **kwargs: Any,
    ) -> MyFAISS:
        faiss = dependable_faiss_import()
        index = faiss.IndexFlatL2(len(embeddings[0]))
        vector = np.array(embeddings, dtype=np.float32)
        if normalize_L2:
            faiss.normalize_L2(vector)
        index.add(vector)
        documents = []
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            documents.append(Document(page_content=text, metadata=metadata))
        index_to_id = dict(enumerate(ids))
        docstore = InMemoryDocstore(dict(zip(index_to_id.values(), documents)))
        return cls(
            index,
            docstore,
            index_to_id,
            normalize_L2=normalize_L2,
            **kwargs,
        )

    @classmethod
    def from_texts(
            cls,
            texts: List[str],
            embeddings,
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> MyFAISS:

        return cls.__from(
            texts,
            embeddings,
            metadatas=metadatas,
            ids=ids,
            **kwargs,
        )

    def save_local(self, folder_path: str, index_name: str = "index") -> None:
        """Save FAISS index, docstore, and index_to_docstore_id to disk.

        Args:
            folder_path: folder path to save index, docstore,
                and index_to_docstore_id to.
            index_name: for saving with a specific index file name
        """
        path = Path(folder_path)
        path.mkdir(exist_ok=True, parents=True)

        # save index separately since it is not picklable
        faiss = dependable_faiss_import()
        faiss.write_index(
            self.index, str(path / "{index_name}.faiss".format(index_name=index_name))
        )

        # save docstore and index_to_docstore_id
        with open(path / "{index_name}.pkl".format(index_name=index_name), "wb") as f:
            pickle.dump((self.docstore, self.index_to_docstore_id), f)

    @classmethod
    def load_local(
            cls, folder_path: str, index_name: str = "index"
    ) -> MyFAISS:
        """Load FAISS index, docstore, and index_to_docstore_id from disk.

        Args:
            folder_path: folder path to load index, docstore,
                and index_to_docstore_id from.
            embeddings: Embeddings to use when generating queries
            index_name: for saving with a specific index file name
        """
        path = Path(folder_path)
        # load index separately since it is not picklable
        faiss = dependable_faiss_import()
        index = faiss.read_index(
            str(path / "{index_name}.faiss".format(index_name=index_name))
        )

        # load docstore and index_to_docstore_id
        with open(path / "{index_name}.pkl".format(index_name=index_name), "rb") as f:
            docstore, index_to_docstore_id = pickle.load(f)
        return cls(index, docstore, index_to_docstore_id)

    def _similarity_search_with_relevance_scores(
            self,
            embedding: List[float],
            k: int = 4,
            filter: Optional[Dict[str, Any]] = None,
            fetch_k: int = 20,
            **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs and their similarity scores on a scale from 0 to 1."""
        if self.relevance_score_fn is None:
            raise ValueError(
                "normalize_score_fn must be provided to"
                " FAISS constructor to normalize scores"
            )
        docs_and_scores = self.similarity_search_with_score_by_vector(
            embedding,
            k=k,
            filter=filter,
            fetch_k=fetch_k,
            **kwargs,
        )
        for doc in docs_and_scores:
            doc.metadata.update({"score": self.relevance_score_fn(doc.metadata['score'])})
        return docs_and_scores

    def similarity_search_with_relevance_scores(
            self,
            embedding: List[float],
            k: int = 4,
            **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs and relevance scores in the range [0, 1].

        0 is dissimilar, 1 is most similar.

        Args:
            query: input text
            k: Number of Documents to return. Defaults to 4.
            **kwargs: kwargs to be passed to similarity search. Should include:
                score_threshold: Optional, a floating point value between 0 to 1 to
                    filter the resulting set of retrieved docs

        Returns:
            List of Tuples of (doc, similarity_score)
        """
        docs_and_similarities = self._similarity_search_with_relevance_scores(
            embedding, k=k, **kwargs
        )
        if any(
                similarity < 0.0 or similarity > 1.0
                for _, similarity in docs_and_similarities
        ):
            warnings.warn(
                "Relevance scores must be between"
                f" 0 and 1, got {docs_and_similarities}"
            )

        score_threshold = kwargs.get("score_threshold")
        if score_threshold is not None:
            docs_and_similarities = [
                (doc, similarity)
                for doc, similarity in docs_and_similarities
                if similarity >= score_threshold
            ]
            if len(docs_and_similarities) == 0:
                warnings.warn(
                    "No relevant docs were retrieved using the relevance score"
                    f" threshold {score_threshold}"
                )
        return docs_and_similarities

    @classmethod
    def from_documents(
            cls,
            documents: List[Document],
            embeddings,
            **kwargs: Any,
    ):
        """Return VectorStore initialized from documents and embeddings."""
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        return cls.from_texts(texts, embeddings, metadatas=metadatas, **kwargs)

    def seperate_list(self, ls: List[int]) -> List[List[int]]:
        lists = []
        ls1 = [ls[0]]
        file_hash = self.docstore.search(self.index_to_docstore_id[ls1[-1]]).metadata['file_hash']
        for i in range(1, len(ls)):
            _file_hash = self.docstore.search(self.index_to_docstore_id[ls[i]]).metadata['file_hash']
            if _file_hash == file_hash:
                ls1.append(ls[i])
            else:
                lists.append(deepcopy(ls1))
                ls1 = [ls[i]]
                file_hash = _file_hash

        lists.append(ls1)
        return lists
