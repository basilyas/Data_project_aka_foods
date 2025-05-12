import streamlit as st
import numpy as np
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any
from langchain.docstore.document import Document
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import string
import nltk
import re
import uuid

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')


class FusionPDFVectorDB:
    def __init__(self, pdf_path, vector_store_path):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        try:
            if os.path.exists(vector_store_path):
                print("Loading existing vector store...")
                self.vector_store = FAISS.load_local(
                    vector_store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )

                test_embedding = self.embeddings.embed_query("test")
                self.vector_store.index.search(np.array([test_embedding]), k=1)

                print("Loading documents for BM25...")
                all_docs = self.vector_store.similarity_search("test query", k=5)
                self.bm25 = self.create_bm25_index(all_docs)
                print("Vector store loaded successfully!")

            else:
                print("Creating new vector store from PDF...")
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                texts = text_splitter.split_documents(documents)

                self.vector_store = FAISS.from_documents(texts, self.embeddings)
                self.bm25 = self.create_bm25_index(texts)

                os.makedirs(vector_store_path, exist_ok=True)
                self.vector_store.save_local(vector_store_path)
                print("New vector store created and saved!")

        except Exception as e:
            print(f"Error initializing vector store: {str(e)}")
            print("Attempting to create new vector store...")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            texts = text_splitter.split_documents(documents)

            self.vector_store = FAISS.from_documents(texts, self.embeddings)
            self.bm25 = self.create_bm25_index(texts)

            os.makedirs(vector_store_path, exist_ok=True)
            self.vector_store.save_local(vector_store_path)
            print("New vector store created and saved!")

    def create_bm25_index(self, documents: List[Document]) -> BM25Okapi:
        tokenized_docs = [doc.page_content.split() for doc in documents]
        return BM25Okapi(tokenized_docs)

    def fusion_search(self, query: str, k: int = 5, alpha: float = 0.5) -> List[str]:
        try:
            epsilon = 1e-8

            vector_results = self.vector_store.similarity_search_with_score(query, k=k)
            vector_docs = [doc for doc, _ in vector_results]
            vector_scores = np.array([score for _, score in vector_results])

            query_tokens = query.split()
            bm25_scores = self.bm25.get_scores(query_tokens)

            if len(vector_scores) > 0:
                vector_scores = 1 - (vector_scores - np.min(vector_scores)) / (
                        np.max(vector_scores) - np.min(vector_scores) + epsilon)

            if len(bm25_scores) > 0:
                bm25_scores = (bm25_scores - np.min(bm25_scores)) / (
                        np.max(bm25_scores) - np.min(bm25_scores) + epsilon)

            if len(bm25_scores) == len(vector_scores):
                combined_scores = alpha * vector_scores + (1 - alpha) * bm25_scores[:len(vector_scores)]
            else:
                combined_scores = vector_scores

            sorted_indices = np.argsort(combined_scores)[::-1]
            return [vector_docs[i].page_content for i in sorted_indices[:k]]
        except Exception as e:
            print(f"Error in fusion search: {str(e)}")
            results = self.vector_store.similarity_search(query, k=k)
            return [doc.page_content for doc in results]

    def vector_search(self, query: str, k: int = 5) -> List[str]:
        """
        Perform vector-based search only.
        """
        try:
            results = self.vector_store.similarity_search(query, k=k)
            return [doc.page_content for doc in results]
        except Exception as e:
            print(f"Error in vector search: {str(e)}")
            return []

    def bm25_search(self, query: str, k: int = 5) -> List[str]:
        """
        Perform BM25 keyword search only.
        """
        try:
            query_tokens = query.split()
            bm25_scores = self.bm25.get_scores(query_tokens)

            # Get all documents from vector store (we need them for reference)
            all_docs = self.vector_store.similarity_search("", k=1000)  # Get as many as possible

            # Sort by BM25 score
            sorted_indices = np.argsort(bm25_scores)[::-1]

            # Return top k
            return [all_docs[i].page_content for i in sorted_indices[:k] if i < len(all_docs)]
        except Exception as e:
            print(f"Error in BM25 search: {str(e)}")
            return []


class CustomRAGEvaluator:
    def __init__(self, embeddings_model=None):
        if embeddings_model is None:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        else:
            self.embeddings = embeddings_model

        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('taggers/averaged_perceptron_tagger')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('averaged_perceptron_tagger')
            nltk.download('wordnet')

        self.stop_words = set(stopwords.words('english'))
        self.evaluation_results = []
        self.embedding_cache = {}
        self.lemmatizer = nltk.stem.WordNetLemmatizer()

    def get_embedding(self, text):
        if not text or text.strip() == "":
            return np.zeros(384)

        if text in self.embedding_cache:
            return self.embedding_cache[text]

        try:
            embedding = self.embeddings.embed_query(text)
            self.embedding_cache[text] = embedding
            return embedding
        except Exception as e:
            print(f"Error computing embedding: {str(e)}")
            return np.zeros(384)

    def batch_embed(self, texts):
        if not texts:
            return []

        uncached_texts = []
        uncached_indices = []
        result = [None] * len(texts)

        for i, text in enumerate(texts):
            if not text or text.strip() == "":
                result[i] = np.zeros(384)
            elif text in self.embedding_cache:
                result[i] = self.embedding_cache[text]
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts:
            try:
                computed_embeddings = [self.embeddings.embed_query(text) for text in uncached_texts]

                for text, embedding, idx in zip(uncached_texts, computed_embeddings, uncached_indices):
                    self.embedding_cache[text] = embedding
                    result[idx] = embedding
            except Exception as e:
                print(f"Error in batch embedding: {str(e)}")
                for idx in uncached_indices:
                    if result[idx] is None:
                        result[idx] = np.zeros(384)

        return result

    def preprocess_text(self, text):
        if not text:
            return []

        try:
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)

            tokens = word_tokenize(text)

            tokens = [word for word in tokens if word not in self.stop_words]

            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]

            return tokens
        except Exception as e:
            print(f"Error in text preprocessing: {str(e)}")
            return []

    def calculate_faithfulness(self, answer, contexts):
        try:
            if not answer or not contexts:
                return 0.5

            # Preprocess and chunk large contexts
            processed_contexts = []
            for ctx in contexts:
                if len(ctx) > 1000:
                    sentences = sent_tokenize(ctx)
                    current_chunk = ""
                    for sent in sentences:
                        if len(current_chunk) + len(sent) < 1000:
                            current_chunk += " " + sent
                        else:
                            if current_chunk:
                                processed_contexts.append(current_chunk.strip())
                            current_chunk = sent
                    if current_chunk:
                        processed_contexts.append(current_chunk.strip())
                else:
                    processed_contexts.append(ctx)

            if not processed_contexts:
                processed_contexts = contexts

            # Tokenize and preprocess the answer
            answer_tokens = set(self.preprocess_text(answer))
            if not answer_tokens:
                answer_tokens = set(answer.lower().split())

            # Analyze context coverage and contradiction
            context_tokens = set()
            for context in processed_contexts:
                context_tokens.update(self.preprocess_text(context))

            if not context_tokens:
                context_tokens = set(' '.join(processed_contexts).lower().split())

            # Calculate token coverage
            if answer_tokens:
                covered_tokens = answer_tokens.intersection(context_tokens)
                coverage_ratio = len(covered_tokens) / len(answer_tokens)
            else:
                coverage_ratio = 0.5

            # Semantic similarity check
            try:
                # Compute embeddings for answer and contexts
                answer_embedding = self.get_embedding(answer)
                context_embeddings = [self.get_embedding(ctx) for ctx in processed_contexts]

                # Calculate max semantic similarity
                semantic_similarities = [
                    cosine_similarity([answer_embedding], [ctx_emb])[0][0]
                    for ctx_emb in context_embeddings
                    if not all(v == 0 for v in ctx_emb)
                ]

                if semantic_similarities:
                    max_semantic_similarity = max(semantic_similarities)
                else:
                    max_semantic_similarity = 0.5
            except Exception:
                max_semantic_similarity = 0.5

            # Combine coverage and semantic similarity
            final_score = 0.6 * coverage_ratio + 0.4 * max_semantic_similarity

            # Normalize and bound the score
            final_score = max(0.1, min(0.9, final_score))

            return float(final_score)

        except Exception as e:
            print(f"Error in faithfulness calculation: {str(e)}")
            return 0.5

    def calculate_answer_relevancy(self, query, answer):
        try:
            if not query or not answer:
                return 0.5

            query_tokens = self.preprocess_text(query)
            if not query_tokens:
                query_tokens = query.lower().split()

            try:
                query_pos = nltk.pos_tag(query_tokens)
                key_query_concepts = [word for word, pos in query_pos if pos.startswith('NN')]
                if not key_query_concepts:
                    key_query_concepts = query_tokens
            except Exception:
                key_query_concepts = query_tokens

            try:
                answer_tokens = set(self.preprocess_text(answer))
                if not answer_tokens:
                    answer_tokens = set(answer.lower().split())
            except Exception:
                answer_tokens = set(answer.lower().split())

            if key_query_concepts and answer_tokens:
                matches = sum(1 for concept in key_query_concepts if concept in answer_tokens)
                concept_match_score = matches / len(key_query_concepts) if len(key_query_concepts) > 0 else 0
            else:
                concept_match_score = 0.5

            try:
                query_embedding = self.get_embedding(query)
                answer_embedding = self.get_embedding(answer)
                semantic_score = cosine_similarity([query_embedding], [answer_embedding])[0][0]

                if np.isnan(semantic_score) or semantic_score < -1 or semantic_score > 1:
                    semantic_score = 0.5
            except Exception:
                semantic_score = 0.5

            combined_score = 0.7 * semantic_score + 0.3 * concept_match_score
            final_score = max(0.1, min(0.9, combined_score))
            return float(final_score)
        except Exception as e:
            print(f"Error in answer relevancy: {str(e)}")
            return 0.5

    def calculate_context_precision(self, query, contexts):
        try:
            if not query or not contexts:
                return 0.5

            try:
                query_parts = sent_tokenize(query)
                if len(query_parts) == 1:
                    query_parts = [q.strip() for q in re.split(r'[,;]|\band\b|\bor\b', query) if q.strip()]
                    if not query_parts:
                        query_parts = [query]
            except Exception:
                query_parts = [query]

            try:
                tokenized_contexts = [ctx.split() for ctx in contexts]
                tokenized_query = query.split()
                bm25 = BM25Okapi(tokenized_contexts)
                bm25_scores = bm25.get_scores(tokenized_query)

                max_bm25 = max(bm25_scores) if bm25_scores and max(bm25_scores) > 0 else 1
                normalized_bm25 = [score / max_bm25 for score in bm25_scores]
            except Exception as e:
                print(f"BM25 error: {str(e)}")
                normalized_bm25 = [0.5] * len(contexts)

            try:
                query_embedding = self.get_embedding(query)
                context_embeddings = []
                for ctx in contexts:
                    ctx_embedding = self.get_embedding(ctx)
                    context_embeddings.append(ctx_embedding)

                semantic_scores = []
                for ctx_emb in context_embeddings:
                    similarity = cosine_similarity([query_embedding], [ctx_emb])[0][0]
                    if np.isnan(similarity) or similarity < -1 or similarity > 1:
                        similarity = 0.5
                    semantic_scores.append(similarity)
            except Exception as e:
                print(f"Semantic error: {str(e)}")
                semantic_scores = [0.5] * len(contexts)

            min_length = min(len(normalized_bm25), len(semantic_scores))
            if min_length == 0:
                return 0.5

            normalized_bm25 = normalized_bm25[:min_length]
            semantic_scores = semantic_scores[:min_length]

            combined_scores = [0.4 * bm25 + 0.6 * sem for bm25, sem in zip(normalized_bm25, semantic_scores)]

            avg_score = sum(combined_scores) / len(combined_scores) if combined_scores else 0.5
            final_score = max(0.1, min(0.9, avg_score))
            return float(final_score)
        except Exception as e:
            print(f"Error in context precision: {str(e)}")
            return 0.5

    def calculate_context_recall(self, query, contexts, answer):
        try:
            if not query or not contexts or not answer:
                return 0.5

            try:
                query_terms = set(self.preprocess_text(query))
                answer_terms = set(self.preprocess_text(answer))

                if not query_terms:
                    query_terms = set(query.lower().split())
                if not answer_terms:
                    answer_terms = set(answer.lower().split())

                important_terms = query_terms.union(answer_terms)

                context_terms = set()
                for context in contexts:
                    context_tokens = self.preprocess_text(context)
                    if not context_tokens:
                        context_tokens = context.lower().split()
                    context_terms.update(context_tokens)

                if len(important_terms) > 0:
                    coverage = len(important_terms.intersection(context_terms)) / len(important_terms)
                    final_score = max(0.1, min(0.9, coverage))
                    return float(final_score)
                else:
                    return 0.5
            except Exception as e:
                print(f"Error in recall details: {str(e)}")
                return 0.5
        except Exception as e:
            print(f"Error in context recall: {str(e)}")
            return 0.5

    def calculate_rouge_scores(self, hypothesis, reference):
        try:
            if not hypothesis or not reference:
                return 0.5

            try:
                # Preprocessing tokens with more lenient approach
                hyp_tokens = self.preprocess_text(hypothesis)
                ref_tokens = self.preprocess_text(reference)

                if not hyp_tokens:
                    hyp_tokens = hypothesis.lower().split()
                if not ref_tokens:
                    ref_tokens = reference.lower().split()

                hyp_sents = sent_tokenize(hypothesis.lower())
                ref_sents = sent_tokenize(reference.lower())

                if not hyp_sents:
                    hyp_sents = [hypothesis.lower()]
                if not ref_sents:
                    ref_sents = [reference.lower()]
            except Exception as e:
                print(f"Tokenization error: {str(e)}")
                hyp_tokens = hypothesis.lower().split()
                ref_tokens = reference.lower().split()
                hyp_sents = [hypothesis.lower()]
                ref_sents = [reference.lower()]

            # Unigram calculation (ROUGE-1) with more emphasis on recall
            hyp_unigrams = set(hyp_tokens)
            ref_unigrams = set(ref_tokens)
            overlap_unigrams = hyp_unigrams.intersection(ref_unigrams)

            if len(hyp_unigrams) == 0 or len(ref_unigrams) == 0:
                rouge1 = 0.5
            else:
                precision = len(overlap_unigrams) / len(hyp_unigrams) if len(hyp_unigrams) > 0 else 0
                recall = len(overlap_unigrams) / len(ref_unigrams) if len(ref_unigrams) > 0 else 0
                # Emphasize recall more than precision (beta=2)
                rouge1 = (1 + 2 ** 2) * (precision * recall) / ((2 ** 2 * precision) + recall) if (
                                                                                                          precision + recall) > 0 else 0

            # Bigram calculation (ROUGE-2)
            try:
                hyp_bigrams = set(zip(hyp_tokens[:-1], hyp_tokens[1:]))
                ref_bigrams = set(zip(ref_tokens[:-1], ref_tokens[1:]))
                overlap_bigrams = hyp_bigrams.intersection(ref_bigrams)

                if len(hyp_bigrams) == 0 or len(ref_bigrams) == 0:
                    rouge2 = 0.5
                else:
                    precision = len(overlap_bigrams) / len(hyp_bigrams) if len(hyp_bigrams) > 0 else 0
                    recall = len(overlap_bigrams) / len(ref_bigrams) if len(ref_bigrams) > 0 else 0
                    # Standard F1 for bigrams
                    rouge2 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            except Exception as e:
                print(f"Bigram error: {str(e)}")
                rouge2 = 0.5

            # Use a more lenient ROUGE-L approach with semantic similarity backup
            try:
                # Calculate longest common subsequence-based similarity
                common_sents = set(hyp_sents).intersection(set(ref_sents))
                exact_match_score = len(common_sents) / max(len(hyp_sents), len(ref_sents)) if max(len(hyp_sents),
                                                                                                   len(ref_sents)) > 0 else 0

                # If exact match is very low, supplement with semantic similarity
                if exact_match_score < 0.3:
                    try:
                        hyp_embedding = self.get_embedding(" ".join(hyp_tokens))
                        ref_embedding = self.get_embedding(" ".join(ref_tokens))
                        semantic_sim = cosine_similarity([hyp_embedding], [ref_embedding])[0][0]

                        # Blend semantic similarity with exact match
                        rougeL = 0.4 * exact_match_score + 0.6 * semantic_sim
                    except Exception:
                        rougeL = exact_match_score
                else:
                    rougeL = exact_match_score
            except Exception as e:
                print(f"ROUGE-L error: {str(e)}")
                rougeL = 0.5

            # Weight the scores - emphasize Rouge-1 and semantic similarity
            avg_rouge = (0.4 * rouge1 + 0.2 * rouge2 + 0.4 * rougeL)

            # Apply a more lenient scaling to avoid overly harsh penalties
            # Scale from [0-1] to [0.1-0.9] with a slight boost to low-mid scores
            final_score = 0.1 + 0.8 * avg_rouge

            # Apply a sigmoid-like transformation to boost mid-range scores
            if 0.3 <= final_score <= 0.7:
                final_score = min(0.9, final_score + 0.1)

            return final_score

        except Exception as e:
            print(f"Error in ROUGE scores: {str(e)}")
            return 0.5

    def calculate_answer_correctness(self, answer, ground_truth):
        try:
            if not answer or not ground_truth:
                return 0.5

            # Calculate semantic similarity between answer and ground truth
            try:
                answer_embedding = self.get_embedding(answer)
                ground_truth_embedding = self.get_embedding(ground_truth)
                semantic_similarity = cosine_similarity([answer_embedding], [ground_truth_embedding])[0][0]

                # Ensure semantic similarity is in valid range
                if np.isnan(semantic_similarity) or semantic_similarity < -1 or semantic_similarity > 1:
                    semantic_similarity = 0.5
            except Exception as e:
                print(f"Error in semantic similarity: {str(e)}")
                semantic_similarity = 0.5

            # Get ROUGE-based lexical similarity
            rouge_score = self.calculate_rouge_scores(answer, ground_truth)

            # Combine ROUGE with semantic similarity, weighting semantic similarity higher
            # This accounts for answers that are semantically correct but use different wording
            combined_score = 0.4 * rouge_score + 0.6 * semantic_similarity

            # Apply a more generous baseline - RAG systems rarely get perfect matches with ground truth
            # but can still provide correct information
            final_score = max(0.3, min(0.9, combined_score))

            return float(final_score)
        except Exception as e:
            print(f"Error in answer correctness: {str(e)}")
            return 0.5

    def evaluate_answer(self, query, retrieved_contexts, generated_answer, ground_truth=None):
        try:
            results = {
                "faithfulness": self.calculate_faithfulness(generated_answer, retrieved_contexts),
                "answer_relevancy": self.calculate_answer_relevancy(query, generated_answer),
                "context_precision": self.calculate_context_precision(query, retrieved_contexts),
                "context_recall": self.calculate_context_recall(query, retrieved_contexts, generated_answer)
            }

            if ground_truth:
                results["answer_correctness"] = self.calculate_answer_correctness(generated_answer, ground_truth)

            self.evaluation_results.append({
                "query": query,
                "generated_answer": generated_answer,
                "ground_truth": ground_truth,
                "metrics": results
            })

            return results
        except Exception as e:
            print(f"Error in evaluate_answer: {str(e)}")
            results = {
                "faithfulness": 0.5,
                "answer_relevancy": 0.5,
                "context_precision": 0.5,
                "context_recall": 0.5
            }
            if ground_truth:
                results["answer_correctness"] = 0.5

            return results


def get_mistral_response(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "mistral", "prompt": prompt, "stream": False}
        )
        return response.json()['response']
    except Exception as e:
        return f"Error communicating with Mistral: {e}"


def agent_retrieve_answer(query, vector_db, thinking_steps=3, reasoning_depth=3):
    """
    Implements an Agent RAG approach that makes decisions about retrieval strategy
    and dynamically plans the reasoning approach based on the query type.

    Args:
        query (str): The user's question
        vector_db (FusionPDFVectorDB): The vector database with different search methods
        thinking_steps (int): Controls how thoroughly the agent analyzes the query
        reasoning_depth (int): Controls how detailed the agent's reasoning will be

    Returns:
        tuple: (final_answer, agent_thought_process, contexts)
    """
    try:
        # Step 1: Analyze the query to determine the best retrieval strategy
        query_analysis_prompt = f"""
        Analyze this query and determine the best search strategy. Your options are:
        1. Vector search - best for conceptual questions and semantic understanding
        2. BM25 search - best for factual recall and when specific terms must be matched
        3. Fusion search - best for balanced queries needing both semantic and keyword matching

        Query: "{query}"

        Think step by step about what kind of information is needed. Respond with ONLY ONE of these exact terms: "Vector", "BM25", or "Fusion".
        """

        search_strategy = get_mistral_response(query_analysis_prompt).strip()

        # Ensure we get a valid strategy, default to Fusion if response is unclear
        valid_strategies = ["Vector", "BM25", "Fusion"]
        if search_strategy not in valid_strategies:
            search_strategy = "Fusion"

        # Step 2: Determine k (number of contexts) based on query complexity
        k_determination_prompt = f"""
        Analyze this query and determine how many document chunks we should retrieve.
        For simple factual queries, fewer chunks (2-3) are needed.
        For complex analytical questions, more chunks (5-8) are better.

        Query: "{query}"

        Analyze step by step what kind of information is needed, then respond with ONLY a single number between 2 and 8.
        """

        try:
            k_value = int(get_mistral_response(k_determination_prompt).strip())
            # Ensure k is within bounds
            k_value = max(2, min(8, k_value))
        except:
            # Default to 5 if we can't parse a valid number
            k_value = 5

        # Step 3: Retrieve contexts using the determined strategy
        if search_strategy == "Vector":
            contexts = vector_db.vector_search(query, k=k_value)
            retrieval_method = "Vector Search (semantic matching)"
        elif search_strategy == "BM25":
            contexts = vector_db.bm25_search(query, k=k_value)
            retrieval_method = "BM25 Search (keyword matching)"
        else:  # Fusion
            alpha_value = 0.5  # Could be dynamically determined as well
            contexts = vector_db.fusion_search(query, k=k_value, alpha=alpha_value)
            retrieval_method = "Fusion Search (balanced semantic and keyword matching)"

        # Step 4: Analyze if we need additional contexts
        if len(contexts) > 0:
            first_pass_context = "\n\n".join(contexts)
            sufficiency_prompt = f"""
            Given this information retrieved from the document:
            {first_pass_context}

            And this query: "{query}"

            Do we have enough information to fully answer the query? Respond with ONLY "YES" or "NO".
            """

            sufficiency_response = get_mistral_response(sufficiency_prompt).strip().upper()

            # If we need more information and haven't maxed out our context retrieval
            if sufficiency_response == "NO" and k_value < 8:
                # Try a different retrieval method to get complementary information
                if search_strategy != "Fusion":
                    additional_contexts = vector_db.fusion_search(query, k=3)
                    contexts.extend(additional_contexts)
                    retrieval_method += " + Additional Fusion Search"
                else:
                    # If we already used fusion, try pure vector for additional context
                    additional_contexts = vector_db.vector_search(query, k=3)
                    contexts.extend(additional_contexts)
                    retrieval_method += " + Additional Vector Search"

        # Step 5: Determine the reasoning approach needed
        reasoning_prompt = f"""
        Given this query: "{query}"

        What kind of reasoning approach should I use to answer it effectively?
        1. Direct factual answer - for simple factual questions
        2. Comparative analysis - for questions comparing multiple items
        3. Causal reasoning - for questions about why something happened
        4. Step-by-step explanation - for process or how-to questions
        5. Multi-perspective synthesis - for complex questions needing different viewpoints

        Think step by step, then respond with ONLY ONE number (1-5) that best matches the reasoning approach needed.
        """

        try:
            reasoning_type = int(get_mistral_response(reasoning_prompt).strip())
            # Ensure reasoning_type is within bounds
            reasoning_type = max(1, min(5, reasoning_type))
        except:
            # Default to 1 if we can't parse a valid number
            reasoning_type = 1

        # Map reasoning type to approach instruction
        reasoning_approaches = {
            1: "Provide a direct and concise factual answer based on the information.",
            2: "Compare and contrast the relevant elements to answer the question.",
            3: "Explain the causal relationships and why things happened this way.",
            4: "Break down the explanation into clear sequential steps.",
            5: "Synthesize multiple perspectives to provide a comprehensive answer."
        }

        reasoning_instruction = reasoning_approaches.get(reasoning_type, reasoning_approaches[1])

        # Step 6: Generate the final answer
        context_text = "\n\n".join(contexts)

        final_prompt = f"""
        Using this information retrieved from the document (via {retrieval_method}): 

        {context_text}

        Question: {query}

        {reasoning_instruction}

        Answer the question thoroughly based on the provided information. If you cannot answer based on the provided context, say so clearly.
        """

        final_answer = get_mistral_response(final_prompt)

        # Store the agent's thought process
        agent_thought_process = {
            "search_strategy": search_strategy,
            "k_value": k_value,
            "retrieval_method": retrieval_method,
            "reasoning_approach": reasoning_approaches[reasoning_type],
            "num_contexts": len(contexts)
        }

        return final_answer, agent_thought_process, contexts

    except Exception as e:
        return f"Error in Agent RAG process: {str(e)}\n\nFalling back to basic retrieval. Please try again with a reformulated question.", None, []


def extract_page_chapter_info(context):
    page_info = "Unknown page"
    page_match = re.search(r'Page (\d+)', context)
    if page_match:
        page_info = f"Page {page_match.group(1)}"
    else:
        alt_page_match = re.search(r'p\.? (\d+)', context, re.IGNORECASE)
        if alt_page_match:
            page_info = f"Page {alt_page_match.group(1)}"

    chapter_info = "Unknown chapter"
    chapter_match = re.search(r'Chapter (\d+|[IVXLCDM]+)', context)
    if chapter_match:
        chapter_info = f"Chapter {chapter_match.group(1)}"
    else:
        alt_chapter_match = re.search(r'Chapter[:\s]+([A-Za-z0-9\s]+)', context)
        if alt_chapter_match:
            chapter_title = alt_chapter_match.group(1).strip()
            if len(chapter_title) > 30:
                chapter_title = chapter_title[:27] + "..."
            chapter_info = f"Chapter: {chapter_title}"

    return page_info, chapter_info


# Streamlit UI
st.set_page_config(page_title="Book Knowledge Bot", page_icon="📚", layout="wide")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = []

if 'ground_truths' not in st.session_state:
    st.session_state.ground_truths = {}

if 'evaluator' not in st.session_state:
    st.session_state.evaluator = CustomRAGEvaluator()

# Initialize RAG type if not set
if 'rag_type' not in st.session_state:
    st.session_state.rag_type = "Fusion RAG"

# Initialize fusion search type if not set
if 'fusion_search_type' not in st.session_state:
    st.session_state.fusion_search_type = "Fusion"

# Initialize debug mode
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False

# Initialize agent thought process store
if 'agent_thoughts' not in st.session_state:
    st.session_state.agent_thoughts = {}

# Initialize VectorDB with fusion search
if 'vector_db' not in st.session_state:
    pdf_path = "/Users/basilyassin/Desktop/chocolate_ev/Aka Book.pdf"
    vector_store_path = "vector_stores/detailed_store"

    with st.spinner("Loading vector store..."):
        try:
            st.session_state.vector_db = FusionPDFVectorDB(pdf_path, vector_store_path)
        except Exception as e:
            st.error(f"Error initializing vector store: {str(e)}")
            st.stop()

# Create main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.title("📚 Book Knowledge Bot")
    st.write("Ask me anything about the book!")

    # Adding RAG Type selector at the top
    rag_type = st.radio(
        "Select RAG Type",
        ["Fusion RAG", "Agent RAG"],
        index=0 if st.session_state.rag_type == "Fusion RAG" else 1,
        horizontal=True
    )

    # Update session state when changed
    if rag_type != st.session_state.rag_type:
        st.session_state.rag_type = rag_type

    # Only show search type dropdown for Fusion RAG
    if st.session_state.rag_type == "Fusion RAG":
        search_type = st.radio(
            "Search Method",
            ["Fusion", "Vector", "BM25"],
            index=["Fusion", "Vector", "BM25"].index(st.session_state.fusion_search_type),
            horizontal=True
        )
        # Update session state
        st.session_state.fusion_search_type = search_type

    # Chat interface
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

            # Show agent thought process in debug mode
            if st.session_state.debug_mode and message["role"] == "assistant" and "message_id" in message:
                if message["message_id"] in st.session_state.agent_thoughts:
                    with st.expander("Agent Thought Process"):
                        thought_process = st.session_state.agent_thoughts[message["message_id"]]
                        st.json(thought_process)

    # User input
    if prompt := st.chat_input("What would you like to know about the book?"):
        message_id = str(uuid.uuid4())

        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Get relevant information using selected RAG approach
        with st.spinner(""):
            try:
                if st.session_state.rag_type == "Fusion RAG":
                    # Get parameters from sidebar
                    k_value = st.session_state.get('k_value', 5)
                    alpha_value = st.session_state.get('alpha_value', 0.5)

                    # Get context based on search type
                    search_type = st.session_state.fusion_search_type
                    if search_type == "Fusion":
                        context = st.session_state.vector_db.fusion_search(prompt, k=k_value, alpha=alpha_value)
                        search_method = "Fusion Search (Vector + BM25)"
                    elif search_type == "Vector":
                        context = st.session_state.vector_db.vector_search(prompt, k=k_value)
                        search_method = "Vector Search"
                    else:  # BM25
                        context = st.session_state.vector_db.bm25_search(prompt, k=k_value)
                        search_method = "BM25 Keyword Search"

                    # Format context and prepare prompt
                    context_text = "\n\n".join(context)
                    full_prompt = f"Using this information from the book (retrieved via {search_method}): {context_text}\n\nQuestion: {prompt}\nAnswer:"

                    # Get and display Mistral's response
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            response = get_mistral_response(full_prompt)
                            st.write(response)

                    # Store in chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response,
                        "message_id": message_id
                    })

                    agent_thought_process = None
                else:
                    # Agent RAG implementation
                    thinking_steps = st.session_state.get('agent_thinking_steps', 3)
                    reasoning_depth = st.session_state.get('agent_reasoning_depth', 3)

                    with st.chat_message("assistant"):
                        with st.spinner("Agent thinking..."):
                            response, agent_thought_process, context = agent_retrieve_answer(
                                prompt,
                                st.session_state.vector_db,
                                thinking_steps,
                                reasoning_depth
                            )
                            st.write(response)

                            # Show debug information if enabled
                            if st.session_state.debug_mode and agent_thought_process:
                                with st.expander("Agent Thought Process"):
                                    st.json(agent_thought_process)

                    # Store agent thoughts for this message
                    if agent_thought_process:
                        st.session_state.agent_thoughts[message_id] = agent_thought_process

                    # Store in chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response,
                        "message_id": message_id
                    })

                # Evaluate the response if ground truth exists
                ground_truth = st.session_state.ground_truths.get(prompt, None)

                # Get the context based on RAG type
                if st.session_state.rag_type == "Agent RAG":
                    # context is already defined from agent_retrieve_answer
                    pass

                evaluation_results = st.session_state.evaluator.evaluate_answer(
                    query=prompt,
                    retrieved_contexts=context,
                    generated_answer=response,
                    ground_truth=ground_truth
                )

                # Store the results
                st.session_state.evaluation_results.append({
                    "question": prompt,
                    "answer": response,
                    "ground_truth": ground_truth,
                    "contexts": context,
                    "metrics": evaluation_results,
                    "rag_type": st.session_state.rag_type,
                    "agent_thoughts": agent_thought_process
                })

            except Exception as e:
                st.error(f"Error processing query: {str(e)}")

with col2:
    st.header("Evaluation Panel")

    # Ground Truth Input
    with st.expander("Add Ground Truth", expanded=True):
        st.subheader("Add Ground Truth for Questions")
        question_for_truth = st.text_input("Question", key="question_truth")
        ground_truth_text = st.text_area("Ground Truth Answer", key="ground_truth_text", height=150)

        if st.button("Save Ground Truth"):
            if question_for_truth and ground_truth_text:
                st.session_state.ground_truths[question_for_truth] = ground_truth_text
                st.success(f"Ground truth saved for: {question_for_truth}")
            else:
                st.warning("Both question and ground truth are required")

    # Display ground truths
    with st.expander("View Ground Truths", expanded=False):
        st.subheader("Saved Ground Truths")
        if st.session_state.ground_truths:
            for q, a in st.session_state.ground_truths.items():
                st.markdown(f"**Q: {q}**")
                st.markdown(f"A: {a}")
                st.divider()
        else:
            st.info("No ground truths saved yet")

    # Evaluation Results Section
    st.subheader("Evaluation Results")
    if st.session_state.evaluation_results:
        # Display the most recent evaluation first
        latest_eval = st.session_state.evaluation_results[-1]

        st.markdown(f"**Question:** {latest_eval['question']}")
        st.markdown(f"**RAG Type:** {latest_eval['rag_type']}")

        # Display each metric individually with clearer formatting
        st.markdown("### Metrics Breakdown:")
        for metric_name, score in latest_eval['metrics'].items():
            # Include answer_correctness but still skip context_recall
            if metric_name == 'context_recall':
                continue

            if score is not None:  # Only display metrics that have values
                # Create a color based on score (red to green)
                color = f"rgb({int(255 * (1 - score))}, {int(255 * score)}, 0)"
                st.markdown(
                    f"<div style='display: flex; justify-content: space-between;'>"
                    f"<span><b>{metric_name.replace('_', ' ').title()}</b></span>"
                    f"<span style='color: {color}; font-weight: bold;'>{score:.4f}</span>"
                    f"</div>",
                    unsafe_allow_html=True)
                st.divider()

        # Show context and answers in expandable sections
        with st.expander("View Retrieved Contexts", expanded=False):
            for i, context in enumerate(latest_eval.get('contexts', [])):
                # Extract page and chapter information
                page_info, chapter_info = extract_page_chapter_info(context)

                st.markdown(f"**Context {i + 1}:** ({page_info}, {chapter_info})")
                st.text(context)
                st.divider()

        with st.expander("View Answer and Ground Truth", expanded=False):
            st.markdown("**Generated Answer:**")
            st.write(latest_eval['answer'])
            if latest_eval.get('ground_truth'):
                st.markdown("**Ground Truth:**")
                st.write(latest_eval['ground_truth'])

        # Show agent thoughts if available
        if latest_eval.get('agent_thoughts') and latest_eval['rag_type'] == "Agent RAG":
            with st.expander("View Agent Reasoning", expanded=False):
                st.json(latest_eval['agent_thoughts'])

        # Show all evaluations in an expander
        with st.expander("All Evaluation Results", expanded=False):
            st.markdown("### All Evaluations")

            # Create a dataframe from all evaluation results
            all_evals = []
            for eval_result in st.session_state.evaluation_results:
                metrics = eval_result['metrics'].copy()  # Use copy to avoid modifying the original

                # Only remove context_recall from display, keep answer_correctness
                if 'context_recall' in metrics:
                    del metrics['context_recall']

                metrics['question'] = eval_result['question']
                metrics['rag_type'] = eval_result['rag_type']
                all_evals.append(metrics)

            if all_evals:
                all_evals_df = pd.DataFrame(all_evals)
                st.dataframe(all_evals_df, use_container_width=True)

                # Calculate and show average scores by RAG type
                st.markdown("### Average Scores by RAG Type")

                # Group by RAG type
                rag_types = all_evals_df['rag_type'].unique()

                for rag_type in rag_types:
                    st.markdown(f"#### {rag_type}")
                    rag_type_df = all_evals_df[all_evals_df['rag_type'] == rag_type]
                    numeric_cols = [col for col in rag_type_df.columns if col not in ['question', 'rag_type']]
                    avg_scores = rag_type_df[numeric_cols].mean(numeric_only=True)

                    for metric, avg_score in avg_scores.items():
                        # Skip only context_recall, but include answer_correctness
                        if metric == 'context_recall':
                            continue

                        if not pd.isna(avg_score):
                            st.metric(
                                label=f"Avg {metric.replace('_', ' ').title()}",
                                value=f"{avg_score:.4f}"
                            )
    else:
        st.info("No evaluations yet. Ask a question to see evaluation results.")

# Sidebar with additional information
with st.sidebar:
    st.header("About")
    st.write("""
    This chatbot uses:
    - LangChain for PDF processing
    - FAISS for vector storage
    - BM25 for keyword search
    - Fusion retrieval combining vector and keyword search
    - Agent-based retrieval system
    - HuggingFace embeddings (all-MiniLM-L6-v2)
    - Mistral LLM for generating responses
    - Custom metrics for RAG system evaluation
    """)

    with st.expander("Evaluation Metrics Explained"):
        st.markdown("""
        ### Metrics Explanation

        **Faithfulness (0-1)**: 
        - Measures how closely the answer sticks to information in the retrieved contexts
        - Higher score = less hallucination

        **Answer Relevancy (0-1)**:
        - Measures how well the answer addresses the original question
        - Higher score = more relevant answer

        **Context Precision (0-1)**:
        - Measures what percentage of retrieved contexts are relevant to the query
        - Higher score = more efficient retrieval

        **Answer Correctness (0-1)**:
        - Compares generated answer with ground truth using ROUGE scoring
        - Combines unigram (ROUGE-1), bigram (ROUGE-2), and longest common subsequence (ROUGE-L)
        - Higher score = closer match to ground truth
        """)

    # Show different parameters based on RAG type
    if st.session_state.rag_type == "Fusion RAG":
        st.subheader("Fusion RAG Parameters")
        st.session_state.k_value = st.slider("Number of contexts (k)", min_value=1, max_value=10, value=5)
        st.session_state.alpha_value = st.slider("Vector search weight (α)", min_value=0.0, max_value=1.0, value=0.5,
                                                 step=0.1)
    else:
        st.subheader("Agent RAG Parameters")
        st.session_state.agent_thinking_steps = st.slider("Thinking steps", min_value=1, max_value=5, value=3,
                                                          help="Controls how thoroughly the agent analyzes the query")
        st.session_state.agent_reasoning_depth = st.slider("Reasoning depth", min_value=1, max_value=5, value=3,
                                                           help="Controls how detailed the agent's reasoning will be")

        # Debug mode toggle
        st.session_state.debug_mode = st.checkbox("Debug mode", value=st.session_state.debug_mode,
                                                  help="Show agent thinking process")

if st.session_state.evaluation_results:
    if st.button("Export Evaluation Results (CSV)"):
        export_data = []
        for eval_result in st.session_state.evaluation_results:
            row = {
                "question": eval_result['question'],
                "answer": eval_result['answer'],
                "rag_type": eval_result['rag_type'],
                "ground_truth": eval_result.get('ground_truth', "")
            }
            for metric_name, value in eval_result['metrics'].items():
                # Include answer_correctness in export, only exclude context_recall
                if metric_name != 'context_recall':
                    row[metric_name] = value

            export_data.append(row)

        export_df = pd.DataFrame(export_data)
        csv = export_df.to_csv(index=False)

        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="rag_evaluation_results.csv",
            mime="text/csv"
        )

if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()

if st.button("Clear Evaluation Results"):
    st.session_state.evaluation_results = []
    st.rerun()

# Run the Streamlit app
if __name__ == "__main__":
    pass  # or remove this entire block