import time
import numpy as np
import pandas as pd
import torch
import faiss
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer, util
from pythainlp import Tokenizer
import pickle
import evaluate
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import regex as re
from pythainlp.tokenize import sent_tokenize
import os

# Constants
DEFAULT_MODEL = 'wangchanberta'
DEFAULT_SENTENCE_EMBEDDING_MODEL = 'intfloat/multilingual-e5-base'
DATA_PATH = '../data/dataset.xlsx'
EMBEDDINGS_PATH = '../data/embeddings.pkl'
MODEL_DICT = {
    'wangchanberta': 'powerpuf-bot/wangchanberta-th-wiki-qa_hyp-params',
    'mdeberta': 'powerpuf-bot/mdeberta-v3-th-wiki-qa_hyp-params',
}


class Chatbot:
    def __init__(self, model=DEFAULT_MODEL):
        """
        Initialize the Chatbot object.

        Parameters:
        - model (str): The name of the model to be used. Defaults to DEFAULT_MODEL.
        """
        self.df = None
        self.model = None
        self.model_name = None
        self.tokenizer = None
        self.embedding_model = None
        self.vectors = None
        self.index = None
        self.k = 1

        self.load_data()
        self.load_model(model)
        self.load_embedding_model(DEFAULT_SENTENCE_EMBEDDING_MODEL)
        self.set_vectors()
        self.set_index()

    def load_data(self, path=DATA_PATH):
        self.df = pd.read_excel(path, sheet_name='Default')
        self.df['Context'] = pd.read_excel(
            path, sheet_name='mdeberta')['Context']
        print('Load data done')

    def load_model(self, model_name=DEFAULT_MODEL):
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            MODEL_DICT[model_name])
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DICT[model_name])
        self.model_name = model_name
        print('Load model done')

    def load_embedding_model(self, model_name=DEFAULT_SENTENCE_EMBEDDING_MODEL):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedding_model = SentenceTransformer(model_name, device=device)
        print('Load sentence embedding model done')

    def set_vectors(self):
        self.vectors = self.prepare_sentences_vector(
            self.load_embeddings(EMBEDDINGS_PATH))

    def set_index(self):
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            self.index = faiss.IndexFlatL2(self.vectors.shape[1])
            gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, self.index)
            gpu_index_flat.add(self.vectors)
            self.index = gpu_index_flat
        else:
            self.index = faiss.IndexFlatL2(self.vectors.shape[1])
            self.index.add(self.vectors)

    def get_embeddings(self, text_list):
        return self.embedding_model.encode(text_list)

    def prepare_sentences_vector(self, encoded_list):
        encoded_list = np.vstack([i.reshape(1, -1)
                                    for i in encoded_list]).astype('float32')
        encoded_list = normalize(encoded_list)
        return encoded_list

    def load_embeddings(self, file_path):
        with open(file_path, "rb") as fIn:
            stored_data = pickle.load(fIn)
            stored_embeddings = stored_data['embeddings']
        print('Load (questions) embeddings done')
        return stored_embeddings
    
    def store_embeddings(self,df, embeddings):
        with open('data/embeddings.pkl', "wb") as fOut:
            pickle.dump({'sentences': df['Question'], 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)
        print('Store embeddings done')

    # Generate Output by LLM
    def model_pipeline(self, question, similar_context):
        inputs = self.tokenizer(question, similar_context, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()
        predict_answer_tokens = inputs.input_ids[0,
                                                answer_start_index: answer_end_index + 1]
        return self.tokenizer.decode(predict_answer_tokens)

    # search for similar info, using the index of embedding vectors
    def faiss_search( self,index,question_vector, k=1):
        distances, indices = index.search(question_vector, k)
        return distances,indices
    
    def predict(self, message, answer_with_model=True):
        t = time.time()
        message = message.strip()
        question_vector = self.get_embeddings(message)
        question_vector = self.prepare_sentences_vector([question_vector])
        distances, indices = self.faiss_search(self.index,question_vector)
        if not answer_with_model:
            Answers = [self.df['Answer'][i] for i in indices[0]]
            return {
                "user_question": message,
                "distances": [distances[0][i] for i in range(self.k)],
                "answer": Answers[0],
                "totaltime": round(time.time() - t, 3)
            }
            
        most_sim_context = self.df['Context'][indices[0][0]]
        most_sim_question = self.df['Question'][indices[0][0]]
        Answer = self.model_pipeline(most_sim_question, most_sim_context)
        elapsed_time = round(time.time() - t, 3)
        output = {
            "user_question": message,
            "answer": Answer,
            "totaltime": elapsed_time,
            "distance": round(distances[0][0], 4),
        }
        return output
    
    def create_segment_index(self,vector):
        segment_index = faiss.IndexFlatL2(vector.shape[1])
        segment_index.add(vector)
        return segment_index

    def eval(
        self, 
        model_name: str ='wangchanberta', 
        answer_with_model: bool = True
        ) -> dict:
        """
        Evaluation Method.
        
        :param str model_name: the model to be used for prediction
        :param bool answer_with_model: choose if you want to use the model or not \
        **Options for model_name**
            * *wangchanberta* - (default) Thai based LLM
            * *mdeberta* - multilingual LLM 
        """
        
        # load model 
        if answer_with_model & (model_name != self.model_name):
            self.load_model(model_name)

        # load metric for evaluation
        exact_match_metric = evaluate.load("exact_match")
        bert_metric = evaluate.load("bertscore")
        rouge_metric = evaluate.load("rouge")
        
        # load Test data
        ref = pd.read_excel(DATA_PATH, sheet_name='Test')
        output = [self.predict(ref['Aug'][i],answer_with_model)
                        for i in range(len(ref))]
        
        # store the prediction log for all questions in test set    
        output = pd.DataFrame(output)
        output.to_excel(os.path.join(os.path.dirname('results'), 'predict_log.xlsx'),index=False)
        
        # Compute score
        result = {"model": model_name, "answer_with_model": answer_with_model}
        exact_match = exact_match_metric.compute(
            references=ref['Answer'], predictions=output["answer"])
        result['exact_match'] = round(exact_match['exact_match'], 4)
        bert_score = bert_metric.compute(
            predictions=output["answer"], references=ref['Answer'], lang="th", use_fast_tokenizer=True)
        df_bert_prec = pd.DataFrame(bert_score['precision'])
        df_bert_prec.ffill(inplace=True)
        bert_score['precision'] = df_bert_prec.values.tolist()
        for i in bert_score:
            if i == 'hashcode':
                break
            result[i] = round(np.mean(bert_score[i]), 4)
        tk = Tokenizer()
        rouge_score = rouge_metric.compute(
            predictions=output["answer"], references=ref['Answer'], tokenizer=tk.word_tokenize)
        for i in rouge_score:
            result[i] = round(rouge_score[i], 4)
        mean_time = round(np.mean(output['totaltime']), 3)
        result['mean_time'] = f'{mean_time * 1000} ms.'
        return result
