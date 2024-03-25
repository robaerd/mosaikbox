from sentence_transformers import SentenceTransformer, util


class TextSimilarity:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def calculate_embeddings(self, text):
        return self.model.encode(text, convert_to_tensor=True)

    @staticmethod
    def embedding_similarity(embeddings1, embeddings2):
        return util.cos_sim(embeddings1, embeddings2)
