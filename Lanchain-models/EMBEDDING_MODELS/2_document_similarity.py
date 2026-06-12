import os
os.environ['HF_HOME']='D:/huggingface_cache'
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
document=['This is a bat',
          'Virat Kohli is a great cricketer',
          'Nabey-chabey']
doc_embedding=embedding.embed_documents(document)
query='Who is Virat Kohli'
vector=embedding.embed_query(query)

scores=cosine_similarity([vector],doc_embedding)[0]
score = max(scores)
index = scores.argmax()

print(query)
print(document[index])
print("similarity score is:", score)

