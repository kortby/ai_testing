from pathlib import Path
import re
from genesis.utils import simple_preprocess
from textBlob import TextBlob

from transformers import TFAutoModel, AutoTokenizer
import transformers as tf

import fais
import numpy as np

# Directory containing the text file
DATA_DIR = Path("./mimic_textbook/textbook/en")


def load_text_file(directory):
    texts = []
    for file_path in directory.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            texts.append(file.read())
        return texts
    

documents = load_text_file(DATA_DIR)
print(f"Load {len(documents)} documents.")

def clean_and_tokenize(text):
    # Basic regx cleaning
    text = re.sub(r'\s+', ' ', text) # Replace multiple spaces with a single space
    text = text.lower() # convert to lower case
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) # remove special characters
    
    # Tokenize with gensim
    tokens = simple_preprocess(text)
    return ' '.join(tokens)

# Apply cleaning and tokenization to each document
cleaned_documents = [clean_and_tokenize(doc) in documents for doc in documents]

print("Text cleaning and tokenization completed.")


# /**
# Now we're going to apply spell correction, and we'll use TextBlob. using TextBlob to support multiple languages, including English, and support multiple features, including spell correction, which is the one that we're going to use, but you can use all the other ones to tokenize as well, if you're handling another language, like, I dunno, German or French. This step ensures that misspelled words are corrected before embedding, improving the quality of our dataset and also improving later the quality of the query. Because if we get a misspelled user query, we will correct it and understand it regardless. This code is very simple to do. You just use TextBlob on that piece of text and use the correct method.
# */

def correct_spelling(text):
    # Use TextBlob for spelling correction
    correct_text = str(TextBlob(text).correct())
    return correct_text

# Apply spell correction to each clean document
corrected_documents = [correct_spelling(doc) for doc in cleaned_documents]
print("Spell correction completed")

# Prepare for the embedding
# datasets readability and semantic accuracy. To prepare for the embedding, now, what we need to do is we need to split the text into manageable chunks, for performance and also memory management. So we will divide each corrected document into smaller segments of 200 tokens. This function that we have will split each clean and corrected document into smaller chunks, making it ready for the embedding. Once we do that, we just apply it on every document, so now we'll have a list of chunked documents. Each chunked document will be basically 200 tokens, give or take. Depends on what we use on the chunk_size variable, and those documents will be already cleaned

def chunk_text(text, chunk_size = 200):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# check all correct documents
chunked_documents = []
for doc in corrected_documents:
    chunked_documents.extend(chunk_text(doc))

print(f"total of chunk created {len(chunked_documents)}")

# cleaned, processed, and corrected. Now our dataset has gone through several pre-processing stages: loading, cleaning, tokenizing, spell correction, and chunking. Each chunk is now ready for embedding and indexing in FAISS, ensuring clean, structured data for the next steps.


# ===================

# - A vector database is a specialized data structure that stores data in the form of embeddings or high dimensional vectors, which represent the semantic meaning of text, images, or other complex data types. Unlike traditional databases, a vector database excels at similarity based retrieval, identifying items that are close to each other in the embedding space. This makes vector databases particularly useful for tasks like information retrieval, recommendation systems, and, in our case, answering questions by retrieving the relevant chunks. The vector databases work by, first, we embed the data such as the text documents into vectors. Each vector will encode the meaning of that text as a point in a multidimensional space. Then we store those embeddings in the database, ready for the retrieval. And then when a query is embedded, it is matched with the most similar vectors in the database using similarity measure. Typically, cosine similarity or the clicking distance. This retrieves the most semantically relevant items to our query. In our example, 

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence_transformers/all-MiniLM-L6-v2")
model = TFAutoModel.from_pretrained("sentence_transformers/all-MiniLM-L6-v2")

def get_embedding(text):
    inputs = tokenizer(text, return_tensor="tf", truncation=True, padding=True)
    outputs = model(inputs).last_hidden_state
    return tf.reduce_mean(outputs, axis=1).numpy() # arrange pooling for sentence embedding

# Create embedding for each document chunk
embeddings = [get_embedding(chunk) for chunk in chunked_documents]
print(f"Generated embedding for {len(embeddings)} documents chunks.")


# define the dimensions for embeddings
dimension = 384 # dimension size for MiniLM
index = fais.IndexFlatL2(dimension)

# Convert embeddings to a Numpy array for FAISS compatibility
embedding_matrix = np.array(embeddings).astype('float32')
print(f"embedding matrix shape: {embedding_matrix.shape}")

# add all embeddings to index
index.add(embedding_matrix)
print(f"Total of embeddings indexed: {index.ntotal}")
# Query -> Embedding -> FAIS similarity search -> Relevant document Retrieval
