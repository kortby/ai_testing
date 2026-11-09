import fais 
import numpy as np

from transformers import TFAutoModelForCausalLM

# Create a Faiss index for storing embeddings
dimension = 384 # embedding size from MiniLM
index = faiss.IndexFlatL2(dimension)

# embedding to vectors
# Add document embeddings to the index
embeddings = np.array([get_embedding(doc).numpy() for doc in ["Doc 1 text", "Doc 2 text"]])
index.add(embeddings)


# Query the index
query_embedding = get_embedding("Query text").numpy().distances, indices = index.search(query_embedding, k=3) # Retrieve top 3 matches
print("Closest documents:", indices)

# Load Phi-3.5-mini-instruct model
generator_model = TFAutoModelForCausalLM.from_pretrained("microsoft/Phi-3.5-mini-instruct")


def generate_response(query, context):
    input_text = f"User query: {query}\n\nContent\n{context}\n\nAnswer:"
    inputs = tokenizer_phi3(input_text, return_tensors="tf")
    response = generator_model.generate(**inputs, max_length=100)
    
    return tokenizer_phi3.decode(response[0], skip_special_tokens=True)

# Using retrieved documents as context
context = "\n\n".join(["Document 1 text", "Document 2 text"]) # replace with actual retrieved docs
generated_response = generate_response("How does RAG work?", context)
print(generated_response)