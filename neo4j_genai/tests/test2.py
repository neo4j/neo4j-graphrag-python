from neo4j_genai import GenAIClient
from neo4j import GraphDatabase

URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "password")

driver = GraphDatabase.driver(URI, auth=AUTH)

client = GenAIClient(driver)


client.create_vector_index("indexMovies", "Movie", "embedding", dimensions=666, similarity_function="cosine")

node = driver.execute_query("MATCH (m:Movie {movieId: row.movieId}) RETURN m")

######### EXPLICIT ENCODING ####################
try:
    from langchain_community.embeddings import OllamaEmbeddings
    embeddings = OllamaEmbeddings()

    embedded_vectors = embeddings.embed_query("This is the query")
except ImportError:
    embedded_vectors = requests.post("ollama.com/api/embeddings", query="This is the query")

######### EXPLICIT ENDODING ####################


client.setNodeVectorProperty(node, "embedding", embedded_vectors)

client.similarity_search("indexMovies", vectors=embedded_vectors)

#############################

from langchain_community.embeddings import OllamaEmbeddings
from neo4j_genai import GenAIClient
from neo4j import GraphDatabase

URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "password")

driver = GraphDatabase.driver(URI, auth=AUTH)

embedding_model = "ollama7b"
ollama7b_embedding_size = 666
embeddings = OllamaEmbeddings(embedding_model)
client = GenAIClient(driver, embeddings=embeddings)



client.create_vector_index("indexMovies", "Movie", "embedding", dimensions=ollama7b_embedding_size, similarity_function="cosine")


node = driver.execute_query("MATCH (m:Movie {movieId: row.movieId}) RETURN m")



# client.setNodeVectorProperty(node, "embedding", embedded_vectors)
client.similarity_search("indexMovies", text="Landing on the moon")
##

# def similarity_search(self, index_name, query_text):
#     embedded_vectors = self.embedddings.embed_query(query_text)

#     # similary()


# FUTURE
# client.generateEmbeddingsAndSetVectorProperty(node, text_property="plot", vector_property="embedding")