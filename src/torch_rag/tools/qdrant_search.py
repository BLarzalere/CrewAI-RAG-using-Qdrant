# tools/qdrant_search.py
from crewai_tools.tools.couchbase_tool import couchbase_tool
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import openai
import os
from dotenv import load_dotenv

load_dotenv()


class QdrantSearchInput(BaseModel):
    query: str = Field(description="The search query to find relevant information")

class QdrantSearchTool(BaseTool):
    name: str = "qdrant_search"
    description: str = "Search the knowledge base for information relevant to a query"
    args_schema: type[BaseModel] = QdrantSearchInput

    # initialize variables
    qdrant_url: str = os.getenv("QDRANT_URL")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY")
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    
    # Pydantic fields — not __init__ args
    collection: str = "MIT"
    top_k: int = 5

    def vectorize_query(self, query:str) -> list[float]:
        openai_client = openai.Client(api_key=self.openai_api_key)
        embedding = (openai_client.embeddings.create(input=[query], model="text-embedding-3-small").data[0].embedding)
        return embedding    
    
    def _run(self, query: str) -> str:
        client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key, cloud_inference=True)
        #model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

        #vector = model.encode([query])[0].tolist()
        query_vector = self.vectorize_query(query)
        results = client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=self.top_k,
        )

        if not results:
            return "No relevant documents found."

        return (
            f"[Score: {r.score:.3f}] {r.payload.get('text', '')}"
            for r in results
        )