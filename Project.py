import logging
import numpy as np
import pandas as pd
import faiss
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import List, Dict, Any, Optional, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from langchain.schema import Document
import datetime
import uvicorn

# Configure logging to file only
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='rag_system.log',  # Log to file instead of console
    filemode='a'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class AdaptiveRAGSystem:
    def __init__(self, excel_file_path: str, embedding_model: str = 'models/embedding-001', 
                 llm_model: str = 'gemini-2.0-flash-lite', temperature: float = 0.3, concise_prompt: bool = False,
                 index_file: str = "faiss_index.bin"):
        """
        Initialize the RAG system with specified models and data source.
        
        Args:
            excel_file_path: Path to the Excel file containing the knowledge base
            embedding_model: Google embedding model to use
            llm_model: Google LLM model to use for response generation
            temperature: Temperature for the LLM (higher = more creative)
            concise_prompt: Whether to use a concise prompt template
            index_file: Path to the FAISS index file for persistence
        """
        self.excel_file_path = excel_file_path
        self.concise_prompt = concise_prompt
        self.index_file = index_file

        # Initialize models
        logger.info("Initializing models...")
        self.embedding_model = GoogleGenerativeAIEmbeddings(model=embedding_model)
        self.llm = ChatGoogleGenerativeAI(model=llm_model, temperature=temperature)

        # Load data and build or load index
        self.data = None
        self.metadata = None
        self.index = None
        self.dimension = None
        self.column_info = {}  # Store information about columns

        self._load_data()

        # Check if persisted index exists
        try:
            self._load_index(index_file)
        except Exception:
            logger.info("Persisted index not found. Building a new index...")
            self._build_index()

        # Create response generation chain
        if concise_prompt:
            self.response_template = """
            You are an AI Assistant. Given the following context:{context}Answer the following question:{question}Assistant:
            """
        else:
            self.response_template = """
            You are a helpful technical support assistant with expertise in software issues. Based on the user query and the relevant historical defect data provided,
            give a comprehensive yet concise response that addresses the user's issue in natural language.
            
            User Query: {query}
            
            Relevant Historical Defect Data:
            {context}
            
            Provide a professional, conversational response that includes:
            1. A clear summary of the identified issue
            2. When this type of issue has occurred in the past (dates/frequency if available)
            3. The root causes that were identified for similar issues
            4. The solution or resolution that was most effective, explained in detail
            5. Any preventative measures that could avoid this issue in the future
            6. If multiple similar issues were found, explain any patterns or common factors
            
            Make your response conversational and easy to understand. Avoid technical jargon unless necessary and explain any complex terms. 
            Format your response in clear paragraphs rather than as a list of facts.
            
            Response:
            """

        self.prompt = PromptTemplate(
            input_variables=["query", "context"],
            template=self.response_template
        )

        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def _load_data(self):
        """Load data from Excel file and prepare for embedding"""
        logger.info(f"Loading data from {self.excel_file_path}...")
        try:
            # Try to load the Excel file
            excel_data = pd.read_excel(self.excel_file_path, sheet_name=None)
            
            # Find the first non-empty sheet
            self.data = None
            for sheet_name, df in excel_data.items():
                if not df.empty:
                    self.data = df
                    logger.info(f"Using sheet '{sheet_name}' with {len(df)} rows and {len(df.columns)} columns")
                    break
            
            if self.data is None:
                raise ValueError("No non-empty sheets found in the Excel file")
            
            # Clean up data
            self._prepare_data()
            
            # Store original records for retrieval
            self.metadata = self.data.to_dict(orient='records')
            
            logger.info(f"Loaded {len(self.data)} records from Excel file")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _prepare_data(self):
        """Prepare data for embedding and retrieval"""
        # Replace NaN with empty strings
        self.data = self.data.fillna('')
        
        # Drop completely empty rows and columns
        self.data = self.data.dropna(how='all').dropna(axis=1, how='all')
        
        # Analyze columns
        self._analyze_columns()
        
        # For embedding, treat all columns equally by combining them
        self.data['combined_text'] = self.data.apply(
            lambda row: ' '.join(f"{col}: {val}" for col, val in row.items() if val != ''),
            axis=1
        )
    
    def _analyze_columns(self):
        """Analyze columns to gather information about data types and content"""
        columns = self.data.columns
        
        for col in columns:
            # Skip if it's our generated column
            if col == 'combined_text':
                continue
                
            col_data = self.data[col]
            
            # Determine data type
            if pd.api.types.is_numeric_dtype(col_data):
                data_type = 'numeric'
            elif self._is_date_column(col_data):
                data_type = 'date'
            else:
                data_type = 'text'
                
            # Calculate sparsity (proportion of empty values)
            non_empty = (col_data != '').sum()
            sparsity = 1 - (non_empty / len(col_data))
            
            # Calculate value diversity (proportion of unique values)
            unique_values = col_data.nunique()
            value_diversity = unique_values / max(1, len(col_data))
            
            # Store this information
            self.column_info[col] = {
                'data_type': data_type,
                'sparsity': sparsity,
                'value_diversity': value_diversity,
                'unique_values': unique_values
            }
            
            # For highly diverse text columns, they're likely descriptions or unique identifiers
            if data_type == 'text' and value_diversity > 0.8:
                if col_data.str.len().mean() > 50:
                    self.column_info[col]['semantic_type'] = 'description'
                else:
                    self.column_info[col]['semantic_type'] = 'identifier'
            
            # For date columns, note this for later use
            if data_type == 'date':
                self.column_info[col]['semantic_type'] = 'date'
                
            # For low diversity text columns, they may be categories or status values
            if data_type == 'text' and value_diversity < 0.2:
                self.column_info[col]['semantic_type'] = 'category'
                self.column_info[col]['categories'] = col_data.unique().tolist()
    
    def _is_date_column(self, series):
        """Check if a column appears to contain date values"""
        # Try to convert to datetime
        try:
            pd.to_datetime(series, errors='coerce')
            # If more than 80% of values converted successfully, consider it a date column
            return pd.to_datetime(series, errors='coerce').notna().mean() > 0.8
        except:
            return False
            
    def _build_index(self, chunk_size: int = 1000, chunk_overlap: int = 200, index_file: str = "faiss_index.bin"):
        """Generate embeddings and build FAISS index with dynamic chunking options"""
        logger.info("Building knowledge base...")
        try:
            # Use the combined text of all columns for embedding
            documents = self.data['combined_text'].tolist()

            # Handle empty documents
            if not documents:
                raise ValueError("No valid documents found for embedding")

            # Generate embeddings
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, separators=["\n", "\n\n"], chunk_overlap=chunk_overlap
            )
            split_documents = text_splitter.split_documents([Document(page_content=doc) for doc in documents])
            doc_embeddings = self.embedding_model.embed_documents([doc.page_content for doc in split_documents])

            # Initialize FAISS index
            self.dimension = len(doc_embeddings[0])
            self.index = faiss.IndexFlatL2(self.dimension)

            # Add embeddings to the index
            self.index.add(np.array(doc_embeddings))

            # Save the index to disk
            faiss.write_index(self.index, index_file)
            logger.info(f"Built and saved FAISS index with {len(documents)} documents, dimension {self.dimension}")
        except Exception as e:
            logger.error(f"Error building search index: {e}")
            raise

    def _load_index(self, index_file: str = "faiss_index.bin"):
        """Load FAISS index from disk if it exists"""
        try:
            self.index = faiss.read_index(index_file)
            logger.info(f"Loaded FAISS index from {index_file}")
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            raise

    def retrieve(self, query: str, k: int = 3) -> List[Dict[Any, Any]]:
        """
        Retrieve the top k most relevant documents for the query
        
        Args:
            query: User query text
            k: Number of documents to retrieve
        
        Returns:
            List of retrieved documents with metadata and similarity scores
        """
        logger.info(f"Retrieving top {k} documents for query: {query[:50]}...")
        
        # Generate embedding for the query
        query_embedding = self.embedding_model.embed_query(query)
        
        # Adjust k if we have fewer documents than requested
        actual_k = min(k, self.index.ntotal)
        if actual_k < k:
            logger.warning(f"Only {actual_k} documents available, retrieving all")
        
        # Search the index
        distances, indices = self.index.search(np.array([query_embedding]), actual_k)
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.metadata):  # Ensure index is valid
                doc = self.metadata[idx].copy()
                
                # Remove the combined text field from the result
                if 'combined_text' in doc:
                    del doc['combined_text']
                
                result = {
                    "content": doc,
                    "distance": float(distances[0][i]),
                    "similarity": 1 / (1 + float(distances[0][i]))  # Convert distance to similarity score
                }
                results.append(result)
        
        logger.info(f"Retrieved {len(results)} documents")
        return results
    
    def format_retrieved_document(self, doc: Dict) -> str:
        """Format a retrieved document for better readability"""
        # Format the document content for display
        formatted_content = []
        
        # Process each field in the document
        for key, value in doc["content"].items():
            # Skip the combined text field
            if key == 'combined_text':
                continue
                
            # Format the value based on its type
            if isinstance(value, (int, float)) or (isinstance(value, str) and value.strip()):
                formatted_key = key.replace('_', ' ').title()
                formatted_content.append(f"{formatted_key}: {value}")
        
        # Add similarity score
        formatted_content.append(f"Relevance: {doc['similarity']:.4f}")
        
        return "\n".join(formatted_content)
    
    def analyze_patterns(self, retrieved_docs: List[Dict]) -> Dict[str, Any]:
        """
        Analyze patterns across retrieved documents
        
        Args:
            retrieved_docs: List of retrieved document dictionaries
        
        Returns:
            Dictionary containing pattern analysis
        """
        if not retrieved_docs:
            return {"count": 0}
            
        analysis = {
            "count": len(retrieved_docs),
            "patterns": {},
            "date_range": None
        }
        
        # Find columns that might contain useful patterns
        for col, info in self.column_info.items():
            # Skip columns with too many unique values (likely not categorizations)
            if info.get('value_diversity', 1.0) > 0.5 and info.get('data_type') != 'date':
                continue
                
            # Count occurrences of values
            value_counts = {}
            for doc in retrieved_docs:
                value = doc["content"].get(col)
                if value:
                    value_counts[value] = value_counts.get(value, 0) + 1
            
            # Add to patterns if we found anything
            if value_counts:
                analysis["patterns"][col] = value_counts
        
        # Look for date ranges
        date_cols = [col for col, info in self.column_info.items() if info.get('data_type') == 'date']
        if date_cols:
            for date_col in date_cols:
                dates = []
                for doc in retrieved_docs:
                    date_val = doc["content"].get(date_col)
                    if date_val:
                        try:
                            date = pd.to_datetime(date_val)
                            dates.append(date)
                        except:
                            pass
                
                if dates:
                    min_date = min(dates)
                    max_date = max(dates)
                    analysis["date_range"] = {
                        "column": date_col,
                        "min_date": min_date.strftime("%Y-%m-%d"),
                        "max_date": max_date.strftime("%Y-%m-%d"),
                        "span_days": (max_date - min_date).days
                    }
                    break  # Use the first date column with valid dates
        
        return analysis
    
    # Add detailed logging to the generate_response method
    def generate_response(self, query: str, k: int = 3) -> Dict[str, Any]:
        """
        End-to-end RAG process: retrieve documents and generate response

        Args:
            query: User query text
            k: Number of documents to retrieve

        Returns:
            Dictionary containing the response and retrieved context
        """
        try:
            logger.info(f"Starting generate_response for query: {query}")

            # Retrieve relevant documents
            retrieved_docs = self.retrieve(query, k)

            if not retrieved_docs:
                logger.warning("No relevant documents found.")
                return {
                    "response": "I couldn't find any relevant information to answer your query.",
                    "retrieved_docs": [],
                    "pattern_analysis": {"count": 0}
                }

            # Format documents for context
            doc_contexts = [self.format_retrieved_document(doc) for doc in retrieved_docs]

            # Analyze patterns
            pattern_analysis = self.analyze_patterns(retrieved_docs)

            # Add pattern information to context
            context_sections = doc_contexts.copy()

            if pattern_analysis["count"] > 1:
                pattern_text = []
                pattern_text.append(f"Found {pattern_analysis['count']} similar records.")

                # Add date range if available
                if pattern_analysis.get("date_range"):
                    date_range = pattern_analysis["date_range"]
                    pattern_text.append(
                        f"Date range: {date_range['min_date']} to {date_range['max_date']} "
                        f"({date_range['span_days']} days span)"
                    )

                # Add common patterns if available
                for col, values in pattern_analysis.get("patterns", {}).items():
                    if len(values) > 5:
                        continue

                    readable_col = col.replace('_', ' ').title()
                    value_counts = sorted(values.items(), key=lambda x: x[1], reverse=True)
                    value_str = ", ".join([f"{val} ({count})" for val, count in value_counts])
                    pattern_text.append(f"Common {readable_col}: {value_str}")

                context_sections.append("\n".join(pattern_text))

            # Join all context sections
            context = "\n\n===\n\n".join(context_sections)

            # Generate response using LLM
            logger.info("Generating response using LLM...")
            response = self.chain.invoke({"query": query, "context": context})

            logger.info("Response generated successfully.")
            return {
                "response": response["text"],
                "retrieved_docs": retrieved_docs,
                "pattern_analysis": pattern_analysis
            }

        except Exception as e:
            logger.error(f"Error in generate_response: {e}")
            raise
    
    def get_column_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about detected columns
        
        Returns:
            Dictionary with column information
        """
        return self.column_info

def main():
    """Main entry point for the script"""
    logger.info("\n[bold blue]Defect Knowledge Explorer[/bold blue] - Powered by RAG\n")
    
    # Check for required packages
    try:
        import rich
        import pandas
        import faiss
        import langchain_google_genai
    except ImportError as e:
        logger.error(f"Missing required package: {str(e)}")
        logger.info("Please install required packages using: pip install -r requirements.txt")
        return
    
    # Get file path from user or use default
    default_path = "Defects.xlsx"  # Changed to match the file name in the logs
    
    logger.info(f"Enter the path to your Excel file (or press Enter for '{default_path}')")
    file_path = input("> ").strip()
    
    if not file_path:
        file_path = default_path
    

    # Initialize the RAG system
    try:
        logger.info(f"Initializing system with {file_path}...")
        rag = AdaptiveRAGSystem(file_path)
        
        # Show basic column information
        column_info = rag.get_column_info()
        
        logger.info("Successfully loaded data with the following structure:")
        logger.info("Column Information:")
        for col, info in column_info.items():
            logger.info(f"Column: {col}, Type: {info.get('data_type', 'unknown')}, Semantic Type: {info.get('semantic_type', 'general')}, Unique Values: {info.get('unique_values', 'N/A')}, Sparsity: {info.get('sparsity', 0):.2f}")
        
        logger.info("System is ready for your questions!")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.info("Please check that your Excel file exists and contains valid data.")

# Define the FastAPI app
app = FastAPI()

# Define request and response models
class QueryRequest(BaseModel):
    query: str
    k: int = 3

class QueryResponse(BaseModel):
    response: str
    retrieved_docs: list
    pattern_analysis: dict

# Initialize the RAG system globally
rag_system = None

@app.on_event("startup")
def initialize_rag_system():
    global rag_system
    try:
        file_path = "Defects.xlsx"  # Default file path
        rag_system = AdaptiveRAGSystem(file_path)
        logger.info("RAG system initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        raise RuntimeError("Failed to initialize RAG system.")

# Modify the process_query function to ensure all data is JSON serializable
@app.post("/query", response_model=QueryResponse)
def process_query(request: QueryRequest):
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system is not initialized.")

    try:
        logger.info(f"Received query: {request.query}")
        result = rag_system.generate_response(request.query, request.k)

        # Convert any Timestamp objects to strings, including dictionary keys
        def make_serializable(obj):
            if isinstance(obj, (datetime.date, datetime.datetime, pd.Timestamp)):
                return obj.isoformat()
            if isinstance(obj, dict):
                return {make_serializable(k): make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [make_serializable(i) for i in obj]
            return obj

        serialized_result = {
            "response": result["response"],
            "retrieved_docs": [
                {"content": {k: make_serializable(v) for k, v in doc["content"].items()}, "distance": doc["distance"], "similarity": doc["similarity"]}
                for doc in result["retrieved_docs"]
            ],
            "pattern_analysis": make_serializable(result["pattern_analysis"])
        }

        logger.info(f"Generated response: {serialized_result}")
        return JSONResponse(content=serialized_result)
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail="Error processing query.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)