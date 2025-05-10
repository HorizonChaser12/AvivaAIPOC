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
from fastapi.middleware.cors import CORSMiddleware # Import CORSMiddleware
from langchain.schema import Document
import datetime
import uvicorn
import os # For checking file existence

# Configure logging to file only
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='rag_system_api.log',  # Log to a specific file for the API
    filemode='a'
)
logger = logging.getLogger(__name__)

# Load environment variables
# Make sure you have a .env file with your GOOGLE_API_KEY or set it in your environment
if not load_dotenv():
    logger.warning("Could not load .env file. Ensure GOOGLE_API_KEY is set in your environment.")
elif not os.getenv("GOOGLE_API_KEY"):
    logger.warning("GOOGLE_API_KEY not found in .env file or environment.")


class AdaptiveRAGSystem:
    def __init__(self, excel_file_path: str, embedding_model: str = 'models/embedding-001',
                 llm_model: str = 'gemini-2.0-flash', temperature: float = 0.3, concise_prompt: bool = False,
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
        logger.info(f"Initializing models. Embedding: {embedding_model}, LLM: {llm_model}")
        try:
            self.embedding_model = GoogleGenerativeAIEmbeddings(model=embedding_model)
            self.llm = ChatGoogleGenerativeAI(model=llm_model, temperature=temperature, convert_system_message_to_human=True)
            logger.info("Google AI Models initialized successfully.")
        except Exception as e:
            logger.error(f"Fatal: Failed to initialize Google AI models: {e}. Ensure GOOGLE_API_KEY is correctly set and valid.")
            logger.error("The application might not function correctly without AI models.")
            # Depending on desired behavior, you might want to raise the error
            # or allow the system to run in a degraded (non-AI) mode if applicable.
            # For this RAG system, models are crucial, so we'll make them None and handle it.
            self.embedding_model = None
            self.llm = None
            # raise RuntimeError(f"Failed to initialize Google AI models: {e}") from e


        # Load data and build or load index
        self.data = None
        self.metadata = None
        self.index = None
        self.dimension = None # Will be set when index is built or loaded
        self.column_info = {}  # Store information about columns

        if self.embedding_model and self.llm: # Proceed only if models are initialized
            self._load_data() # This will also call _prepare_data and _analyze_columns

            # Check if persisted index exists and load it, otherwise build a new one
            try:
                logger.info(f"Attempting to load FAISS index from {self.index_file}...")
                self._load_index(self.index_file)
            except Exception as e:
                logger.warning(f"Could not load persisted index from {self.index_file} (Reason: {e}). Building a new index...")
                self._build_index() # Uses self.index_file by default internally
        else:
            logger.error("Skipping data loading and index building due to model initialization failure.")


        # Create response generation chain
        if self.llm: # Only create chain if LLM is available
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
                input_variables=["query", "context"], # Corrected from ["question", "context"] for the detailed prompt
                template=self.response_template
            )
            self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
            logger.info("LLMChain initialized.")
        else:
            self.chain = None
            logger.error("LLMChain could not be initialized because LLM is not available.")

    def _load_data(self):
        """Load data from Excel file and prepare for embedding"""
        logger.info(f"Loading data from {self.excel_file_path}...")
        if not os.path.exists(self.excel_file_path):
            logger.error(f"Excel file not found at path: {self.excel_file_path}")
            # Create a minimal dummy DataFrame to prevent crashes, but log error
            self.data = pd.DataFrame({'Error': [f'File not found: {self.excel_file_path}']})
            self._prepare_data() # Process this dummy data
            self.metadata = self.data.to_dict(orient='records')
            logger.warning("Proceeding with dummy data due to missing Excel file. RAG system will not be effective.")
            return

        try:
            excel_data = pd.read_excel(self.excel_file_path, sheet_name=None)
            self.data = None
            for sheet_name, df in excel_data.items():
                if not df.empty:
                    self.data = df
                    logger.info(f"Using sheet '{sheet_name}' with {len(df)} rows and {len(df.columns)} columns")
                    break
            
            if self.data is None:
                logger.error("No non-empty sheets found in the Excel file. Creating dummy data.")
                self.data = pd.DataFrame({'Error': ['No non-empty sheets in Excel file.']})
                # raise ValueError("No non-empty sheets found in the Excel file")

            self._prepare_data() # Cleans data, analyzes columns, creates 'combined_text'
            self.metadata = self.data.to_dict(orient='records') # Store original records for retrieval
            logger.info(f"Loaded {len(self.data)} records from Excel file.")
        except Exception as e:
            logger.error(f"Error loading data: {e}", exc_info=True)
            logger.warning("Creating dummy data due to data loading error. RAG system will not be effective.")
            self.data = pd.DataFrame({'Error': [f'Error loading data: {str(e)}']})
            self._prepare_data()
            self.metadata = self.data.to_dict(orient='records')
            # raise # Optionally re-raise

    def _prepare_data(self):
        """Prepare data for embedding and retrieval"""
        if self.data is None:
            logger.error("Cannot prepare data: self.data is None.")
            return

        # Replace NaN with empty strings
        self.data = self.data.fillna('')
        
        # Drop completely empty rows and columns
        self.data = self.data.dropna(how='all').dropna(axis=1, how='all')
        
        # Analyze columns (must be done before creating combined_text if it relies on column types)
        self._analyze_columns()
        
        # For embedding, treat all columns equally by combining them
        # Ensure 'combined_text' column itself is not included in the join if it somehow exists
        self.data['combined_text'] = self.data.apply(
            lambda row: ' '.join(f"{col}: {val}" for col, val in row.items() if val != '' and col != 'combined_text'),
            axis=1
        )
        logger.info("Data preparation complete. 'combined_text' field created.")

    def _analyze_columns(self):
        """Analyze columns to gather information about data types and content"""
        if self.data is None:
            logger.error("Cannot analyze columns: self.data is None.")
            return
        
        logger.info("Analyzing data columns...")
        columns = self.data.columns
        self.column_info = {} # Reset column info
        
        for col in columns:
            if col == 'combined_text': # Skip our generated column
                continue
                
            col_data = self.data[col]
            
            # Determine data type
            data_type = 'text' # Default
            if pd.api.types.is_numeric_dtype(col_data.infer_objects()): # Infer objects to catch numbers stored as strings
                data_type = 'numeric'
            elif self._is_date_column(col_data):
                data_type = 'date'
            
            # Calculate sparsity (proportion of empty or NaN values)
            # Ensure we count NaNs correctly after fillna('') might have turned them to empty strings
            empty_count = (col_data == '').sum() + col_data.isna().sum()
            sparsity = empty_count / max(1, len(col_data)) # Avoid division by zero for empty series
            
            # Calculate value diversity (proportion of unique values)
            # Ensure NaNs/empty strings are handled consistently in nunique
            unique_values = col_data.nunique(dropna=False) # Count NaNs/empty as a unique value if present
            value_diversity = unique_values / max(1, len(col_data))
            
            self.column_info[col] = {
                'data_type': data_type,
                'sparsity': sparsity,
                'value_diversity': value_diversity,
                'unique_values_count': unique_values
            }
            
            # Semantic type detection (example logic from original)
            if data_type == 'text':
                avg_len = col_data.astype(str).str.len().mean() if not col_data.empty else 0
                if value_diversity > 0.8: # High diversity
                    self.column_info[col]['semantic_type'] = 'description' if avg_len > 50 else 'identifier'
                elif value_diversity < 0.2 and unique_values < 20 : # Low diversity, few unique values
                     self.column_info[col]['semantic_type'] = 'category'
                     # Only store categories if there are a manageable number
                     self.column_info[col]['categories'] = col_data.unique().tolist() if unique_values < 20 else 'Too many to list'
                else:
                    self.column_info[col]['semantic_type'] = 'general_text'

            elif data_type == 'date':
                self.column_info[col]['semantic_type'] = 'date'
        logger.info(f"Column analysis complete: {self.column_info}")

    def _is_date_column(self, series: pd.Series) -> bool:
        """Check if a column appears to contain date values"""
        if series.empty:
            return False
        # Attempt to convert a sample to datetime, more robustly
        try:
            # Try converting non-null values only, up to a certain sample size
            sample = series.dropna().sample(min(len(series.dropna()), 100)) if not series.dropna().empty else pd.Series(dtype=object)
            if sample.empty and series.notna().any(): # If all were NaN but some original values exist
                 sample = series.loc[series.notna()].head(100)


            if sample.empty: return False # If still empty (all NaNs or truly empty series)

            # Attempt conversion, if a high percentage succeeds, it's likely a date column
            converted_sample = pd.to_datetime(sample, errors='coerce')
            # Check if more than 80% of the non-null sample converted successfully
            success_rate = converted_sample.notna().sum() / max(1, len(sample))
            return success_rate > 0.8
        except Exception:
            return False
            
    def _build_index(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Generate embeddings and build FAISS index with dynamic chunking options"""
        if self.data is None or 'combined_text' not in self.data.columns or self.embedding_model is None:
            logger.error("Cannot build FAISS index: Data, 'combined_text' column, or embedding model is not available.")
            # Create a dummy index to prevent downstream errors if essential
            self.dimension = 768 # A common embedding dimension, e.g. for 'models/embedding-001'
            logger.warning(f"Creating a dummy FAISS index with dimension {self.dimension} as fallback.")
            self.index = faiss.IndexFlatL2(self.dimension)
            # Add a single zero vector so index.ntotal is not 0, which can cause issues
            self.index.add(np.zeros((1, self.dimension), dtype='float32'))
            return

        logger.info(f"Building FAISS index from 'combined_text'. Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
        try:
            documents_content = self.data['combined_text'].tolist()

            if not documents_content or all(s.strip() == "" for s in documents_content):
                logger.warning("No valid text content found in 'combined_text' for embedding. Index will be empty or dummy.")
                self.dimension = self.embedding_model.client.get_embedding_dimensionality(self.embedding_model.model) if hasattr(self.embedding_model.client, 'get_embedding_dimensionality') else 768
                self.index = faiss.IndexFlatL2(self.dimension)
                self.index.add(np.zeros((1, self.dimension), dtype='float32'))
                return

            # Text splitting
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, separators=["\n\n", "\n", " ", ""], chunk_overlap=chunk_overlap
            )
            # Create Document objects for the splitter
            langchain_documents = [Document(page_content=doc) for doc in documents_content if doc.strip()]
            
            if not langchain_documents:
                logger.warning("No non-empty documents after filtering for text splitting. Index will be dummy.")
                self.dimension = 768 # Fallback dimension
                self.index = faiss.IndexFlatL2(self.dimension)
                self.index.add(np.zeros((1, self.dimension), dtype='float32'))
                return

            split_documents = text_splitter.split_documents(langchain_documents)
            
            if not split_documents:
                logger.warning("Text splitting resulted in zero documents. Index will be dummy.")
                self.dimension = 768
                self.index = faiss.IndexFlatL2(self.dimension)
                self.index.add(np.zeros((1, self.dimension), dtype='float32'))
                return

            logger.info(f"Generating embeddings for {len(split_documents)} document chunks...")
            doc_embeddings = self.embedding_model.embed_documents([doc.page_content for doc in split_documents])

            if not doc_embeddings or not doc_embeddings[0]:
                logger.error("Embedding process yielded no embeddings. Cannot build FAISS index.")
                self.dimension = 768 # Fallback
                self.index = faiss.IndexFlatL2(self.dimension)
                self.index.add(np.zeros((1, self.dimension), dtype='float32'))
                return

            self.dimension = len(doc_embeddings[0])
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(np.array(doc_embeddings, dtype='float32'))

            faiss.write_index(self.index, self.index_file)
            logger.info(f"Built and saved FAISS index to '{self.index_file}' with {self.index.ntotal} vectors, dimension {self.dimension}")
        except Exception as e:
            logger.error(f"Error building FAISS search index: {e}", exc_info=True)
            # Fallback to a dummy index
            self.dimension = 768 # Default if error before dimension is known
            logger.warning(f"Creating a dummy FAISS index with dimension {self.dimension} due to build error.")
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(np.zeros((1, self.dimension), dtype='float32'))
            # raise # Optionally re-raise

    def _load_index(self, index_file: str):
        """Load FAISS index from disk if it exists"""
        if not os.path.exists(index_file):
            logger.warning(f"FAISS index file {index_file} not found. A new one will be built if _build_index is called.")
            raise FileNotFoundError(f"FAISS index file {index_file} not found.")
        try:
            self.index = faiss.read_index(index_file)
            self.dimension = self.index.d # Set dimension from loaded index
            logger.info(f"Successfully loaded FAISS index from {index_file}. N_vectors: {self.index.ntotal}, Dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Error loading FAISS index from {index_file}: {e}", exc_info=True)
            raise # Re-raise to be caught by the constructor's try-except

    def retrieve(self, query: str, k: int = 3) -> List[Dict[Any, Any]]:
        """
        Retrieve the top k most relevant documents for the query.
        """
        if self.index is None or self.embedding_model is None or self.metadata is None:
            logger.warning("Cannot retrieve: FAISS index, embedding model, or metadata is not available.")
            return []
        if self.index.ntotal == 0 :
             logger.warning("Retrieval attempted but FAISS index is empty.")
             return []


        logger.info(f"Retrieving top {k} documents for query: {query[:50]}...")
        try:
            query_embedding = self.embedding_model.embed_query(query)
        except Exception as e:
            logger.error(f"Error embedding query '{query[:50]}...': {e}", exc_info=True)
            return []
        
        actual_k = min(k, self.index.ntotal)
        if actual_k == 0:
            logger.warning("No documents in index to retrieve.")
            return []
        if actual_k < k:
            logger.warning(f"Requested k={k} documents, but only {actual_k} available in index. Retrieving {actual_k}.")
            
        distances, indices = self.index.search(np.array([query_embedding], dtype='float32'), actual_k)
        
        results = []
        for i, doc_idx_in_faiss in enumerate(indices[0]):
            # The indices from FAISS correspond to the order of embeddings added,
            # which should match the order of `split_documents` if `_build_index` was successful.
            # However, self.metadata corresponds to the original, unsplit documents.
            # This RAG setup implies that FAISS indices map directly to self.metadata indices.
            # This is true if `split_documents` was NOT used and `documents_content` (from `self.data['combined_text']`)
            # was directly embedded. If `split_documents` *was* used, then `doc_idx_in_faiss` refers to a *chunk*,
            # not an original document in `self.metadata`.
            # The original code's `_build_index` embeds `split_documents` but `retrieve` uses `self.metadata`.
            # This is a common mismatch. For simplicity here, we assume FAISS indices map to metadata indices.
            # A more robust solution would store metadata alongside each chunk or map chunk indices back to original doc indices.

            # Assuming direct mapping for now, as per original structure:
            original_doc_idx = int(doc_idx_in_faiss) # FAISS returns int64, ensure it's Python int

            if 0 <= original_doc_idx < len(self.metadata):
                doc_content = self.metadata[original_doc_idx].copy() # Get the original document
                
                # Remove the combined text field from the result if it exists
                if 'combined_text' in doc_content:
                    del doc_content['combined_text']
                
                result = {
                    "content": doc_content, # This is the full original document content
                    "distance": float(distances[0][i]),
                    "similarity": 1 / (1 + float(distances[0][i])) # Convert L2 distance to a similarity score
                }
                results.append(result)
            else:
                logger.warning(f"Retrieved FAISS index {original_doc_idx} is out of bounds for metadata (len: {len(self.metadata)}). Skipping.")
        
        logger.info(f"Retrieved {len(results)} documents.")
        return results

    def format_retrieved_document(self, doc: Dict) -> str:
        """Format a retrieved document for better readability in the LLM context"""
        formatted_content = []
        for key, value in doc["content"].items():
            if key == 'combined_text': # Should have been removed already by retrieve
                continue
            # Ensure value is a string, format it, and handle empty/None values gracefully
            value_str = str(value) if value is not None else ""
            if value_str.strip(): # Only include if there's actual content
                formatted_key = key.replace('_', ' ').title()
                formatted_content.append(f"{formatted_key}: {value_str}")
        
        # Add similarity score to the context for the LLM
        formatted_content.append(f"RelevanceScore: {doc['similarity']:.4f}") # LLM might use this
        return "\n".join(formatted_content)

    def analyze_patterns(self, retrieved_docs: List[Dict]) -> Dict[str, Any]:
        """Analyze patterns across retrieved documents (e.g., common values, date ranges)"""
        if not retrieved_docs:
            return {"count": 0, "patterns": {}, "date_range": None}
            
        analysis = {
            "count": len(retrieved_docs),
            "patterns": {},
            "date_range": None # To store info about date spans if found
        }
        
        # Iterate through columns defined in self.column_info
        for col_name, info in self.column_info.items():
            # Focus on categorical or low-diversity text columns for common value patterns
            if info.get('semantic_type') == 'category' or (info.get('data_type') == 'text' and info.get('value_diversity', 1.0) < 0.5):
                value_counts = {}
                for doc in retrieved_docs:
                    value = doc["content"].get(col_name)
                    if value is not None and str(value).strip(): # Ensure value exists and is not empty
                        value_str = str(value)
                        value_counts[value_str] = value_counts.get(value_str, 0) + 1
                if value_counts:
                    analysis["patterns"][col_name] = value_counts
            
            # Date range analysis for date columns
            if info.get('data_type') == 'date':
                dates = []
                for doc in retrieved_docs:
                    date_val = doc["content"].get(col_name)
                    if date_val:
                        try:
                            # Convert to datetime, handling various possible formats if necessary
                            # pd.to_datetime is quite flexible
                            dt = pd.to_datetime(date_val, errors='coerce')
                            if pd.notna(dt):
                                dates.append(dt)
                        except Exception:
                            logger.debug(f"Could not parse date value '{date_val}' in column '{col_name}' during pattern analysis.")
                            pass # Ignore if a value can't be parsed
                
                if dates:
                    min_date = min(dates)
                    max_date = max(dates)
                    if analysis["date_range"] is None: # Take the first date column found
                        analysis["date_range"] = {
                            "column": col_name,
                            "min_date": min_date.strftime("%Y-%m-%d"),
                            "max_date": max_date.strftime("%Y-%m-%d"),
                            "span_days": (max_date - min_date).days
                        }
        logger.info(f"Pattern analysis complete: {analysis}")
        return analysis

    def generate_response(self, query: str, k: int = 3) -> Dict[str, Any]:
        """End-to-end RAG process: retrieve documents, analyze patterns, and generate response"""
        if not self.chain:
            logger.error("Cannot generate response: LLM chain is not initialized (likely due to LLM/API key issues).")
            return {
                "response": "I am currently unable to process your request due to an internal system issue. Please try again later.",
                "retrieved_docs": [],
                "pattern_analysis": {"count": 0}
            }

        logger.info(f"Starting generate_response for query: {query[:100]}..., k={k}")
        retrieved_docs = self.retrieve(query, k)

        if not retrieved_docs:
            logger.warning("No relevant documents found for the query.")
            return {
                "response": "I couldn't find any specific information related to your query in the available data.",
                "retrieved_docs": [],
                "pattern_analysis": {"count": 0}
            }

        # Format documents for context
        doc_contexts = [self.format_retrieved_document(doc) for doc in retrieved_docs]
        
        # Analyze patterns in the retrieved documents
        pattern_analysis = self.analyze_patterns(retrieved_docs)
        
        # Construct the full context string for the LLM
        context_parts = ["Retrieved Information Entries:"]
        context_parts.extend(doc_contexts) # Add individual document details

        # Add a summary of patterns if significant
        if pattern_analysis["count"] > 0:
            context_parts.append("\nSummary of Patterns Found:")
            context_parts.append(f"- Number of similar records found: {pattern_analysis['count']}")
            if pattern_analysis.get("date_range"):
                dr = pattern_analysis["date_range"]
                context_parts.append(f"- These records span from {dr['min_date']} to {dr['max_date']} (Column: {dr['column']}).")
            if pattern_analysis.get("patterns"):
                for col, val_counts in pattern_analysis["patterns"].items():
                    # Only show most frequent or if few categories
                    if len(val_counts) < 5 or any(c > 1 for c in val_counts.values()):
                        common_vals_str = ", ".join([f"'{val}' ({count} times)" for val, count in sorted(val_counts.items(), key=lambda item: item[1], reverse=True)[:3]]) # Top 3
                        context_parts.append(f"- Common values for '{col.replace('_',' ').title()}': {common_vals_str}.")
        
        final_context = "\n\n===\n\n".join(context_parts)
        logger.debug(f"Context for LLM: {final_context[:500]}...") # Log beginning of context

        # Generate response using LLM
        logger.info("Invoking LLMChain to generate response...")
        try:
            llm_response_dict = self.chain.invoke({"query": query, "context": final_context})
            response_text = llm_response_dict.get("text", "No response text generated by LLM.")
            logger.info("LLM response generated successfully.")
        except Exception as e:
            logger.error(f"Error during LLM chain invocation: {e}", exc_info=True)
            response_text = "I encountered an issue while trying to formulate a response based on the retrieved information."
        
        return {
            "response": response_text,
            "retrieved_docs": retrieved_docs, # Return the structured docs for frontend
            "pattern_analysis": pattern_analysis # Return pattern analysis for frontend
        }

    def get_column_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about detected columns and their characteristics"""
        return self.column_info

# --- FastAPI App Setup ---
app = FastAPI(
    title="Adaptive RAG System API",
    description="API for querying a RAG system backed by an Excel knowledge base.",
    version="1.0.0"
)

# --- CORS (Cross-Origin Resource Sharing) Middleware ---
# Allows your frontend (running on a different port/domain) to communicate with this API.
origins = [
    "http://localhost",         # General localhost
    "http://localhost:3000",    # Common for React dev server
    "http://localhost:8000",    # If frontend is served by FastAPI itself (e.g. static files)
    "http://localhost:8080",    # Common for other dev servers
    "http://127.0.0.1:5500",    # VS Code Live Server default
    "http://127.0.0.1",         # General 127.0.0.1
    "null",                     # For `file:///` origins (opening HTML directly in browser)
    "https://horizonchaser12.github.io/AvivaAIPOC/",  # <-- Add your GitHub Pages domain here
    # Add your deployed frontend's URL here if applicable
    # e.g., "https://your-frontend-domain.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Specific origins allowed
    # allow_origins=["*"], # Alternatively, allow all origins (less secure)
    allow_credentials=True, # Allow cookies
    allow_methods=["*"],    # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],    # Allow all headers
)

# --- Pydantic Models for API Request and Response ---
class QueryRequest(BaseModel):
    query: str
    k: int = 3 # Default number of documents to retrieve

class QueryResponse(BaseModel):
    response: str
    retrieved_docs: List[Dict[Any, Any]] # List of retrieved document dictionaries
    pattern_analysis: Dict[str, Any]    # Dictionary of pattern analysis

# Global variable to hold the RAG system instance
rag_system_instance: Optional[AdaptiveRAGSystem] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system when the FastAPI application starts."""
    global rag_system_instance
    logger.info("FastAPI application startup: Initializing RAG system...")
    try:
        # Ensure 'hehe.xlsx' (or your chosen file) is in the correct path relative to where the script is run
        # Or use an absolute path.
        excel_file = "Defects.xlsx"
        if not os.path.exists(excel_file):
            logger.warning(f"Knowledge base file '{excel_file}' not found at startup. Creating a dummy file for system to run.")
            # Create a dummy excel file if it doesn't exist, so the system can start
            # This is for demonstration; in production, you'd ensure the file exists.
            dummy_df = pd.DataFrame({
                'ID': [1, 2, 3],
                'Problem Description': ['Login button not working on Chrome', 'Website loads very slowly after 5 PM', 'Error 503 when submitting payment'],
                'Solution': ['Cleared browser cache and cookies, worked.', 'Identified high traffic, scaled up server resources.', 'Payment gateway API was down, issue resolved after they fixed it.'],
                'Date Reported': [datetime.date(2023,1,10), datetime.date(2023,2,15), datetime.date(2023,3,20)],
                'Status': ['Closed', 'Closed', 'Closed']
            })
            dummy_df.to_excel(excel_file, index=False)
            logger.info(f"Created dummy '{excel_file}'.")


        rag_system_instance = AdaptiveRAGSystem(excel_file_path=excel_file)
        logger.info("RAG system initialized successfully.")
    except Exception as e:
        logger.error(f"CRITICAL: Failed to initialize RAG system during startup: {e}", exc_info=True)
        rag_system_instance = None # Ensure it's None if initialization fails
        # Depending on policy, you might want the app to fail to start entirely:
        # raise RuntimeError(f"RAG system initialization failed: {e}") from e

@app.post("/query", response_model=QueryResponse)
async def process_query_endpoint(request: QueryRequest):
    """
    Endpoint to process a user query using the RAG system.
    """
    if rag_system_instance is None:
        logger.error("API call to /query failed: RAG system is not available (initialization may have failed).")
        raise HTTPException(status_code=503, detail="RAG system is not initialized or currently unavailable. Please try again later.")

    logger.info(f"Received API query: '{request.query}', k={request.k}")
    try:
        result = rag_system_instance.generate_response(request.query, request.k)
        
        # FastAPI will automatically handle serialization of standard Python types,
        # including Pydantic models and basic dicts/lists.
        # Custom objects like pd.Timestamp need to be converted.
        # The `make_serializable` function from previous context is good for this.

        def make_serializable(obj):
            if isinstance(obj, (datetime.date, datetime.datetime, pd.Timestamp)):
                return obj.isoformat()
            if isinstance(obj, dict):
                return {make_serializable(k): make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [make_serializable(i) for i in obj]
            if isinstance(obj, (np.ndarray, np.generic)): # Handle numpy types
                return obj.tolist()
            # Add other type conversions if necessary
            return obj

        # Apply serialization to ensure all parts of the response are JSON-friendly
        serialized_result = {
            "response": make_serializable(result["response"]),
            "retrieved_docs": make_serializable(result["retrieved_docs"]),
            "pattern_analysis": make_serializable(result["pattern_analysis"])
        }
        
        logger.info(f"Successfully processed query. Sending response.")
        return JSONResponse(content=serialized_result)
    except Exception as e:
        logger.error(f"Error processing query via API: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while processing your query: {str(e)}")

@app.get("/health", summary="Health Check", description="Returns the status of the API and RAG system.")
async def health_check():
    status = "ok"
    rag_status = "initialized"
    if rag_system_instance is None:
        status = "error"
        rag_status = "not_initialized_or_failed"
    elif rag_system_instance.llm is None or rag_system_instance.embedding_model is None:
        status = "degraded"
        rag_status = "models_not_loaded"
    elif rag_system_instance.index is None or rag_system_instance.index.ntotal == 0 :
        status = "degraded"
        rag_status = "index_not_loaded_or_empty"


    return {
        "api_status": status,
        "rag_system_status": rag_status,
        "timestamp": datetime.datetime.utcnow().isoformat()
    }

# --- Main execution block to run the FastAPI server ---
if __name__ == "__main__":
    logger.info("Starting FastAPI server using Uvicorn...")
    # host="0.0.0.0" makes the server accessible from other devices on the network.
    # port=8000 is a common port for development.
    # reload=True is useful for development as it automatically restarts the server on code changes.
    # However, for production, you'd typically use a process manager like Gunicorn.
    uvicorn.run("Project:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
    # Note: If your file is named something else, e.g. `app.py`, use "app:app".
    # The "main_api:app" string means "in the file main_api.py, find the FastAPI instance named app".
