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
if not load_dotenv():
    logger.warning("Could not load .env file. Ensure GOOGLE_API_KEY is set in your environment.")
elif not os.getenv("GOOGLE_API_KEY"):
    logger.warning("GOOGLE_API_KEY not found in .env file or environment.")

# --- Utility function for JSON serialization ---
def make_serializable(obj: Any) -> Any:
    """
    Recursively converts non-serializable objects (like datetime, numpy types)
    in a data structure to JSON-serializable types.
    """
    if isinstance(obj, (datetime.date, datetime.datetime, pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {make_serializable(k): make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(i) for i in obj]
    if isinstance(obj, (np.ndarray, np.generic)): # Handle numpy types
        return obj.tolist()
    return obj

class AdaptiveRAGSystem:
    def __init__(self, excel_file_path: str, embedding_model: str = 'models/embedding-001',
                 llm_model: str = 'gemini-2.0-flash', temperature: float = 0.7, concise_prompt: bool = False,
                 index_file: str = "faiss_index.bin"):
        self.excel_file_path = excel_file_path
        self.concise_prompt = concise_prompt # This flag is now less impactful due to the new prompt structure
        self.index_file = index_file
        self.chunk_to_original_doc_mapping: List[int] = []

        logger.info(f"Initializing models. Embedding: {embedding_model}, LLM: {llm_model}")
        try:
            self.embedding_model = GoogleGenerativeAIEmbeddings(model=embedding_model)
            self.llm = ChatGoogleGenerativeAI(model=llm_model, temperature=temperature, convert_system_message_to_human=True)
            logger.info("Google AI Models initialized successfully.")
        except Exception as e:
            logger.error(f"Fatal: Failed to initialize Google AI models: {e}.")
            self.embedding_model = None
            self.llm = None

        self.data = None
        self.metadata: Optional[List[Dict[Any, Any]]] = None
        self.index = None
        self.dimension = None
        self.column_info: Dict[str, Dict[str, Any]] = {}

        if self.embedding_model and self.llm:
            self._load_data()
            try:
                logger.info(f"Attempting to load FAISS index from {self.index_file}...")
                self._load_index(self.index_file)
                if not self.chunk_to_original_doc_mapping and self.data is not None and 'combined_text' in self.data.columns:
                    logger.info("Re-populating chunk_to_original_doc_mapping after loading index.")
                    self._populate_chunk_mapping_from_data()
            except Exception as e:
                logger.warning(f"Could not load persisted index from {self.index_file} (Reason: {e}). Building new index...")
                self._build_index()
        else:
            logger.error("Skipping data loading and index building due to model initialization failure.")

        if self.llm:
            # New comprehensive prompt template
            self.response_template = """
            You are a helpful technical support assistant. Your goal is to provide comprehensive and accurate answers based on the information available.

            Here's an overview of the dataset you are working with:
            {dataset_overview}

            For the user's specific query, the following documents have been retrieved as potentially relevant:
            {retrieved_documents_context}

            Additionally, here's an analysis of patterns found within these retrieved documents:
            {pattern_analysis_summary}

            User Query: {query}

            Based on all the information above (the dataset overview, the specific retrieved documents, and the pattern analysis),
            provide a professional, conversational response that addresses the user's query.
            If the query is general, use the dataset overview more. If specific, focus on the retrieved documents.
            Ensure your response includes:
            1. A clear summary of the identified issue or topic from the query.
            2. Relevant information from the dataset, citing document IDs if referring to specific retrieved documents.
            3. Insights from past occurrences, root causes, and solutions if applicable and found in the data.
            4. Any common factors or patterns if they are significant.
            5. Preventative measures or recommendations if appropriate.

            Make your response easy to understand. Explain complex terms if necessary.
            Format your response in clear paragraphs.

            Response:
            """
            self.prompt = PromptTemplate(
                input_variables=["query", "dataset_overview", "retrieved_documents_context", "pattern_analysis_summary"],
                template=self.response_template
            )
            self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
            logger.info("LLMChain initialized with new comprehensive prompt.")
        else:
            self.chain = None
            logger.error("LLMChain could not be initialized because LLM is not available.")

    def _generate_dataset_overview_summary(self) -> str:
        """Generates a textual summary of the dataset's structure."""
        if self.data is None or self.metadata is None:
            return "Dataset information is currently unavailable."

        num_records = len(self.metadata)
        summary_parts = [f"The dataset contains {num_records} records (e.g., rows or entries)."]

        if not self.column_info:
            summary_parts.append("Column details are not analyzed.")
            return "\n".join(summary_parts)

        summary_parts.append("It has the following columns:")
        for col_name, info in self.column_info.items():
            col_desc = f"- '{col_name}': Type: {info.get('data_type', 'N/A')}"
            if 'semantic_type' in info:
                col_desc += f", Semantic Role: {info.get('semantic_type')}"
            if 'categories' in info and isinstance(info['categories'], list) and info['categories']:
                preview_cats = info['categories'][:3]
                etc_cats = "..." if len(info['categories']) > 3 else ""
                col_desc += f" (e.g., {', '.join(map(str, preview_cats))}{etc_cats})"
            summary_parts.append(col_desc)
        
        return "\n".join(summary_parts)

    def _load_data(self):
        logger.info(f"Loading data from {self.excel_file_path}...")
        if not os.path.exists(self.excel_file_path):
            logger.error(f"Excel file not found: {self.excel_file_path}")
            self.data = pd.DataFrame({'Error': [f'File not found: {self.excel_file_path}']})
            self._prepare_data() # Process this dummy data
            self.metadata = self.data.to_dict(orient='records')
            logger.warning("Proceeding with dummy data due to missing Excel file.")
            return

        try:
            excel_data = pd.read_excel(self.excel_file_path, sheet_name=None)
            self.data = None
            for sheet_name, df in excel_data.items():
                if not df.empty:
                    self.data = df
                    logger.info(f"Using sheet '{sheet_name}' ({len(df)}x{len(df.columns)})")
                    break
            if self.data is None:
                logger.error("No non-empty sheets in Excel. Creating dummy data.")
                self.data = pd.DataFrame({'Error': ['No non-empty sheets in Excel.']})

            self._prepare_data()
            self.metadata = self.data.to_dict(orient='records')
            logger.info(f"Loaded {len(self.data)} records.")
        except Exception as e:
            logger.error(f"Error loading data: {e}", exc_info=True)
            self.data = pd.DataFrame({'Error': [f'Error loading data: {str(e)}']})
            self._prepare_data()
            self.metadata = self.data.to_dict(orient='records')

    def _prepare_data(self):
        if self.data is None:
            logger.error("Cannot prepare data: self.data is None.")
            return
        self.data = self.data.fillna('')
        self.data = self.data.dropna(how='all').dropna(axis=1, how='all')
        self._analyze_columns() # Analyze before creating combined_text
        self.data['combined_text'] = self.data.apply(
            lambda row: ' '.join(f"{col}: {val}" for col, val in row.items() if str(val).strip() != '' and col != 'combined_text'),
            axis=1
        )
        logger.info("Data preparation complete. 'combined_text' created.")

    def _analyze_columns(self):
        if self.data is None: return
        logger.info("Analyzing data columns...")
        self.column_info = {}
        for col in self.data.columns:
            if col == 'combined_text': continue # Skip our generated column
            col_data = self.data[col].astype(str) # Treat all as string for consistent analysis here
            
            # Determine data type more robustly before casting to string for analysis
            original_col_data = self.data[col]
            data_type = 'text' # Default
            if pd.api.types.is_numeric_dtype(original_col_data.infer_objects()):
                data_type = 'numeric'
            elif self._is_date_column(original_col_data):
                data_type = 'date'
            
            empty_count = (original_col_data.isna()).sum() + (original_col_data.astype(str) == '').sum()
            sparsity = empty_count / max(1, len(original_col_data))
            unique_values = original_col_data.nunique(dropna=False)
            value_diversity = unique_values / max(1, len(original_col_data))
            
            self.column_info[col] = {
                'data_type': data_type, 'sparsity': sparsity,
                'value_diversity': value_diversity, 'unique_values_count': unique_values
            }
            if data_type == 'text':
                avg_len = col_data.str.len().mean() if not col_data.empty else 0
                if value_diversity > 0.8 and unique_values > 0.8 * len(original_col_data) : self.column_info[col]['semantic_type'] = 'identifier' if avg_len < 50 else 'description'
                elif value_diversity < 0.2 and unique_values < 20 :
                     self.column_info[col]['semantic_type'] = 'category'
                     # Store unique categories if few and diverse enough
                     if unique_values > 0 : # only if there are actual values
                        self.column_info[col]['categories'] = original_col_data.dropna().unique().tolist() if unique_values < 20 else 'Too many to list'
                else: self.column_info[col]['semantic_type'] = 'general_text'
            elif data_type == 'date': self.column_info[col]['semantic_type'] = 'date'
        logger.info(f"Column analysis complete: {self.column_info}")

    def _is_date_column(self, series: pd.Series) -> bool:
        if series.empty: return False
        # Try to convert a sample to datetime
        try:
            # Consider only non-null values for sampling
            non_null_series = series.dropna()
            if non_null_series.empty: return False # All values are null

            sample_size = min(len(non_null_series), 20) # Sample up to 20 non-null values
            sample = non_null_series.sample(sample_size)
            
            # Attempt conversion, if a high percentage succeeds, it's likely a date column
            converted_sample = pd.to_datetime(sample, errors='coerce')
            # Check if more than 80% of the non-null sample converted successfully
            success_rate = converted_sample.notna().sum() / max(1, len(sample))
            return success_rate > 0.8
        except Exception:
            return False # Error during sampling or conversion

    def _populate_chunk_mapping_from_data(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        if self.data is None or 'combined_text' not in self.data.columns:
            logger.warning("Cannot populate chunk mapping: Data or 'combined_text' column unavailable.")
            self.chunk_to_original_doc_mapping = []
            return

        documents_content = self.data['combined_text'].tolist()
        if not any(s.strip() for s in documents_content): # Check if all are empty/whitespace
            self.chunk_to_original_doc_mapping = []
            logger.warning("No valid content in 'combined_text' to populate chunk mapping.")
            return
            
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, separators=["\n\n", "\n", " ", ""], chunk_overlap=chunk_overlap
        )
        
        langchain_documents = []
        for i, doc_content_str in enumerate(documents_content):
            if doc_content_str.strip(): # Ensure content is not just whitespace
                langchain_documents.append(Document(page_content=doc_content_str, metadata={"original_doc_index": i}))
        
        if not langchain_documents:
            self.chunk_to_original_doc_mapping = []
            logger.warning("No langchain documents created after filtering for chunk mapping.")
            return
            
        split_documents = text_splitter.split_documents(langchain_documents)
        
        self.chunk_to_original_doc_mapping = [doc.metadata["original_doc_index"] for doc in split_documents if "original_doc_index" in doc.metadata]
        logger.info(f"Populated chunk_to_original_doc_mapping with {len(self.chunk_to_original_doc_mapping)} entries.")


    def _build_index(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        if self.data is None or 'combined_text' not in self.data.columns or self.embedding_model is None:
            logger.error("Cannot build FAISS index: Data, 'combined_text', or embedding model not available.")
            self.dimension = 768 # Default
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(np.zeros((1, self.dimension), dtype='float32')) # Add dummy vector
            self.chunk_to_original_doc_mapping = []
            return

        logger.info(f"Building FAISS index. Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
        try:
            documents_content = self.data['combined_text'].tolist()
            if not any(s.strip() for s in documents_content):
                logger.warning("No valid text content for embedding. Index will be dummy.")
                self.dimension = 768 
                self.index = faiss.IndexFlatL2(self.dimension)
                self.index.add(np.zeros((1, self.dimension), dtype='float32'))
                self.chunk_to_original_doc_mapping = []
                return

            from langchain.text_splitter import RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, separators=["\n\n", "\n", " ", ""], chunk_overlap=chunk_overlap
            )
            
            langchain_documents = []
            for i, doc_content_str in enumerate(documents_content):
                if doc_content_str.strip():
                    langchain_documents.append(Document(page_content=doc_content_str, metadata={"original_doc_index": i}))
            
            if not langchain_documents:
                logger.warning("No non-empty documents for text splitting. Index will be dummy.")
                self.dimension = 768; self.index = faiss.IndexFlatL2(self.dimension); self.index.add(np.zeros((1, self.dimension), dtype='float32')); self.chunk_to_original_doc_mapping = []; return

            split_documents = text_splitter.split_documents(langchain_documents)
            if not split_documents:
                logger.warning("Text splitting resulted in zero documents. Index will be dummy.")
                self.dimension = 768; self.index = faiss.IndexFlatL2(self.dimension); self.index.add(np.zeros((1, self.dimension), dtype='float32')); self.chunk_to_original_doc_mapping = []; return

            self.chunk_to_original_doc_mapping = [doc.metadata["original_doc_index"] for doc in split_documents]
            logger.info(f"Generating embeddings for {len(split_documents)} document chunks...")
            doc_embeddings_list = self.embedding_model.embed_documents([doc.page_content for doc in split_documents])

            if not doc_embeddings_list or not doc_embeddings_list[0]:
                logger.error("Embedding process yielded no embeddings. Index will be dummy."); self.dimension = 768; self.index = faiss.IndexFlatL2(self.dimension); self.index.add(np.zeros((1, self.dimension), dtype='float32')); return

            self.dimension = len(doc_embeddings_list[0])
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(np.array(doc_embeddings_list, dtype='float32'))
            faiss.write_index(self.index, self.index_file)
            logger.info(f"Built FAISS index '{self.index_file}' ({self.index.ntotal} vectors). Mapping has {len(self.chunk_to_original_doc_mapping)} entries.")
        except Exception as e:
            logger.error(f"Error building FAISS index: {e}", exc_info=True)
            self.dimension = 768; self.index = faiss.IndexFlatL2(self.dimension); self.index.add(np.zeros((1, self.dimension), dtype='float32')); self.chunk_to_original_doc_mapping = []


    def _load_index(self, index_file: str):
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"FAISS index file {index_file} not found.")
        try:
            self.index = faiss.read_index(index_file)
            self.dimension = self.index.d
            logger.info(f"Loaded FAISS index from {index_file} ({self.index.ntotal} vectors, Dim: {self.dimension})")
        except Exception as e:
            logger.error(f"Error loading FAISS index {index_file}: {e}", exc_info=True); raise

    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        if not all([self.index, self.embedding_model, self.metadata, self.chunk_to_original_doc_mapping is not None]): # Check mapping explicitly
            logger.warning("Cannot retrieve: System components (index, model, metadata, or chunk_mapping) missing.")
            return []
        if self.index.ntotal == 0: logger.warning("Retrieval attempted but FAISS index is empty."); return []

        logger.info(f"Retrieving top {k} for query: {query[:50]}...")
        try: query_embedding = self.embedding_model.embed_query(query)
        except Exception as e: logger.error(f"Error embedding query: {e}", exc_info=True); return []
        
        actual_k = min(k, self.index.ntotal)
        if actual_k == 0: return []
        if actual_k < k: logger.warning(f"Requested k={k}, retrieving {actual_k}.")
            
        distances, faiss_indices = self.index.search(np.array([query_embedding], dtype='float32'), actual_k)
        
        original_doc_indices_found = set()
        results = []
        for i, chunk_idx_in_faiss in enumerate(faiss_indices[0]):
            if not (0 <= chunk_idx_in_faiss < len(self.chunk_to_original_doc_mapping)):
                logger.warning(f"FAISS index {chunk_idx_in_faiss} out of bounds for chunk mapping. Skipping."); continue
            
            original_doc_idx = self.chunk_to_original_doc_mapping[chunk_idx_in_faiss]
            if not (0 <= original_doc_idx < len(self.metadata)):
                logger.warning(f"Original doc index {original_doc_idx} (from chunk {chunk_idx_in_faiss}) out of bounds for metadata. Skipping."); continue

            if original_doc_idx not in original_doc_indices_found: # Add unique original documents
                doc_content_full = self.metadata[original_doc_idx].copy()
                if 'combined_text' in doc_content_full: del doc_content_full['combined_text']
                
                results.append({
                    "id": original_doc_idx, "content": doc_content_full,
                    "distance_to_chunk": float(distances[0][i]), # Distance of query to the specific chunk
                    "similarity_to_chunk": 1 / (1 + float(distances[0][i])) 
                })
                original_doc_indices_found.add(original_doc_idx)
                if len(results) >= actual_k: break # Stop if we have k unique documents
        
        logger.info(f"Retrieved {len(results)} unique original documents based on top chunks.")
        return results

    def format_retrieved_document_for_llm(self, doc: Dict) -> str:
        """Formats a single retrieved original document for the LLM context."""
        formatted_content = [f"--- Document ID: {doc['id']} (Similarity of best chunk: {doc['similarity_to_chunk']:.4f}) ---"]
        for key, value in doc["content"].items():
            value_str = str(value) if value is not None else ""
            if value_str.strip(): # Only include if there's actual content
                formatted_key = key.replace('_', ' ').title()
                formatted_content.append(f"{formatted_key}: {value_str}")
        return "\n".join(formatted_content)

    def analyze_patterns(self, retrieved_docs: List[Dict]) -> Dict[str, Any]:
        if not retrieved_docs: return {"count": 0, "patterns": {}, "date_range": None}
        analysis = {"count": len(retrieved_docs), "patterns": {}, "date_range": None}
        
        for col_name, info in self.column_info.items():
            if info.get('semantic_type') == 'category' or \
               (info.get('data_type') == 'text' and info.get('value_diversity', 1.0) < 0.5):
                value_counts = {}
                for doc in retrieved_docs:
                    value = doc["content"].get(col_name)
                    if value is not None and str(value).strip():
                        value_str = str(value)
                        value_counts[value_str] = value_counts.get(value_str, 0) + 1
                if value_counts: analysis["patterns"][col_name] = value_counts
            
            if info.get('data_type') == 'date':
                dates = []
                for doc in retrieved_docs:
                    date_val = doc["content"].get(col_name)
                    if date_val:
                        try: dt = pd.to_datetime(date_val, errors='coerce')
                        except: dt = None # Catch any parsing error
                        if pd.notna(dt): dates.append(dt)
                if dates:
                    min_date, max_date = min(dates), max(dates)
                    if analysis["date_range"] is None: # Take first date column found
                        analysis["date_range"] = {
                            "column": col_name, "min_date": min_date.strftime("%Y-%m-%d"),
                            "max_date": max_date.strftime("%Y-%m-%d"), "span_days": (max_date - min_date).days
                        }
        logger.info(f"Pattern analysis complete: {analysis}")
        return analysis

    def generate_response(self, query: str, k: int = 3) -> Dict[str, Any]:
        if not self.chain:
            logger.error("Cannot generate response: LLM chain not initialized.")
            return {"response": "System error: Unable to process request.", "retrieved_docs": [], "pattern_analysis": {"count": 0}}

        logger.info(f"Generating response for query: {query[:100]}..., k={k}")

        # 1. Get dataset overview
        dataset_overview_summary = self._generate_dataset_overview_summary()

        # 2. Retrieve relevant documents
        retrieved_docs = self.retrieve(query, k) # These are original docs with 'id' and chunk similarity

        # 3. Format retrieved documents for LLM context
        if retrieved_docs:
            retrieved_documents_llm_context = "\n\n===\n\n".join([self.format_retrieved_document_for_llm(doc) for doc in retrieved_docs])
        else:
            retrieved_documents_llm_context = "No specific documents were found to be highly relevant to this query."
            logger.warning("No relevant documents found for the query to pass to LLM.")
        
        # 4. Analyze patterns in the retrieved documents (even if empty, to get structure)
        pattern_analysis = self.analyze_patterns(retrieved_docs)
        
        # 5. Format pattern analysis for LLM context
        pattern_analysis_llm_summary_parts = ["Summary of Patterns Found in Retrieved Documents:"]
        if pattern_analysis["count"] > 0:
            pattern_analysis_llm_summary_parts.append(f"- Number of similar records found: {pattern_analysis['count']}")
            if pattern_analysis.get("date_range"):
                dr = pattern_analysis["date_range"]
                pattern_analysis_llm_summary_parts.append(f"- These records span from {dr['min_date']} to {dr['max_date']} (Column: {dr['column']}).")
            if pattern_analysis.get("patterns"):
                for col, val_counts in pattern_analysis["patterns"].items():
                    if len(val_counts) < 5 or any(c > 1 for c in val_counts.values()): # Show if few categories or some are frequent
                        common_vals_str = ", ".join([f"'{val}' ({count} times)" for val, count in sorted(val_counts.items(), key=lambda item: item[1], reverse=True)[:3]]) # Top 3
                        pattern_analysis_llm_summary_parts.append(f"- Common values for '{col.replace('_',' ').title()}': {common_vals_str}.")
        else:
            pattern_analysis_llm_summary_parts.append("- No specific patterns were identified in the retrieved documents (or no documents were retrieved).")
        pattern_analysis_llm_summary = "\n".join(pattern_analysis_llm_summary_parts)

        # 6. Invoke LLM
        logger.info("Invoking LLMChain with comprehensive context...")
        try:
            llm_input = {
                "query": query,
                "dataset_overview": dataset_overview_summary,
                "retrieved_documents_context": retrieved_documents_llm_context,
                "pattern_analysis_summary": pattern_analysis_llm_summary
            }
            # logger.debug(f"LLM Input: {llm_input}") # Can be very verbose
            llm_response_dict = self.chain.invoke(llm_input)
            response_text = llm_response_dict.get("text", "No response text generated by LLM.")
            logger.info("LLM response generated successfully.")
        except Exception as e:
            logger.error(f"Error during LLM chain invocation: {e}", exc_info=True)
            response_text = "I encountered an issue while trying to formulate a response."
        
        return {
            "response": response_text,
            "retrieved_docs": retrieved_docs, # Return structured docs for frontend
            "pattern_analysis": pattern_analysis # Return pattern analysis for frontend
        }

    def get_column_info(self) -> Dict[str, Dict[str, Any]]:
        return self.column_info

    def get_documents_by_ids(self, doc_ids: List[Union[int, str]]) -> List[Dict[Any, Any]]:
        if self.metadata is None: logger.warning("Metadata unavailable for fetching by IDs."); return []
        docs_to_return = []
        for doc_id in doc_ids:
            try:
                idx = int(doc_id)
                if 0 <= idx < len(self.metadata):
                    doc_content = self.metadata[idx].copy()
                    if 'combined_text' in doc_content: del doc_content['combined_text']
                    docs_to_return.append({"id": idx, "content": doc_content})
                else: logger.warning(f"Doc ID {idx} out of bounds for metadata.")
            except ValueError: logger.warning(f"Invalid doc ID format: {doc_id}.")
            except Exception as e: logger.error(f"Error retrieving doc for ID {doc_id}: {e}", exc_info=True)
        return docs_to_return

# --- FastAPI App Setup ---
app = FastAPI(
    title="Adaptive RAG System API",
    description="API for querying a RAG system with enhanced contextual understanding.",
    version="1.2.0" # Incremented version
)

origins = [
    "http://localhost", "http://localhost:3000", "http://localhost:8000",
    "http://localhost:8080", "http://127.0.0.1:5500", "http://127.0.0.1",
    "null", "https://horizonchaser12.github.io",
]
app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    query: str
    k: int = 5 # Default k increased slightly, can be overridden by client

class QueryResponse(BaseModel):
    response: str
    retrieved_docs: List[Dict[Any, Any]] 
    pattern_analysis: Dict[str, Any]

class DocumentDetailsRequest(BaseModel):
    doc_ids: List[Union[int, str]]

class DocumentDetailsResponse(BaseModel):
    documents: List[Dict[Any, Any]]

# --- Global RAG System Instance ---
rag_system_instance: Optional[AdaptiveRAGSystem] = None

@app.on_event("startup")
async def startup_event():
    global rag_system_instance
    logger.info("FastAPI startup: Initializing RAG system...")
    try:
        excel_file = "Defects.xlsx" 
        if not os.path.exists(excel_file):
            logger.warning(f"KB file '{excel_file}' not found. Creating dummy.")
            dummy_df = pd.DataFrame({
                'TicketID': [f'TICKET-{i:03}' for i in range(1, 6)], 
                'ProblemSummary': ['Login button broken', 'Slow dashboard loading', 'Payment fails with 503', 'Cannot export report', 'User profile image missing'],
                'ProductArea': ['Authentication', 'Performance', 'Billing', 'Reporting', 'User Management'],
                'ReportedDate': pd.to_datetime([f'2023-01-{d:02}' for d in range(10, 15)]),
                'Status': ['Closed', 'Investigating', 'Closed', 'Open', 'Closed'],
                'Resolution': ['Fixed in v1.2', np.nan, 'Gateway issue resolved', np.nan, 'Re-uploaded by user']
            })
            dummy_df.to_excel(excel_file, index=False)
            logger.info(f"Created dummy '{excel_file}'.")
        
        rag_system_instance = AdaptiveRAGSystem(excel_file_path=excel_file, llm_model='gemini-2.0-flash') # Explicitly set LLM
        logger.info("RAG system initialized successfully.")
    except Exception as e:
        logger.error(f"CRITICAL: Failed to initialize RAG system: {e}", exc_info=True)
        rag_system_instance = None

@app.post("/query", response_model=QueryResponse)
async def process_query_endpoint(request: QueryRequest):
    if rag_system_instance is None:
        logger.error("API /query: RAG system not available.")
        raise HTTPException(status_code=503, detail="RAG system unavailable.")

    logger.info(f"API /query: '{request.query}', k={request.k}")
    try:
        result = rag_system_instance.generate_response(request.query, request.k)
        serialized_result = make_serializable(result)
        return JSONResponse(content=serialized_result)
    except Exception as e:
        logger.error(f"API /query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/document_details", response_model=DocumentDetailsResponse)
async def get_document_details_endpoint(request: DocumentDetailsRequest):
    if rag_system_instance is None:
        logger.error("API /document_details: RAG system not available.")
        raise HTTPException(status_code=503, detail="RAG system unavailable.")
    
    logger.info(f"API /document_details: IDs: {request.doc_ids}")
    try:
        documents = rag_system_instance.get_documents_by_ids(request.doc_ids)
        serialized_documents = make_serializable(documents)
        return DocumentDetailsResponse(documents=serialized_documents)
    except Exception as e:
        logger.error(f"API /document_details error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching details: {str(e)}")

@app.get("/health", summary="Health Check")
async def health_check():
    status, rag_status = "ok", "initialized"
    rag_issues = []
    if rag_system_instance is None: status, rag_status = "error", "not_initialized_or_failed"
    else:
        if rag_system_instance.llm is None: rag_issues.append("LLM_not_loaded")
        if rag_system_instance.embedding_model is None: rag_issues.append("Embedding_model_not_loaded")
        if rag_system_instance.index is None : rag_issues.append("Index_not_loaded")
        elif rag_system_instance.index.ntotal == 0: rag_issues.append("Index_is_empty")
        if rag_system_instance.metadata is None: rag_issues.append("Metadata_not_loaded")
        
        if rag_issues: status, rag_status = "degraded", "; ".join(rag_issues)

    return {"api_status": status, "rag_system_status": rag_status, "timestamp": datetime.datetime.utcnow().isoformat()}

if __name__ == "__main__":
    logger.info("Starting FastAPI server with Uvicorn...")
    # Ensure your script is named 'Project.py' or adjust "Project:app" if your filename is different (e.g., "main:app")
    uvicorn.run("Project:app", host="0.0.0.0", port=8000, reload=True, log_level="info")

