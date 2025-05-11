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
    # Add other type conversions if necessary (e.g., custom objects)
    return obj

class AdaptiveRAGSystem:
    def __init__(self, excel_file_path: str, embedding_model: str = 'models/embedding-001',
                 llm_model: str = 'gemini-2.0-flash', temperature: float = 0.3, concise_prompt: bool = False,
                 index_file: str = "faiss_index.bin"):
        self.excel_file_path = excel_file_path
        self.concise_prompt = concise_prompt
        self.index_file = index_file
        self.chunk_to_original_doc_mapping: List[int] = [] # For mapping FAISS chunk indices to original metadata indices

        logger.info(f"Initializing models. Embedding: {embedding_model}, LLM: {llm_model}")
        try:
            self.embedding_model = GoogleGenerativeAIEmbeddings(model=embedding_model)
            self.llm = ChatGoogleGenerativeAI(model=llm_model, temperature=temperature, convert_system_message_to_human=True)
            logger.info("Google AI Models initialized successfully.")
        except Exception as e:
            logger.error(f"Fatal: Failed to initialize Google AI models: {e}. Ensure GOOGLE_API_KEY is correctly set and valid.")
            self.embedding_model = None
            self.llm = None

        self.data = None
        self.metadata: Optional[List[Dict[Any, Any]]] = None # Type hint for clarity
        self.index = None
        self.dimension = None
        self.column_info: Dict[str, Dict[str, Any]] = {}

        if self.embedding_model and self.llm:
            self._load_data()
            try:
                logger.info(f"Attempting to load FAISS index from {self.index_file}...")
                self._load_index(self.index_file)
                 # If index is loaded, we also need to potentially load the chunk_to_original_doc_mapping
                # This mapping should ideally be saved alongside the FAISS index.
                # For simplicity, we'll rebuild it if loading the index directly.
                # A more robust solution would save/load this mapping.
                # If _build_index was called by _load_data (if index file not found), mapping is already set.
                if not self.chunk_to_original_doc_mapping and self.data is not None and 'combined_text' in self.data.columns:
                    logger.info("Re-populating chunk_to_original_doc_mapping after loading index (if not already set).")
                    self._populate_chunk_mapping_from_data()

            except Exception as e:
                logger.warning(f"Could not load persisted index from {self.index_file} (Reason: {e}). Building a new index...")
                self._build_index() # This will also populate chunk_to_original_doc_mapping
        else:
            logger.error("Skipping data loading and index building due to model initialization failure.")

        if self.llm:
            if concise_prompt:
                self.response_template = """
                You are an AI Assistant. Given the following context:{context}Answer the following question:{question}Assistant:
                """
            else:
                self.response_template = """
                You are a helpful technical support assistant with expertise in software issues. Based on the user query and the relevant historical defect data provided,
                give a comprehensive yet concise response that addresses the user's issue in natural language.

                User Query: {query}

                Relevant Historical Defect Data (Chunks):
                {context}

                Provide a professional, conversational response that includes:
                1. A clear summary of the identified issue
                2. When this type of issue has occurred in the past (dates/frequency if available from context)
                3. The root causes that were identified for similar issues (from context)
                4. The solution or resolution that was most effective, explained in detail (from context)
                5. Any preventative measures that could avoid this issue in the future (from context)
                6. If multiple similar issues were found, explain any patterns or common factors (from context)

                Make your response conversational and easy to understand. Avoid technical jargon unless necessary and explain any complex terms.
                Format your response in clear paragraphs rather than as a list of facts.

                Response:
                """
            self.prompt = PromptTemplate(
                input_variables=["query", "context"],
                template=self.response_template
            )
            self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
            logger.info("LLMChain initialized.")
        else:
            self.chain = None
            logger.error("LLMChain could not be initialized because LLM is not available.")

    def _load_data(self):
        logger.info(f"Loading data from {self.excel_file_path}...")
        if not os.path.exists(self.excel_file_path):
            logger.error(f"Excel file not found at path: {self.excel_file_path}")
            self.data = pd.DataFrame({'Error': [f'File not found: {self.excel_file_path}']})
            self._prepare_data()
            self.metadata = self.data.to_dict(orient='records')
            logger.warning("Proceeding with dummy data due to missing Excel file.")
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

            self._prepare_data()
            self.metadata = self.data.to_dict(orient='records')
            logger.info(f"Loaded {len(self.data)} records from Excel file.")
        except Exception as e:
            logger.error(f"Error loading data: {e}", exc_info=True)
            logger.warning("Creating dummy data due to data loading error.")
            self.data = pd.DataFrame({'Error': [f'Error loading data: {str(e)}']})
            self._prepare_data()
            self.metadata = self.data.to_dict(orient='records')

    def _prepare_data(self):
        if self.data is None:
            logger.error("Cannot prepare data: self.data is None.")
            return
        self.data = self.data.fillna('')
        self.data = self.data.dropna(how='all').dropna(axis=1, how='all')
        self._analyze_columns()
        self.data['combined_text'] = self.data.apply(
            lambda row: ' '.join(f"{col}: {val}" for col, val in row.items() if val != '' and col != 'combined_text'),
            axis=1
        )
        logger.info("Data preparation complete. 'combined_text' field created.")

    def _analyze_columns(self):
        if self.data is None: return
        logger.info("Analyzing data columns...")
        self.column_info = {}
        for col in self.data.columns:
            if col == 'combined_text': continue
            col_data = self.data[col]
            data_type = 'text'
            if pd.api.types.is_numeric_dtype(col_data.infer_objects()): data_type = 'numeric'
            elif self._is_date_column(col_data): data_type = 'date'
            
            empty_count = (col_data == '').sum() + col_data.isna().sum()
            sparsity = empty_count / max(1, len(col_data))
            unique_values = col_data.nunique(dropna=False)
            value_diversity = unique_values / max(1, len(col_data))
            
            self.column_info[col] = {
                'data_type': data_type, 'sparsity': sparsity,
                'value_diversity': value_diversity, 'unique_values_count': unique_values
            }
            if data_type == 'text':
                avg_len = col_data.astype(str).str.len().mean() if not col_data.empty else 0
                if value_diversity > 0.8: self.column_info[col]['semantic_type'] = 'description' if avg_len > 50 else 'identifier'
                elif value_diversity < 0.2 and unique_values < 20:
                     self.column_info[col]['semantic_type'] = 'category'
                     self.column_info[col]['categories'] = col_data.unique().tolist() if unique_values < 20 else 'Too many to list'
                else: self.column_info[col]['semantic_type'] = 'general_text'
            elif data_type == 'date': self.column_info[col]['semantic_type'] = 'date'
        logger.info(f"Column analysis complete: {self.column_info}")

    def _is_date_column(self, series: pd.Series) -> bool:
        if series.empty: return False
        try:
            sample = series.dropna().sample(min(len(series.dropna()), 100)) if not series.dropna().empty else pd.Series(dtype=object)
            if sample.empty and series.notna().any(): sample = series.loc[series.notna()].head(100)
            if sample.empty: return False
            converted_sample = pd.to_datetime(sample, errors='coerce')
            success_rate = converted_sample.notna().sum() / max(1, len(sample))
            return success_rate > 0.8
        except Exception: return False

    def _populate_chunk_mapping_from_data(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Helper to (re)populate chunk_to_original_doc_mapping from self.data.
           This is needed if the mapping isn't saved/loaded with the FAISS index.
        """
        if self.data is None or 'combined_text' not in self.data.columns:
            logger.warning("Cannot populate chunk mapping: Data or 'combined_text' column is not available.")
            self.chunk_to_original_doc_mapping = []
            return

        documents_content = self.data['combined_text'].tolist()
        if not documents_content or all(s.strip() == "" for s in documents_content):
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
            self.chunk_to_original_doc_mapping = []
            return
            
        split_documents = text_splitter.split_documents(langchain_documents)
        
        self.chunk_to_original_doc_mapping = []
        if split_documents:
            for chunk_doc in split_documents:
                self.chunk_to_original_doc_mapping.append(chunk_doc.metadata["original_doc_index"])
        logger.info(f"Populated chunk_to_original_doc_mapping with {len(self.chunk_to_original_doc_mapping)} entries.")


    def _build_index(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        if self.data is None or 'combined_text' not in self.data.columns or self.embedding_model is None:
            logger.error("Cannot build FAISS index: Data, 'combined_text', or embedding model not available.")
            self.dimension = 768
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(np.zeros((1, self.dimension), dtype='float32'))
            self.chunk_to_original_doc_mapping = []
            return

        logger.info(f"Building FAISS index. Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
        try:
            documents_content = self.data['combined_text'].tolist()
            if not documents_content or all(s.strip() == "" for s in documents_content):
                logger.warning("No valid text content for embedding. Index will be dummy.")
                # ... (dummy index creation as before)
                self.dimension = getattr(self.embedding_model.client, 'get_embedding_dimensionality', lambda m: 768)(self.embedding_model.model)
                self.index = faiss.IndexFlatL2(self.dimension)
                self.index.add(np.zeros((1, self.dimension), dtype='float32'))
                self.chunk_to_original_doc_mapping = []
                return

            from langchain.text_splitter import RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, separators=["\n\n", "\n", " ", ""], chunk_overlap=chunk_overlap
            )
            
            langchain_documents = []
            for i, doc_content_str in enumerate(documents_content): # Iterate with index
                if doc_content_str.strip():
                    # Store original document index in the metadata of the Document object
                    langchain_documents.append(Document(page_content=doc_content_str, metadata={"original_doc_index": i}))
            
            if not langchain_documents:
                logger.warning("No non-empty documents for text splitting. Index will be dummy.")
                # ... (dummy index creation)
                self.dimension = 768 
                self.index = faiss.IndexFlatL2(self.dimension)
                self.index.add(np.zeros((1, self.dimension), dtype='float32'))
                self.chunk_to_original_doc_mapping = []
                return

            split_documents = text_splitter.split_documents(langchain_documents) # This preserves metadata by default
            
            if not split_documents:
                logger.warning("Text splitting resulted in zero documents. Index will be dummy.")
                # ... (dummy index creation)
                self.dimension = 768
                self.index = faiss.IndexFlatL2(self.dimension)
                self.index.add(np.zeros((1, self.dimension), dtype='float32'))
                self.chunk_to_original_doc_mapping = []
                return

            # Populate self.chunk_to_original_doc_mapping
            self.chunk_to_original_doc_mapping = [doc.metadata["original_doc_index"] for doc in split_documents]

            logger.info(f"Generating embeddings for {len(split_documents)} document chunks...")
            doc_embeddings_list = self.embedding_model.embed_documents([doc.page_content for doc in split_documents])

            if not doc_embeddings_list or not doc_embeddings_list[0]:
                logger.error("Embedding process yielded no embeddings.")
                # ... (dummy index creation)
                self.dimension = 768
                self.index = faiss.IndexFlatL2(self.dimension)
                self.index.add(np.zeros((1, self.dimension), dtype='float32'))
                return

            self.dimension = len(doc_embeddings_list[0])
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(np.array(doc_embeddings_list, dtype='float32'))

            faiss.write_index(self.index, self.index_file)
            # Ideally, save self.chunk_to_original_doc_mapping here too, e.g., to a JSON file.
            # For now, it's rebuilt if needed upon loading.
            logger.info(f"Built and saved FAISS index to '{self.index_file}' with {self.index.ntotal} vectors. Mapping has {len(self.chunk_to_original_doc_mapping)} entries.")

        except Exception as e:
            logger.error(f"Error building FAISS search index: {e}", exc_info=True)
            self.dimension = 768
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(np.zeros((1, self.dimension), dtype='float32'))
            self.chunk_to_original_doc_mapping = []


    def _load_index(self, index_file: str):
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"FAISS index file {index_file} not found.")
        try:
            self.index = faiss.read_index(index_file)
            self.dimension = self.index.d
            logger.info(f"Successfully loaded FAISS index from {index_file}. N_vectors: {self.index.ntotal}, Dimension: {self.dimension}")
            # NOTE: self.chunk_to_original_doc_mapping should also be loaded here if it was saved.
            # If not saved, it will be repopulated by _populate_chunk_mapping_from_data in __init__.
        except Exception as e:
            logger.error(f"Error loading FAISS index from {index_file}: {e}", exc_info=True)
            raise

    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve the top k most relevant document CHUNKS for the query.
        Returns a list of dictionaries, each containing the chunk content, 
        its original document ID (metadata index), and similarity score.
        """
        if self.index is None or self.embedding_model is None or self.metadata is None or not self.chunk_to_original_doc_mapping:
            logger.warning("Cannot retrieve: Index, model, metadata, or chunk mapping not available.")
            return []
        if self.index.ntotal == 0:
             logger.warning("Retrieval attempted but FAISS index is empty.")
             return []

        logger.info(f"Retrieving top {k} document chunks for query: {query[:50]}...")
        try:
            query_embedding = self.embedding_model.embed_query(query)
        except Exception as e:
            logger.error(f"Error embedding query '{query[:50]}...': {e}", exc_info=True)
            return []
        
        actual_k = min(k, self.index.ntotal)
        if actual_k == 0: return []
        if actual_k < k: logger.warning(f"Requested k={k}, retrieving {actual_k}.")
            
        distances, faiss_indices = self.index.search(np.array([query_embedding], dtype='float32'), actual_k)
        
        retrieved_chunks_info = []
        # We need the actual chunk content for the LLM context
        # This requires access to the `split_documents` list from `_build_index` or re-splitting.
        # For simplicity, and to avoid storing all split_documents in memory, we'll focus on returning
        # original document references. The format_retrieved_document will use the full doc.
        # This is a compromise. A more advanced system would pass chunk content to LLM.

        # Re-fetch split documents to get chunk content (can be inefficient, better to store them or map IDs carefully)
        # This part is tricky without storing all split docs.
        # For now, we'll use the original document content for formatting, acknowledging this simplification.
        # The "context" for the LLM will be based on the full original documents corresponding to the best chunks.

        # Get original documents based on chunk mapping
        original_doc_indices_found = set()
        results = []

        for i, chunk_idx_in_faiss in enumerate(faiss_indices[0]):
            if not (0 <= chunk_idx_in_faiss < len(self.chunk_to_original_doc_mapping)):
                logger.warning(f"FAISS index {chunk_idx_in_faiss} out of bounds for chunk mapping. Skipping.")
                continue
            
            original_doc_idx = self.chunk_to_original_doc_mapping[chunk_idx_in_faiss]

            if 0 <= original_doc_idx < len(self.metadata):
                # If we only want to add each unique original document once to results
                if original_doc_idx in original_doc_indices_found and len(results) >= actual_k : # Avoid processing same doc if already have enough unique docs
                    continue
                
                doc_content_full = self.metadata[original_doc_idx].copy()
                
                # The ID for the API response should be the original document's ID (its index in self.metadata)
                doc_id_for_api = original_doc_idx 

                if 'combined_text' in doc_content_full:
                    del doc_content_full['combined_text']
                
                result = {
                    "id": doc_id_for_api, # ID of the original document
                    "content": doc_content_full, # Full content of the original document
                    # "chunk_content": split_documents[chunk_idx_in_faiss].page_content, # IDEALLY, but requires access to split_documents
                    "distance": float(distances[0][i]),
                    "similarity": 1 / (1 + float(distances[0][i])) # Example similarity
                }
                results.append(result)
                original_doc_indices_found.add(original_doc_idx)

                if len(results) >= actual_k: # Stop if we have k unique documents
                    break 
            else:
                logger.warning(f"Original document index {original_doc_idx} (from chunk {chunk_idx_in_faiss}) is out of bounds for metadata. Skipping.")
        
        logger.info(f"Retrieved {len(results)} unique original documents based on top chunks.")
        return results


    def format_retrieved_document(self, doc: Dict) -> str:
        """
        Format a retrieved document (which is an original document's content)
        for better readability in the LLM context.
        """
        formatted_content = []
        # doc["content"] here is the full original document content
        for key, value in doc["content"].items():
            value_str = str(value) if value is not None else ""
            if value_str.strip():
                formatted_key = key.replace('_', ' ').title()
                formatted_content.append(f"{formatted_key}: {value_str}")
        
        formatted_content.append(f"RelevanceScoreToQueryChunk: {doc['similarity']:.4f}")
        return f"--- Document ID: {doc['id']} ---\n" + "\n".join(formatted_content)


    def analyze_patterns(self, retrieved_docs: List[Dict]) -> Dict[str, Any]:
        if not retrieved_docs:
            return {"count": 0, "patterns": {}, "date_range": None}
        analysis = {"count": len(retrieved_docs), "patterns": {}, "date_range": None}
        
        for col_name, info in self.column_info.items():
            if info.get('semantic_type') == 'category' or \
               (info.get('data_type') == 'text' and info.get('value_diversity', 1.0) < 0.5):
                value_counts = {}
                for doc in retrieved_docs: # doc["content"] is the full original document
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
                        try:
                            dt = pd.to_datetime(date_val, errors='coerce')
                            if pd.notna(dt): dates.append(dt)
                        except Exception: pass
                if dates:
                    min_date, max_date = min(dates), max(dates)
                    if analysis["date_range"] is None:
                        analysis["date_range"] = {
                            "column": col_name, "min_date": min_date.strftime("%Y-%m-%d"),
                            "max_date": max_date.strftime("%Y-%m-%d"), "span_days": (max_date - min_date).days
                        }
        logger.info(f"Pattern analysis complete: {analysis}")
        return analysis

    def generate_response(self, query: str, k: int = 3) -> Dict[str, Any]:
        if not self.chain:
            logger.error("Cannot generate response: LLM chain not initialized.")
            return {
                "response": "System error: Unable to process request.",
                "retrieved_docs": [], "pattern_analysis": {"count": 0}
            }

        logger.info(f"Starting generate_response for query: {query[:100]}..., k={k}")
        # retrieved_docs now contains full original documents with an 'id'
        retrieved_docs = self.retrieve(query, k)

        if not retrieved_docs:
            logger.warning("No relevant documents found for the query.")
            return {
                "response": "I couldn't find specific information for your query.",
                "retrieved_docs": [], "pattern_analysis": {"count": 0}
            }

        # doc_contexts will be formatted strings of the *full original documents*
        doc_contexts = [self.format_retrieved_document(doc) for doc in retrieved_docs]
        pattern_analysis = self.analyze_patterns(retrieved_docs)
        
        context_parts = ["Retrieved Information (Original Documents corresponding to best matching text portions):"]
        context_parts.extend(doc_contexts)

        if pattern_analysis["count"] > 0:
            context_parts.append("\nSummary of Patterns Found in these Documents:")
            context_parts.append(f"- Number of similar records found: {pattern_analysis['count']}")
            if pattern_analysis.get("date_range"):
                dr = pattern_analysis["date_range"]
                context_parts.append(f"- These records span from {dr['min_date']} to {dr['max_date']} (Column: {dr['column']}).")
            if pattern_analysis.get("patterns"):
                for col, val_counts in pattern_analysis["patterns"].items():
                    if len(val_counts) < 5 or any(c > 1 for c in val_counts.values()):
                        common_vals_str = ", ".join([f"'{val}' ({count} times)" for val, count in sorted(val_counts.items(), key=lambda item: item[1], reverse=True)[:3]])
                        context_parts.append(f"- Common values for '{col.replace('_',' ').title()}': {common_vals_str}.")
        
        final_context = "\n\n===\n\n".join(context_parts)
        logger.debug(f"Context for LLM (first 500 chars): {final_context[:500]}...")

        logger.info("Invoking LLMChain to generate response...")
        try:
            llm_response_dict = self.chain.invoke({"query": query, "context": final_context})
            response_text = llm_response_dict.get("text", "No response text generated.")
            logger.info("LLM response generated.")
        except Exception as e:
            logger.error(f"Error during LLM chain invocation: {e}", exc_info=True)
            response_text = "Error formulating response from retrieved information."
        
        return {
            "response": response_text,
            "retrieved_docs": retrieved_docs, # These are full docs with 'id'
            "pattern_analysis": pattern_analysis
        }

    def get_column_info(self) -> Dict[str, Dict[str, Any]]:
        return self.column_info

    def get_documents_by_ids(self, doc_ids: List[Union[int, str]]) -> List[Dict[Any, Any]]:
        """Retrieve full documents from metadata given their IDs (indices)."""
        if self.metadata is None:
            logger.warning("Metadata is not available to fetch documents by IDs.")
            return []
        
        docs_to_return = []
        for doc_id in doc_ids:
            try:
                idx = int(doc_id)
                if 0 <= idx < len(self.metadata):
                    doc_content = self.metadata[idx].copy()
                    if 'combined_text' in doc_content: # Remove helper field
                        del doc_content['combined_text']
                    docs_to_return.append({"id": idx, "content": doc_content})
                else:
                    logger.warning(f"Requested document ID {idx} is out of bounds for metadata.")
            except ValueError:
                logger.warning(f"Invalid document ID format: {doc_id}. Must be integer-like.")
            except Exception as e:
                logger.error(f"Error retrieving document for ID {doc_id}: {e}", exc_info=True)
        return docs_to_return

# --- FastAPI App Setup ---
app = FastAPI(
    title="Adaptive RAG System API",
    description="API for querying a RAG system backed by an Excel knowledge base.",
    version="1.1.0" # Incremented version
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
    k: int = 3

class QueryResponse(BaseModel):
    response: str
    retrieved_docs: List[Dict[Any, Any]] # Each dict should have 'id' and 'content'
    pattern_analysis: Dict[str, Any]

class DocumentDetailsRequest(BaseModel):
    doc_ids: List[Union[int, str]] # Document IDs (expected to be indices of metadata)

class DocumentDetailsResponse(BaseModel):
    documents: List[Dict[Any, Any]] # List of document dicts, each with 'id' and 'content'

# --- Global RAG System Instance ---
rag_system_instance: Optional[AdaptiveRAGSystem] = None

@app.on_event("startup")
async def startup_event():
    global rag_system_instance
    logger.info("FastAPI application startup: Initializing RAG system...")
    try:
        excel_file = "Defects.xlsx" # Ensure this file exists or is created
        if not os.path.exists(excel_file):
            logger.warning(f"KB file '{excel_file}' not found. Creating dummy file.")
            dummy_df = pd.DataFrame({
                'ID_Col': [1, 2, 3], 'Description': ['Login fail', 'Slow load', 'Payment error'],
                'Solution': ['Cache clear', 'Server scale', 'Gateway fix'],
                'Date': [datetime.date(2023,1,10), datetime.date(2023,2,15), datetime.date(2023,3,20)]
            })
            dummy_df.to_excel(excel_file, index=False)
            logger.info(f"Created dummy '{excel_file}'.")
        
        rag_system_instance = AdaptiveRAGSystem(excel_file_path=excel_file)
        logger.info("RAG system initialized successfully.")
    except Exception as e:
        logger.error(f"CRITICAL: Failed to initialize RAG system during startup: {e}", exc_info=True)
        rag_system_instance = None

@app.post("/query", response_model=QueryResponse)
async def process_query_endpoint(request: QueryRequest):
    if rag_system_instance is None:
        logger.error("API call to /query failed: RAG system not available.")
        raise HTTPException(status_code=503, detail="RAG system unavailable.")

    logger.info(f"Received API query: '{request.query}', k={request.k}")
    try:
        result = rag_system_instance.generate_response(request.query, request.k)
        # Use the global make_serializable utility
        serialized_result = make_serializable(result)
        return JSONResponse(content=serialized_result)
    except Exception as e:
        logger.error(f"Error processing query via API: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/document_details", response_model=DocumentDetailsResponse)
async def get_document_details_endpoint(request: DocumentDetailsRequest):
    """
    New endpoint to fetch details for specific documents by their IDs.
    The IDs are expected to be the indices from the original metadata list.
    """
    if rag_system_instance is None:
        logger.error("API call to /document_details failed: RAG system not available.")
        raise HTTPException(status_code=503, detail="RAG system unavailable.")
    
    logger.info(f"Received request for document details. IDs: {request.doc_ids}")
    try:
        documents = rag_system_instance.get_documents_by_ids(request.doc_ids)
        # Use the global make_serializable utility
        serialized_documents = make_serializable(documents)
        return DocumentDetailsResponse(documents=serialized_documents)
    except Exception as e:
        logger.error(f"Error fetching document details via API: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching document details: {str(e)}")

@app.get("/health", summary="Health Check")
async def health_check():
    status, rag_status = "ok", "initialized"
    if rag_system_instance is None: status, rag_status = "error", "not_initialized_or_failed"
    elif rag_system_instance.llm is None or rag_system_instance.embedding_model is None: status, rag_status = "degraded", "models_not_loaded"
    elif rag_system_instance.index is None or rag_system_instance.index.ntotal == 0: status, rag_status = "degraded", "index_not_loaded_or_empty"
    return {"api_status": status, "rag_system_status": rag_status, "timestamp": datetime.datetime.utcnow().isoformat()}

if __name__ == "__main__":
    logger.info("Starting FastAPI server using Uvicorn...")
    # Ensure your script is named 'Project.py' or adjust "Project:app" accordingly.
    # If your file is main.py, use "main:app".
    # For the provided code, if saved as e.g., `main_api.py`, use `uvicorn.run("main_api:app", ...)`
    uvicorn.run("Project:app", host="0.0.0.0", port=8000, reload=True, log_level="info")

