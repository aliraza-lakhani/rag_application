import os
import tempfile
import streamlit as st
from llama_index.core import VectorStoreIndex, Settings, QueryBundle
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor, KeywordNodePostprocessor
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.readers.file import PDFReader
import chromadb
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from langfuse.llama_index import LlamaIndexCallbackHandler
from fastapi import FastAPI, Request
 
app = FastAPI()

@app.get("/langfuse/test")
def test_api():
    return {"message": f"Callback received"}

langfuse_callback_handler = LlamaIndexCallbackHandler(
    public_key="pk-lf-6316b34b-8c4d-4d0d-994d-6bb6bc47b2c5",
    secret_key="sk-lf-bd273488-c590-4e4d-b514-d01ce590c774",
    host="https://cloud.langfuse.com"
)
Settings.callback_manager = CallbackManager([langfuse_callback_handler])


# Set page configuration
st.set_page_config(page_title="RAG PDF Assistant", layout="wide")

# Sidebar for API Key and configuration
with st.sidebar:
    st.title("Configuration")
    openai_api_key = st.text_input("OpenAI API Key", key="openai_api_key", type="password")

    st.subheader("Advanced Settings")
    chunk_size = st.slider("Chunk Size", min_value=256, max_value=2048, value=512, step=128,
                         help="Size of text chunks. Smaller chunks may improve precision.")
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=200, value=50, step=10,
                            help="Overlap between chunks to maintain context.")
    similarity_top_k = st.slider("Retrieved Documents", min_value=2, max_value=10, value=4, step=1,
                              help="Number of documents to retrieve per query.")
    
    similarity_cutoff = st.slider("Similarity Cutoff", min_value=0.0, max_value=1.0, value=0.1, step=0.05,
                                help="Minimum relevance score (0-1) to include a result.")
    
    st.divider()
    st.markdown("## About")
    st.markdown("This app uses LlamaIndex to process PDFs, store them in ChromaDB, and answer questions based on their content.")

# Main page
st.title("📚 Enhanced RAG PDF Assistant")

# Initialize session state
if "index" not in st.session_state:
    st.session_state.index = None
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False
if "document_files" not in st.session_state:
    st.session_state.document_files = []


# Function to process PDFs and create index
def process_documents(files):
    if not openai_api_key:
        st.error("Please enter your OpenAI API key in the sidebar!")
        return False
    
    # Set OpenAI API key
    os.environ["OPENAI_API_KEY"] = openai_api_key
    
    with st.spinner("Processing documents..."):
        try:
            # Initialize PDF reader
            reader = PDFReader()
            documents = []
            
            # Read all uploaded documents
            for file in files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(file.getvalue())
                    temp_path = temp_file.name
                
                # Add metadata to documents
                docs = reader.load_data(file=temp_path)
                for doc in docs:
                    doc.metadata.update({
                        "file_name": file.name,
                        "file_size": file.size,
                        "content_type": file.type
                    })
                documents.extend(docs)
                os.unlink(temp_path)  # Clean up temp file
            
            # Initialize ChromaDB
            # chroma_client = chromadb.Client()
            chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma3")
            chroma_client.delete_collection("internal_rag_documents_2")
            chroma_collection = chroma_client.create_collection("internal_rag_documents_2")
            
            # Setup vector store
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Setup embedding and LLM models
            embed_model = OpenAIEmbedding(model="text-embedding-3-large", api_key=openai_api_key)
            llm = OpenAI(model="gpt-4o", api_key=openai_api_key, temperature=0.3)
            
            # Setup node parser with custom chunk size and overlap
            node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            
            # Configure settings
            Settings.llm = llm
            Settings.embed_model = embed_model
            Settings.node_parser = node_parser
            Settings.storage_context = storage_context
            
            # Create index
            index = VectorStoreIndex.from_documents(
                documents, 
                storage_context=storage_context
            )
            
            st.session_state.index = index
            return True
        
        except Exception as e:
            st.error(f"Error processing documents: {e}")
            import traceback
            st.error(traceback.format_exc())
            return False

# Custom query engine creation with advanced retrieval settings
def create_advanced_query_engine(index):
    # Configure retriever with similarity_top_k parameter
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=similarity_top_k,
    )
    
    # Add postprocessors for better results
    postprocessors = [
        # Filter out less relevant nodes by cosine similarity threshold
        SimilarityPostprocessor(similarity_cutoff=similarity_cutoff),
        # Add additional keyword-based filtering 
        KeywordNodePostprocessor(required_keywords=[])
    ]
    
    # Create query engine with the custom retriever and postprocessors
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=postprocessors
    )
    
    return query_engine

# Document upload section
st.header("Upload Documents")
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    if st.button("Process Documents"):
        st.session_state.document_files = uploaded_files
        success = process_documents(uploaded_files)
        if success:
            st.session_state.documents_processed = True
            st.success(f"Successfully processed {len(uploaded_files)} document(s)!")
            st.balloons()

# Query section (only show if documents have been processed)
if st.session_state.documents_processed and st.session_state.index:
    st.header("Ask Questions About Your Documents")
    
    # Query expansion option
    enable_query_expansion = st.checkbox("Enable Query Expansion", value=True, 
                                        help="Expands your query to improve results")
    enable_reranking = st.checkbox("Enable Response Reranking", value=True,
                                  help="Reranks results by relevance")
    
    query = st.text_input("Enter your question:")
    
    if query:
        with st.spinner("Generating answer..."):
            try:
                # Create advanced query engine
                st.write(st.session_state)
                query_engine = create_advanced_query_engine(st.session_state.index)
                
                # Apply query transformation if enabled
                if enable_query_expansion:
                    # Use LLM to expand the query
                    llm_expanded_query = Settings.llm.complete(
                        f"Original query: '{query}'\nExpand this query with additional relevant keywords to improve search results. Output just the expanded query:"
                    )
                    transformed_query = llm_expanded_query.text
                    
                    with st.expander("View Query Transformation"):
                        st.markdown(f"**Original Query:** {query}")
                        st.markdown(f"**Expanded Query:** {transformed_query}")
                    
                    bundle = QueryBundle(query_str=query, custom_embedding_strs=[transformed_query])
                    response = query_engine.query(bundle)
                else:
                    response = query_engine.query(query)
                st.write(response)
                # Apply reranking based on direct question relevance if enabled
                if enable_reranking and hasattr(response, 'source_nodes') and len(response.source_nodes) > 1:
                    # Rerank based on relevance to query
                    for node in response.source_nodes:
                        relevance_score = Settings.llm.complete(
                            f"On a scale of 0-10, how relevant is this text to answering the question: '{query}'?\n\nText: {node.text}\n\nScore: "
                        ).text.strip()
                        try:
                            node.score = float(relevance_score) / 10.0 if relevance_score.isdigit() else node.score
                        except:
                            pass
                    
                    # Sort by new scores
                    response.source_nodes = sorted(response.source_nodes, key=lambda x: getattr(x, 'score', 0), reverse=True)
                
                # System prompt for better formatting
                final_prompt = f"""
                Given the question: {query}
                
                Based on the retrieved documents, provide a comprehensive and accurate answer.
                
                - Format the answer with proper Markdown for readability
                - Include all relevant information from the sources
                - If information is not available in the documents, say so
                - Do not make up information not present in the sources
                - Don't answer questions that are not related to the documents
                """
                
                final_response = Settings.llm.complete(final_prompt + "\n\nHere are the documents:\n" + 
                                              "\n---\n".join([n.text for n in response.source_nodes]) +
                                              "\n\nAnswer:")
                
                st.subheader("Answer")
                st.markdown(final_response.text)
                
                with st.expander("Source Documents"):
                    for i, source_node in enumerate(response.source_nodes):
                        st.markdown(f"**Source {i+1}** (Relevance Score: {getattr(source_node, 'score', 'N/A')})")
                        st.markdown(f"**File:** {source_node.metadata.get('file_name', 'Unknown')}")
                        st.markdown(f"**Text:**")
                        st.markdown(source_node.text)
                        st.divider()
            
            except Exception as e:
                st.error(f"Error generating response: {e}")
                import traceback
                st.error(traceback.format_exc())