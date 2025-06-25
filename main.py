import os
import json
import tempfile
import streamlit as st
import qdrant_client
from pathlib import Path
from datetime import datetime
from llama_index.core.schema import Document
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.mistralai import MistralAI
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.prompts import PromptTemplate
from llama_index.core.settings import Settings
import cv2
import numpy as np
import requests
import easyocr
import traceback

# ================================
# PAGE CONFIG & STYLING
# ================================

st.set_page_config(
    page_title="🏥 Medical Report Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #4CAF50 0%, #2196F3 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #4CAF50;
    }
    .chat-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #4CAF50 0%, #2196F3 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: 600;
    }
    .upload-section {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ================================
# MEDICAL OCR CLASS (Enhanced from your code)
# ================================

class MedicalReportOCR:
    def __init__(self, ollama_url="http://localhost:11434", model_name="llama3.2:1b"):
        self.ollama_url = ollama_url
        self.model_name = model_name
        
        # Initialize EasyOCR reader
        try:
            self.ocr_reader = easyocr.Reader(['en'])
        except Exception as e:
            st.error(f"Failed to initialize EasyOCR: {e}")
            raise
    
    def preprocess_image(self, image_path):
        """Preprocess image for better OCR results"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(sharpened)
        
        return enhanced
    
    def extract_text_easyocr(self, image_path):
        """Extract text using EasyOCR"""
        processed_img = self.preprocess_image(image_path)
        results = self.ocr_reader.readtext(processed_img)
        
        extracted_texts = []
        full_text_parts = []
        
        for (bbox, text, confidence) in results:
            if confidence > 0.3:
                cleaned_text = text.strip()
                if cleaned_text:
                    extracted_texts.append({
                        'text': cleaned_text,
                        'confidence': round(confidence * 100, 2),
                        'bbox': bbox
                    })
                    full_text_parts.append(cleaned_text)
        
        full_text = ' '.join(full_text_parts)
        return full_text, extracted_texts
    
    def generate_json_with_ollama(self, extracted_text, image_filename):
        """Use Ollama to convert extracted text to structured JSON"""
        max_text_length = 6000
        if len(extracted_text) > max_text_length:
            extracted_text = extracted_text[:max_text_length] + "\n[TEXT TRUNCATED DUE TO LENGTH]"
        
        prompt = f"""You are an expert medical report parser. Convert this OCR text into structured JSON:

{extracted_text}

Create JSON with these sections:
1. hospital_info: {{"hospital_name": "", "address": "", "phone": "", "email": ""}}
2. patient_info: {{"name": "", "age": "", "gender": "", "patient_id": ""}}
3. doctor_info: {{"referring_doctor": "", "pathologist": ""}}
4. report_info: {{"report_type": "", "collection_date": "", "report_date": "", "sample_type": ""}}
5. test_results: [{{"test_name": "", "result_value": "", "reference_range": "", "unit": "", "status": ""}}]
6. additional_info: {{"notes": "", "interpretation": ""}}

Return ONLY valid JSON, no explanations."""

        try:
            request_data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "top_p": 0.9, "num_predict": 1024}
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=request_data,
                timeout=180
            )
            
            if response.status_code != 200:
                return {'success': False, 'error': f'Ollama API error: HTTP {response.status_code}'}
            
            result = response.json()
            json_text = result.get('response', '').strip()
            
            # Clean up JSON response
            if json_text.startswith('```json'):
                json_text = json_text[7:]
            elif json_text.startswith('```'):
                json_text = json_text[3:]
            if json_text.endswith('```'):
                json_text = json_text[:-3]
            json_text = json_text.strip()
            
            # Parse JSON
            parsed_json = json.loads(json_text)
            
            # Add metadata
            parsed_json['_metadata'] = {
                'source_image': image_filename,
                'extraction_method': 'easyocr_ollama',
                'processing_timestamp': datetime.now().isoformat(),
                'model_used': self.model_name
            }
            
            return {'success': True, 'json_data': parsed_json, 'raw_response': json_text}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def process_image(self, image_path):
        """Process a single medical report image"""
        image_filename = os.path.basename(image_path)
        
        try:
            extracted_text, extraction_details = self.extract_text_easyocr(image_path)
            
            if not extracted_text.strip():
                return {'success': False, 'error': 'No text extracted from image'}
            
            ollama_result = self.generate_json_with_ollama(extracted_text, image_filename)
            
            if ollama_result['success']:
                return {
                    'success': True,
                    'image_filename': image_filename,
                    'extracted_text': extracted_text,
                    'structured_json': ollama_result['json_data']
                }
            else:
                return {
                    'success': False,
                    'error': ollama_result['error'],
                    'extracted_text': extracted_text
                }
                
        except Exception as e:
            return {'success': False, 'error': str(e)}

# ================================
# INITIALIZATION FUNCTIONS
# ================================

@st.cache_resource
def init_qdrant():
    """Initialize Qdrant Cloud client"""
    try:
        qdrant_url = st.secrets.get("QDRANT_URL") or os.getenv("QDRANT_URL")
        qdrant_api_key = st.secrets.get("QDRANT_API_KEY") or os.getenv("QDRANT_API_KEY")
        
        if not qdrant_url or not qdrant_api_key:
            st.error("❌ Qdrant Cloud credentials not found!")
            st.stop()
        
        client = qdrant_client.QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        client.get_collections()
        return client
        
    except Exception as e:
        st.error(f"❌ Cannot connect to Qdrant Cloud: {str(e)}")
        st.stop()

@st.cache_resource
def init_embedding():
    """Initialize embedding model"""
    return HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)

@st.cache_resource
def init_mistral():
    """Initialize Mistral AI model"""
    api_key = st.secrets.get("MISTRAL_API_KEY") or os.getenv("MISTRAL_API_KEY")
    if not api_key:
        st.error("❌ MISTRAL_API_KEY not found!")
        st.stop()
    return MistralAI(model="mistral-large-latest", temperature=0.7, max_tokens=512, api_key=api_key)

@st.cache_resource
def init_ocr_processor():
    """Initialize OCR processor"""
    return MedicalReportOCR()

# ================================
# DATABASE FUNCTIONS
# ================================

def create_documents_from_json_data(json_reports):
    """Create LlamaIndex documents from JSON data"""
    documents = []
    
    for report in json_reports:
        if not report.get('success', False):
            continue
            
        json_data = report['structured_json']
        
        # Create comprehensive text representation
        text_content = f"""
Medical Report Analysis:

Hospital Information:
- Hospital: {json_data.get('hospital_info', {}).get('hospital_name', 'N/A')}
- Address: {json_data.get('hospital_info', {}).get('address', 'N/A')}
- Phone: {json_data.get('hospital_info', {}).get('phone', 'N/A')}

Patient Information:
- Name: {json_data.get('patient_info', {}).get('name', 'N/A')}
- Age: {json_data.get('patient_info', {}).get('age', 'N/A')}
- Gender: {json_data.get('patient_info', {}).get('gender', 'N/A')}
- Patient ID: {json_data.get('patient_info', {}).get('patient_id', 'N/A')}

Doctor Information:
- Referring Doctor: {json_data.get('doctor_info', {}).get('referring_doctor', 'N/A')}
- Pathologist: {json_data.get('doctor_info', {}).get('pathologist', 'N/A')}

Report Information:
- Report Type: {json_data.get('report_info', {}).get('report_type', 'N/A')}
- Collection Date: {json_data.get('report_info', {}).get('collection_date', 'N/A')}
- Report Date: {json_data.get('report_info', {}).get('report_date', 'N/A')}
- Sample Type: {json_data.get('report_info', {}).get('sample_type', 'N/A')}

Test Results:
"""
        
        # Add test results
        test_results = json_data.get('test_results', [])
        if isinstance(test_results, list):
            for i, test in enumerate(test_results, 1):
                if isinstance(test, dict):
                    text_content += f"""
Test {i}:
- Test Name: {test.get('test_name', 'N/A')}
- Result Value: {test.get('result_value', 'N/A')}
- Reference Range: {test.get('reference_range', 'N/A')}
- Unit: {test.get('unit', 'N/A')}
- Status: {test.get('status', 'N/A')}
"""
        
        # Add additional info
        additional_info = json_data.get('additional_info', {})
        if isinstance(additional_info, dict):
            text_content += f"""
Additional Information:
- Notes: {additional_info.get('notes', 'N/A')}
- Interpretation: {additional_info.get('interpretation', 'N/A')}
"""
        
        # Create document
        document = Document(
            text=text_content,
            metadata={
                'source_image': report['image_filename'],
                'patient_name': json_data.get('patient_info', {}).get('name', 'Unknown'),
                'hospital_name': json_data.get('hospital_info', {}).get('hospital_name', 'Unknown'),
                'report_type': json_data.get('report_info', {}).get('report_type', 'Unknown'),
                'processing_timestamp': json_data.get('_metadata', {}).get('processing_timestamp', ''),
                'test_count': len(test_results) if isinstance(test_results, list) else 0
            }
        )
        documents.append(document)
    
    return documents

def setup_database_from_json(json_reports, client, collection_name):
    """Set up database from processed JSON reports"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("📄 Processing medical reports...")
        progress_bar.progress(20)
        
        documents = create_documents_from_json_data(json_reports)
        
        if not documents:
            return False, "No valid documents created from reports"
        
        status_text.text("🔄 Loading embedding model...")
        progress_bar.progress(40)
        embed_model = init_embedding()
        
        status_text.text("🔄 Setting up vector store...")
        progress_bar.progress(60)
        vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        status_text.text("🔄 Creating index and storing documents...")
        progress_bar.progress(80)
        
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, embed_model=embed_model, show_progress=False
        )
        
        progress_bar.progress(100)
        status_text.text("✅ Database setup complete!")
        
        return True, f"Successfully indexed {len(documents)} medical reports!"
        
    except Exception as e:
        return False, f"Error setting up database: {str(e)}"

@st.cache_resource
def init_query_engine(_client, collection_name):
    """Initialize the query engine for RAG"""
    try:
        embed_model = init_embedding()
        llm = init_mistral()
        Settings.embed_model = embed_model
        Settings.llm = llm
        
        vector_store = QdrantVectorStore(client=_client, collection_name=collection_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context, embed_model=embed_model)
        
        rerank = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=5)
        
        template = """Context information from medical reports:
                    ---------------------
                    {context_str}
                    ---------------------
                    
Based on the medical reports data above, provide accurate answers to questions about:
- Patient demographics and counts
- Test results and abnormal values
- Hospital information and report types
- Date ranges and temporal queries
- Statistical summaries and trends

If you cannot find specific information in the reports, clearly state that the information is not available.

Question: {query_str}

Answer:"""
        
        qa_prompt_tmpl = PromptTemplate(template)
        query_engine = index.as_query_engine(llm=llm, similarity_top_k=10, node_postprocessors=[rerank])
        query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})
        
        return query_engine
        
    except Exception as e:
        st.error(f"Error initializing query engine: {str(e)}")
        raise e

# ================================
# MAIN STREAMLIT APP
# ================================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Medical Report Analytics System</h1>
        <p style="font-size: 1.2em; margin-top: 10px;">AI-Powered Medical Report Processing & Analysis</p>
        <p style="opacity: 0.9;">Upload medical reports → Extract data → Ask intelligent questions!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize components
    client = init_qdrant()
    collection_name = "medical_reports_db"
    
    # ================================
    # SIDEBAR: File Upload & Processing
    # ================================
    
    with st.sidebar:
        st.markdown("### 📤 Upload Medical Reports")
        
        uploaded_files = st.file_uploader(
            "Choose medical report images",
            type=['png', 'jpg', 'jpeg', 'pdf'],
            accept_multiple_files=True,
            help="Upload multiple medical report images for batch processing"
        )
        
        if uploaded_files:
            st.success(f"📁 {len(uploaded_files)} files uploaded")
            
            # Check Ollama connection
            ollama_status = st.empty()
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=5)
                if response.status_code == 200:
                    ollama_status.success("🤖 Ollama: Connected")
                else:
                    ollama_status.error("❌ Ollama: Not connected")
            except:
                ollama_status.error("❌ Ollama: Not running")
            
            if st.button("🚀 Process All Reports", use_container_width=True):
                if len(uploaded_files) == 0:
                    st.error("Please upload at least one file")
                    return
                
                # Initialize OCR processor
                try:
                    ocr_processor = init_ocr_processor()
                except Exception as e:
                    st.error(f"Failed to initialize OCR: {e}")
                    return
                
                # Process all uploaded files
                with st.spinner("Processing medical reports..."):
                    processed_reports = []
                    progress_bar = st.progress(0)
                    
                    for i, uploaded_file in enumerate(uploaded_files):
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        # Process the image
                        result = ocr_processor.process_image(tmp_path)
                        result['original_filename'] = uploaded_file.name
                        processed_reports.append(result)
                        
                        # Clean up temp file
                        os.unlink(tmp_path)
                        
                        # Update progress
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    # Store processed reports in session state
                    st.session_state.processed_reports = processed_reports
                    
                    # Show processing results
                    successful = sum(1 for r in processed_reports if r.get('success', False))
                    failed = len(processed_reports) - successful
                    
                    if successful > 0:
                        st.success(f"✅ Successfully processed {successful} reports")
                        
                        # Setup database
                        success, message = setup_database_from_json(processed_reports, client, collection_name)
                        if success:
                            st.success("🔄 Database updated!")
                            st.cache_resource.clear()
                        else:
                            st.error(f"Database error: {message}")
                    
                    if failed > 0:
                        st.warning(f"⚠️ Failed to process {failed} reports")
        
        # Database status
        st.markdown("---")
        st.markdown("### 📊 Database Status")
        
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if collection_name in collection_names:
            st.success("✅ Database Ready")
            try:
                collection_info = client.get_collection(collection_name)
                st.metric("📄 Reports", collection_info.points_count)
            except:
                pass
        else:
            st.warning("⚠️ No data yet")
            st.info("👆 Upload reports to get started")
    
    # ================================
    # MAIN AREA: Chat Interface
    # ================================
    
    # Check if database exists
    if collection_name not in collection_names:
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
            <h3>🚀 Welcome to Medical Report Analytics</h3>
            <p>Upload medical report images using the sidebar to begin analysis!</p>
            <p>Once processed, you can ask questions like:</p>
            <ul style="text-align: left; max-width: 600px; margin: 0 auto;">
                <li>How many patients' data is available?</li>
                <li>What are the abnormal test results?</li>
                <li>Show me reports from a specific hospital</li>
                <li>Which patients have high blood sugar levels?</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Initialize query engine
    try:
        query_engine = init_query_engine(client, collection_name)
    except Exception as e:
        st.error(f"❌ Failed to initialize AI: {str(e)}")
        return
    
    # Chat interface
    tab1, tab2, tab3 = st.tabs(["💬 Ask Questions", "📊 Report Summary", "🔍 Sample Queries"])
    
    with tab1:
        # Initialize chat history
        if "medical_messages" not in st.session_state:
            st.session_state.medical_messages = []
            st.session_state.medical_messages.append({
                "role": "assistant",
                "content": "👋 Hello! I'm your Medical Report Analytics Assistant. I can help you analyze the processed medical reports. Ask me anything about the patient data, test results, or hospital information!"
            })
        
        # Display chat history
        for message in st.session_state.medical_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("💬 Ask about the medical reports data..."):
            st.session_state.medical_messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("🔍 Analyzing medical data..."):
                    try:
                        response = query_engine.query(prompt)
                        st.markdown(str(response))
                        st.session_state.medical_messages.append({"role": "assistant", "content": str(response)})
                    except Exception as e:
                        error_msg = f"❌ Sorry, I encountered an error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.medical_messages.append({"role": "assistant", "content": error_msg})
    
    with tab2:
        st.markdown("### 📊 Processing Summary")
        
        if hasattr(st.session_state, 'processed_reports'):
            reports = st.session_state.processed_reports
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_reports = len(reports)
                st.metric("📄 Total Reports", total_reports)
            
            with col2:
                successful = sum(1 for r in reports if r.get('success', False))
                st.metric("✅ Successfully Processed", successful)
            
            with col3:
                failed = total_reports - successful
                st.metric("❌ Failed", failed)
            
            # Show detailed results
            if successful > 0:
                st.markdown("### 📋 Processed Reports Details")
                
                for i, report in enumerate(reports):
                    if report.get('success', False):
                        with st.expander(f"📑 {report.get('original_filename', f'Report {i+1}')}"):
                            json_data = report['structured_json']
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Patient Info:**")
                                patient_info = json_data.get('patient_info', {})
                                st.write(f"• Name: {patient_info.get('name', 'N/A')}")
                                st.write(f"• Age: {patient_info.get('age', 'N/A')}")
                                st.write(f"• Gender: {patient_info.get('gender', 'N/A')}")
                            
                            with col2:
                                st.markdown("**Hospital Info:**")
                                hospital_info = json_data.get('hospital_info', {})
                                st.write(f"• Hospital: {hospital_info.get('hospital_name', 'N/A')}")
                                st.write(f"• Report Type: {json_data.get('report_info', {}).get('report_type', 'N/A')}")
                            
                            # Test results count
                            test_results = json_data.get('test_results', [])
                            if isinstance(test_results, list):
                                st.write(f"**Tests Conducted:** {len(test_results)}")
        else:
            st.info("No processed reports yet. Upload and process some medical reports first!")
    
    with tab3:
        st.markdown("### 🔍 Try These Sample Queries")
        
        sample_queries = [
            "📊 How many patients' data is available in the system?",
            "🏥 Which hospitals are represented in the reports?",
            "🧪 What types of medical tests were conducted?",
            "📅 Show me reports from the last month",
            "⚠️ Are there any abnormal test results?",
            "👥 What's the age distribution of patients?",
            "🔬 List all blood test results",
            "📈 Show me patients with high glucose levels"
        ]
        
        for i, query in enumerate(sample_queries):
            if st.button(query, key=f"query_{i}", use_container_width=True):
                # Initialize messages if not exists
                if "medical_messages" not in st.session_state:
                    st.session_state.medical_messages = []
                
                # Add query to chat
                st.session_state.medical_messages.append({"role": "user", "content": query})
                
                # Generate response
                try:
                    with st.spinner("🔍 Processing query..."):
                        response = query_engine.query(query)
                        st.session_state.medical_messages.append({"role": "assistant", "content": str(response)})
                    st.success("✅ Query processed! Check the 'Ask Questions' tab.")
                except Exception as e:
                    error_msg = f"❌ Sorry, I encountered an error: {str(e)}"
                    st.session_state.medical_messages.append({"role": "assistant", "content": error_msg})
                    st.error("Failed to process query.")
                
                # Refresh to show conversation
                st.rerun()

if __name__ == "__main__":
    main()
