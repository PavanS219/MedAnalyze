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
import logging
from PIL import Image
import io
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    .error-details {
        background: #ffe6e6;
        border: 1px solid #ff9999;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ================================
# ENHANCED MEDICAL OCR CLASS
# ================================

class MedicalReportOCR:
    def __init__(self, ollama_url="http://localhost:11434", model_name="llama3.2:1b"):
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.ocr_reader = None
        
        # Initialize EasyOCR with better error handling
        self._init_ocr()
        
        # Test Ollama connection
        self._test_ollama_connection()
    
    def _init_ocr(self):
        """Initialize EasyOCR with proper error handling"""
        try:
            with st.spinner("Initializing OCR engine..."):
                self.ocr_reader = easyocr.Reader(['en'], gpu=False)  # Disable GPU to avoid issues
            logger.info("EasyOCR initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            st.error(f"Failed to initialize EasyOCR: {e}")
            raise
    
    def _test_ollama_connection(self):
        """Test Ollama connection and model availability with timeout"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models_data = response.json()
                available_models = [model['name'] for model in models_data.get('models', [])]
                
                if self.model_name in available_models:
                    st.success(f"✅ Ollama connected - Model {self.model_name} ready")
                else:
                    st.warning(f"⚠️ Model {self.model_name} not found. Available: {available_models}")
                    st.info(f"Run: `ollama pull {self.model_name}` to download the model")
            else:
                st.error(f"❌ Ollama API returned status {response.status_code}")
        except requests.exceptions.Timeout:
            st.error("❌ Ollama connection timeout - check if Ollama is running")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to Ollama - ensure it's running on the correct port")
        except Exception as e:
            st.error(f"❌ Ollama connection error: {e}")
            st.info("Make sure Ollama is running: `ollama serve`")
    
    def _validate_image(self, image_path):
        """Validate image file before processing"""
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                return False, "Image file not found"
            
            # Check file size (max 50MB)
            file_size = os.path.getsize(image_path)
            if file_size > 50 * 1024 * 1024:
                return False, "Image file too large (max 50MB)"
            
            # Try to open with PIL to validate
            with Image.open(image_path) as img:
                img.verify()
            
            # Try to read with OpenCV
            test_img = cv2.imread(image_path)
            if test_img is None:
                return False, "Cannot read image file with OpenCV"
            
            return True, "Valid image"
            
        except Exception as e:
            return False, f"Image validation error: {str(e)}"
    
    def preprocess_image(self, image_path):
        """Enhanced image preprocessing with error handling"""
        try:
            # Validate image first
            is_valid, validation_msg = self._validate_image(image_path)
            if not is_valid:
                raise ValueError(validation_msg)
            
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Get image dimensions
            height, width = img.shape[:2]
            logger.info(f"Image dimensions: {width}x{height}")
            
            # Resize if too large (keep aspect ratio)
            max_dimension = 2000
            if max(height, width) > max_dimension:
                scale = max_dimension / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                logger.info(f"Resized image to: {new_width}x{new_height}")
            
            # Convert to grayscale
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            
            # Apply denoising
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Enhance contrast using CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Image preprocessing error: {e}")
            raise
    
    def extract_text_easyocr(self, image_path):
        """Enhanced text extraction with better error handling"""
        try:
            if self.ocr_reader is None:
                raise ValueError("OCR reader not initialized")
            
            # Process image with enhanced preprocessing
            processed_img = self.preprocess_image(image_path)
            
            # Extract text using EasyOCR with timeout
            logger.info("Starting OCR text extraction...")
            results = self.ocr_reader.readtext(processed_img)
            
            extracted_texts = []
            full_text_parts = []
            
            # Filter and process results with better confidence threshold
            for (bbox, text, confidence) in results:
                if confidence > 0.2:  # Lower threshold for medical text
                    cleaned_text = text.strip()
                    if cleaned_text and len(cleaned_text) > 1:
                        # Remove obvious OCR errors
                        if not self._is_garbage_text(cleaned_text):
                            extracted_texts.append({
                                'text': cleaned_text,
                                'confidence': round(confidence * 100, 2),
                                'bbox': bbox
                            })
                            full_text_parts.append(cleaned_text)
            
            # Combine all text with proper spacing
            full_text = ' '.join(full_text_parts)
            
            # Clean up common OCR errors
            full_text = self._clean_ocr_text(full_text)
            
            logger.info(f"Extracted {len(extracted_texts)} text blocks, {len(full_text)} characters total")
            
            return full_text, extracted_texts
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return "", []
    
    def _is_garbage_text(self, text):
        """Filter out garbage OCR text"""
        # Filter very short text
        if len(text.strip()) < 2:
            return True
        
        # Filter text with too many special characters
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        if special_chars > len(text) * 0.5:
            return True
        
        # Filter repeated single characters
        if len(set(text.replace(' ', ''))) == 1:
            return True
        
        return False
    
    def _clean_ocr_text(self, text):
        """Clean common OCR errors in medical text"""
        import re
        
        if not text:
            return text
        
        # Common OCR corrections for medical text
        corrections = {
            r'\b0\b': 'O',  # Zero to O
            r'\bl\b': 'I',  # lowercase l to I
            r'(\d+)\s*-\s*(\d+)': r'\1-\2',  # Fix range formatting
            r'(\d+)\s*\.\s*(\d+)': r'\1.\2',  # Fix decimal numbers
            r'\s+': ' ',  # Multiple spaces to single space
        }
        
        cleaned_text = text
        for pattern, replacement in corrections.items():
            try:
                cleaned_text = re.sub(pattern, replacement, cleaned_text)
            except:
                continue
        
        return cleaned_text.strip()
    
    def generate_json_with_ollama(self, extracted_text, image_filename):
        """Enhanced JSON generation with better error handling"""
        
        if not extracted_text or len(extracted_text.strip()) < 10:
            return {
                'success': False,
                'error': 'Insufficient text extracted from image for analysis',
                'raw_response': None
            }
        
        # Truncate text if too long
        max_text_length = 4000
        if len(extracted_text) > max_text_length:
            extracted_text = extracted_text[:max_text_length] + "\n[TEXT TRUNCATED]"
        
        # Simplified prompt for better success rate
        prompt = f"""Extract medical report information from this OCR text and format as JSON:

TEXT: {extracted_text}

Create JSON with these fields (use null if not found):
{{
  "hospital_info": {{
    "hospital_name": "string or null",
    "address": "string or null"
  }},
  "patient_info": {{
    "name": "string or null",
    "age": "string or null",
    "gender": "string or null"
  }},
  "doctor_info": {{
    "referring_doctor": "string or null"
  }},
  "report_info": {{
    "report_type": "string or null",
    "report_date": "string or null"
  }},
  "test_results": [
    {{
      "test_name": "string",
      "result_value": "string",
      "reference_range": "string or null",
      "unit": "string or null"
    }}
  ]
}}

Return only valid JSON, no extra text."""

        try:
            request_data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 1024
                }
            }
            
            logger.info("Sending request to Ollama...")
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=request_data,
                timeout=120  # Reduced timeout
            )
            
            if response.status_code != 200:
                return {
                    'success': False,
                    'error': f'Ollama API error: HTTP {response.status_code}',
                    'raw_response': response.text
                }
            
            try:
                result = response.json()
            except json.JSONDecodeError as e:
                return {
                    'success': False,
                    'error': f'Invalid JSON response from Ollama: {str(e)}',
                    'raw_response': response.text[:500]
                }
            
            json_text = result.get('response', '').strip()
            
            if not json_text:
                return {
                    'success': False,
                    'error': 'Empty response from Ollama',
                    'raw_response': str(result)
                }
            
            # Clean and extract JSON
            json_text = self._extract_and_clean_json(json_text)
            
            # Parse JSON with fallback
            try:
                parsed_json = json.loads(json_text)
            except json.JSONDecodeError as e:
                # Try to fix and parse again
                fixed_json = self._fix_json_errors(json_text)
                try:
                    parsed_json = json.loads(fixed_json)
                except json.JSONDecodeError:
                    # Return a basic structure if parsing fails
                    parsed_json = {
                        "hospital_info": {"hospital_name": None, "address": None},
                        "patient_info": {"name": None, "age": None, "gender": None},
                        "doctor_info": {"referring_doctor": None},
                        "report_info": {"report_type": "Medical Report", "report_date": None},
                        "test_results": [],
                        "_extraction_note": "Partial extraction due to JSON parsing issues"
                    }
            
            # Add metadata
            parsed_json['_metadata'] = {
                'source_image': image_filename,
                'extraction_method': 'easyocr_ollama_enhanced',
                'processing_timestamp': datetime.now().isoformat(),
                'model_used': self.model_name,
                'text_length': len(extracted_text)
            }
            
            return {
                'success': True,
                'json_data': parsed_json,
                'raw_response': json_text
            }
            
        except requests.exceptions.Timeout:
            return {
                'success': False,
                'error': 'Ollama request timeout - try reducing image size or text length',
                'raw_response': None
            }
        except requests.exceptions.ConnectionError:
            return {
                'success': False,
                'error': 'Cannot connect to Ollama - ensure it is running',
                'raw_response': None
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Unexpected error in Ollama processing: {str(e)}',
                'raw_response': None
            }
    
    def _extract_and_clean_json(self, json_text):
        """Extract and clean JSON from response"""
        import re
        
        # Remove markdown code blocks
        if '```json' in json_text:
            json_text = json_text.split('```json')[1].split('```')[0]
        elif '```' in json_text:
            json_text = json_text.split('```')[1].split('```')[0]
        
        json_text = json_text.strip()
        
        # Find JSON object
        if not json_text.startswith('{'):
            json_match = re.search(r'(\{.*\})', json_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
        
        return json_text
    
    def _fix_json_errors(self, json_text):
        """Attempt to fix common JSON errors"""
        import re
        
        fixes = [
            (r',\s*}', '}'),  # Remove trailing commas before }
            (r',\s*]', ']'),  # Remove trailing commas before ]
            (r':\s*,', ': null,'),  # Fix empty values
            (r':\s*}', ': null}'),  # Fix empty values at end
        ]
        
        fixed_json = json_text
        for pattern, replacement in fixes:
            fixed_json = re.sub(pattern, replacement, fixed_json)
        
        return fixed_json
    
    def process_image(self, image_path):
        """Process image with comprehensive error handling"""
        image_filename = os.path.basename(image_path)
        
        try:
            logger.info(f"Processing image: {image_filename}")
            
            # Validate image first
            is_valid, validation_msg = self._validate_image(image_path)
            if not is_valid:
                return {
                    'success': False,
                    'error': f'Image validation failed: {validation_msg}',
                    'image_filename': image_filename
                }
            
            # Extract text
            extracted_text, extraction_details = self.extract_text_easyocr(image_path)
            
            if not extracted_text.strip():
                return {
                    'success': False,
                    'error': 'No readable text found in image',
                    'image_filename': image_filename,
                    'extraction_details': extraction_details
                }
            
            # Generate structured JSON
            ollama_result = self.generate_json_with_ollama(extracted_text, image_filename)
            
            if ollama_result['success']:
                return {
                    'success': True,
                    'image_filename': image_filename,
                    'extracted_text': extracted_text,
                    'extraction_details': extraction_details,
                    'structured_json': ollama_result['json_data'],
                    'text_blocks_count': len(extraction_details),
                    'text_confidence_avg': sum(item['confidence'] for item in extraction_details) / len(extraction_details) if extraction_details else 0
                }
            else:
                return {
                    'success': False,
                    'error': ollama_result['error'],
                    'extracted_text': extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text,
                    'image_filename': image_filename,
                    'ollama_details': ollama_result
                }
                
        except Exception as e:
            logger.error(f"Processing error for {image_filename}: {e}")
            return {
                'success': False,
                'error': f'Processing error: {str(e)}',
                'image_filename': image_filename,
                'traceback': traceback.format_exc()
            }

# ================================
# INITIALIZATION FUNCTIONS
# ================================

@st.cache_resource
def init_qdrant():
    """Initialize Qdrant Cloud client with better error handling"""
    try:
        qdrant_url = st.secrets.get("QDRANT_URL") or os.getenv("QDRANT_URL")
        qdrant_api_key = st.secrets.get("QDRANT_API_KEY") or os.getenv("QDRANT_API_KEY")
        
        if not qdrant_url or not qdrant_api_key:
            st.error("❌ Qdrant Cloud credentials not found in secrets or environment!")
            st.info("Please add QDRANT_URL and QDRANT_API_KEY to your Streamlit secrets or environment variables")
            st.stop()
        
        client = qdrant_client.QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        
        # Test connection
        collections = client.get_collections()
        st.success(f"✅ Connected to Qdrant Cloud - {len(collections.collections)} collections found")
        return client
        
    except Exception as e:
        st.error(f"❌ Cannot connect to Qdrant Cloud: {str(e)}")
        st.info("Please check your Qdrant Cloud credentials and connection")
        st.stop()

@st.cache_resource
def init_embedding():
    """Initialize embedding model with error handling"""
    try:
        return HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)
    except Exception as e:
        st.error(f"Failed to initialize embedding model: {e}")
        st.stop()

@st.cache_resource
def init_mistral():
    """Initialize Mistral AI model with error handling"""
    try:
        api_key = st.secrets.get("MISTRAL_API_KEY") or os.getenv("MISTRAL_API_KEY")
        if not api_key:
            st.error("❌ MISTRAL_API_KEY not found!")
            st.info("Please add MISTRAL_API_KEY to your Streamlit secrets or environment variables")
            st.stop()
        return MistralAI(model="mistral-large-latest", temperature=0.7, max_tokens=512, api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize Mistral AI: {e}")
        st.stop()

@st.cache_resource
def init_ocr_processor():
    """Initialize OCR processor with error handling"""
    try:
        return MedicalReportOCR()
    except Exception as e:
        st.error(f"Failed to initialize OCR processor: {e}")
        raise

# ================================
# HELPER FUNCTIONS
# ================================

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary location"""
    try:
        # Create temporary file with proper extension
        file_ext = uploaded_file.name.split('.')[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}")
        raise

def display_processing_results(processed_reports):
    """Display detailed processing results"""
    successful = [r for r in processed_reports if r.get('success', False)]
    failed = [r for r in processed_reports if not r.get('success', False)]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📄 Total Files", len(processed_reports))
    with col2:
        st.metric("✅ Successfully Processed", len(successful))
    with col3:
        st.metric("❌ Failed", len(failed))
    
    # Show failed reports with details
    if failed:
        st.warning("⚠️ Some files failed to process:")
        for report in failed:
            with st.expander(f"❌ {report.get('image_filename', 'Unknown file')} - Error Details"):
                st.error(f"**Error:** {report.get('error', 'Unknown error')}")
                
                if 'extracted_text' in report and report['extracted_text']:
                    st.write("**Extracted Text (partial):**")
                    st.text(report['extracted_text'][:300] + "..." if len(report['extracted_text']) > 300 else report['extracted_text'])
                
                if 'ollama_details' in report:
                    st.write("**Ollama Processing Details:**")
                    st.json(report['ollama_details'])
    
    return len(successful) > 0

# ================================
# DATABASE FUNCTIONS (Simplified)
# ================================

def create_documents_from_json_data(json_reports):
    """Create documents from JSON data with error handling"""
    documents = []
    
    for report in json_reports:
        if not report.get('success', False):
            continue
        
        try:
            json_data = report['structured_json']
            
            # Create text content
            text_parts = []
            
            # Hospital info
            hospital_info = json_data.get('hospital_info', {})
            if hospital_info.get('hospital_name'):
                text_parts.append(f"Hospital: {hospital_info['hospital_name']}")
            
            # Patient info
            patient_info = json_data.get('patient_info', {})
            if patient_info.get('name'):
                text_parts.append(f"Patient: {patient_info['name']}")
            if patient_info.get('age'):
                text_parts.append(f"Age: {patient_info['age']}")
            if patient_info.get('gender'):
                text_parts.append(f"Gender: {patient_info['gender']}")
            
            # Report info
            report_info = json_data.get('report_info', {})
            if report_info.get('report_type'):
                text_parts.append(f"Report Type: {report_info['report_type']}")
            
            # Test results
            test_results = json_data.get('test_results', [])
            if isinstance(test_results, list):
                for test in test_results:
                    if isinstance(test, dict) and test.get('test_name'):
                        test_text = f"Test: {test['test_name']}"
                        if test.get('result_value'):
                            test_text += f" Result: {test['result_value']}"
                        if test.get('unit'):
                            test_text += f" {test['unit']}"
                        text_parts.append(test_text)
            
            # Add original extracted text
            if 'extracted_text' in report:
                text_parts.append(f"Original Text: {report['extracted_text']}")
            
            text_content = "\n".join(text_parts)
            
            # Create document
            document = Document(
                text=text_content,
                metadata={
                    'source_image': report['image_filename'],
                    'patient_name': patient_info.get('name', 'Unknown'),
                    'hospital_name': hospital_info.get('hospital_name', 'Unknown'),
                    'report_type': report_info.get('report_type', 'Medical Report'),
                    'processing_timestamp': json_data.get('_metadata', {}).get('processing_timestamp', ''),
                    'test_count': len(test_results) if isinstance(test_results, list) else 0
                }
            )
            documents.append(document)
            
        except Exception as e:
            logger.error(f"Error creating document from report {report.get('image_filename', 'unknown')}: {e}")
            continue
    
    return documents

def setup_database_from_json(json_reports, client, collection_name):
    """Setup database with better error handling"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("📄 Processing medical reports...")
        progress_bar.progress(20)
        
        documents = create_documents_from_json_data(json_reports)
        
        if not documents:
            return False, "No valid documents created from reports"
        
        status_text.text("🔄 Initializing embedding model...")
        progress_bar.progress(40)
        embed_model = init_embedding()
        
        status_text.text("🔄 Setting up vector store...")
        progress_bar.progress(60)
        
        # Delete existing collection if it exists
        try:
            client.delete_collection(collection_name)
        except:
            pass
        
        vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        status_text.text("🔄 Creating index and storing documents...")
        progress_bar.progress(80)
        
        index = VectorStoreIndex.from_documents(
            documents, 
            storage_context=storage_context, 
            embed_model=embed_model, 
            show_progress=False
        )
        
        progress_bar.progress(100)
        status_text.text("✅ Database setup complete!")
        
        return True, f"Successfully indexed {len(documents)} medical reports!"
        
    except Exception as e:
        logger.error(f"Database setup error: {e}")
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
            type=['png', 'jpg', 'jpeg'],  # Removed PDF for now
            accept_multiple_files=True,
            help="Upload medical report images (PNG, JPG, JPEG)"
        )
        
        if uploaded_files:
            st.success(f"📁 {len(uploaded_files)} files uploaded")
            
            # Ollama status check
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
