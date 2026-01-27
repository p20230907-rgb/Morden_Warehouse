import os
import re
import pandas as pd
import rclpy
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sys
import time
import logging
from collections import defaultdict
from typing import Dict, Tuple, Optional, Any, List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
CORS(app)
app.logger.setLevel(logging.INFO)

# Add paths to the system
new_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, new_path)

try:
    from warehouse_sim.warehouse_sim.navigate_to_shelf import NavToShelf
except ImportError:
    app.logger.warning("Warning: Could not import NavToShelf. Robot navigation functionality may be limited.")
    class NavToShelf:
        def __init__(self):
            app.logger.info("NavToShelf (mock) initialized: Robot navigation is not active.")
        def send_goal(self, x, y, z, ox, oy, oz, ow):
            app.logger.info(f"NavToShelf (mock): Goal sent to X:{x}, Y:{y}, Z:{z}")
            # Simulate ROS processing time
            time.sleep(1) 
        def destroy_node(self):
            app.logger.info("NavToShelf (mock): Node destroyed.")

# Initialize ROS only once
if not rclpy.ok():
    rclpy.init()

# Load environment variables
load_dotenv()
api_key: str = os.getenv('key')
if not api_key:
    raise ValueError("API key not found. Please check your .env file.")
# model_names = ["llama3-70b-8192", "deepseek-r1-distill-llama-70b", "gemma2-9b-it"]
# model_names = ["llama-3.3-70b-versatile", "meta-llama/llama-4-scout-17b-16e-instruct", "openai/gpt-oss-120b"]
# Initialize LLM
model: str = "llama-3.3-70b-versatile"
llm = ChatGroq(api_key=api_key, model_name=model)
parser = StrOutputParser()

# --- Global variables for data (will be loaded by _load_shelf_data) ---
shelf_df: pd.DataFrame = pd.DataFrame()
shelf_locations: Dict[str, Dict[str, Any]] = {}
shelf_name_to_info: Dict[str, Dict[str, Any]] = {}
shelf_id_to_info: Dict[str, Dict[str, Any]] = {}
all_items_for_order_page: List[Dict[str, Any]] = []
traditional_search_corpus: List[str] = []
safety_manual_content: str = ""
safety_chunks_docs: List[Any] = [] # Use Any for LangChain Document objects
vector_store = None # Initialize as None
embedding_cache: Dict[str, Any] = {} # Initialize embedding_cache globally
all_combined_texts: List[str] = [] # Initialize all_combined_texts globally

# Define script_dir globally
script_dir = os.path.dirname(os.path.abspath(__file__))

# --- SentenceTransformer Embeddings Wrapper Class (MOVED TO GLOBAL SCOPE) ---
# This class wraps the SentenceTransformer model to provide the interface
# expected by LangChain's FAISS vector store for embeddings.
class SentenceTransformerEmbeddings:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        # Embed a list of texts (for documents)
        return self.model.encode(texts).tolist()

    def embed_query(self, query):
        # Embed a single query text
        return self.model.encode([query])[0].tolist()

def _load_shelf_data():
    """Loads shelf data from CSV and rebuilds in-memory data structures."""
    global shelf_df, shelf_locations, shelf_name_to_info, shelf_id_to_info, all_items_for_order_page, traditional_search_corpus, safety_manual_content, safety_chunks_docs, vector_store, embedding_cache, all_combined_texts

    # Use the globally defined script_dir
    csv_path = os.path.join(script_dir, 'shelfinfo.csv')
    
    try:
        shelf_df = pd.read_csv(csv_path)
        app.logger.info("Shelf data loaded successfully from shelfinfo.csv")
        expected_columns = ['Shelf Name', 'Item', 'X Position', 'Y Position', 'Z Position', 'Stock', 'Shelf ID']
        for col in expected_columns:
            if col not in shelf_df.columns:
                shelf_df[col] = None
        # Add optional columns if not present
        if 'Category' not in shelf_df.columns:
            shelf_df['Category'] = 'Electronics'
        if 'Value' not in shelf_df.columns:
            shelf_df['Value'] = 0.0
    except FileNotFoundError:
        app.logger.error(f"Error: shelfinfo.csv not found at {csv_path}.")
        shelf_df = pd.DataFrame(columns=['Shelf Name', 'Item', 'X Position', 'Y Position', 'Z Position', 'Stock', 'Shelf ID', 'Category', 'Value'])
        return
    except Exception as e:
        app.logger.error(f"An error occurred reading shelfinfo.csv: {e}")
        shelf_df = pd.DataFrame(columns=['Shelf Name', 'Item', 'X Position', 'Y Position', 'Z Position', 'Stock', 'Shelf ID', 'Category', 'Value'])
        return
    shelf_df['Category'] = shelf_df['Category'].fillna('Electronics')
    shelf_df['Value'] = shelf_df['Value'].fillna(0.0)
    # Rebuild shelf_locations
    shelf_locations.clear()
    for _, row in shelf_df.iterrows():
        shelf_locations[row['Item']] = {
            'x': row['X Position'],
            'y': row['Y Position'],
            'z': row['Z Position'],
            'stock': row['Stock'],
            'shelf_name': row['Shelf Name'],
            'shelf_id': row['Shelf ID']
        }
    app.logger.info(f"Rebuilt shelf_locations with {len(shelf_locations)} items.")

    # Rebuild shelf_name_to_info and shelf_id_to_info (handle potential duplicates by taking last entry)
    shelf_name_to_info.clear()
    shelf_id_to_info.clear()
    for _, row in shelf_df.iterrows():
        shelf_name_to_info[row['Shelf Name']] = {
            'shelf_id': row['Shelf ID'],
            'x': row['X Position'],
            'y': row['Y Position'],
            'z': row['Z Position'],
            'item': row['Item'],
            'stock': row['Stock'],
            'shelf_name': row['Shelf Name']
        }
        shelf_id_to_info[row['Shelf ID']] = {
            'shelf_id': row['Shelf ID'],
            'x': row['X Position'],
            'y': row['Y Position'],
            'z': row['Z Position'],
            'item': row['Item'],
            'stock': row['Stock'],
            'shelf_name': row['Shelf Name']
        }
    app.logger.info(f"Rebuilt shelf_name_to_info and shelf_id_to_info.")

    # Rebuild all_items_for_order_page (for Jinja template)
    all_items_for_order_page.clear()
    unique_items_set = set() 
    for _, row in shelf_df.iterrows():
        item_name = row['Item']
        if item_name not in unique_items_set:
            all_items_for_order_page.append({
                'item_name': item_name,
                'shelf_name': row['Shelf Name'],
                'shelf_id': row['Shelf ID'],
                'x_position': row['X Position'],
                'y_position': row['Y Position'],
                'z_position': row['Z Position'],
                'stock': row['Stock']
            })
            unique_items_set.add(item_name)
    app.logger.info(f"Rebuilt all_items_for_order_page with {len(all_items_for_order_page)} unique items.")

    # --- LLM Shelf Extractor Chain (relies on llm, parser) ---
    global llm_shelf_extractor_chain # Declare global to assign
    llm_shelf_extractor_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a warehouse assistant. Your task is to extract a specific shelf ID (e.g., 'A', 'B', 'C') or a full shelf name (e.g., 'Shelf A', 'Shelf B') if mentioned in the user's query. If no shelf is explicitly mentioned, or if the query is not about a shelf, respond with 'NONE'. Do not provide any other text."),
        ("human", "{query}")
        
    ])
    llm_shelf_extractor_chain = {"query": RunnablePassthrough()} | llm_shelf_extractor_prompt | llm | parser

    # --- RAG LLM Chain (relies on llm, parser) ---
    global rag_llm_chain # Declare global to assign
    rag_llm_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful warehouse robot assistant. Use ONLY the following provided context, which may include details about items, their location, stock, and warehouse safety procedures, to answer the user's question. If the answer is not in the context, politely state that you don't have that information. Do not invent information. Format the answer clearly, including the item, shelf, coordinates, stock, and any relevant safety information if applicable."),
        ("human", "Context: {context}\n\nUser query: {query}")
    ])
    rag_llm_chain = rag_llm_prompt | llm | parser
    
    # --- Direct LLM Answer Chain (for LLM-only reasoning) ---
    global llm_answer_chain # Declare global to assign
    # This chain will take a dictionary with 'query', 'facts_prompt', and 'safety_info' as input
    llm_answer_template_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a warehouse assistant with the following known information:\n\n"
         "--- Inventory and Shelf Info ---\n"
         "{facts_prompt}\n\n" # This will be passed as a variable
         "--- Warehouse Safety Manual ---\n"
         "{safety_info}\n\n"  # This will be passed as a variable
         "Based on the provided information, answer the user's question. If the information is not present, state that you cannot answer."
        ),
        ("human", "{query}")
    ])
    # The chain now correctly processes inputs for the template
    llm_answer_chain = (
        llm_answer_template_prompt
        | llm
        | parser
    )

    # --- SentenceTransformer Model Initialization ---
    global embeddings, get_embedding # embedding_cache is already global
    embeddings = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embedding_cache.clear() # Clear existing cache
    
    def get_embedding(text: str) -> np.ndarray:
        """Get cached embedding or compute new one"""
        if text not in embedding_cache:
            embedding_cache[text] = embeddings.encode(text)
        return embedding_cache[text]

    # --- Prepare Safety Manual for RAG from file ---
    global safety_manual_content, safety_chunks_docs # Declare global
    safety_manual_filename = "data/WAREHOUSE SAFETY STANDARD OPERATING MANUAL - 2025 (WHS-SOP).txt"
    safety_manual_filepath = os.path.join(script_dir, safety_manual_filename)

    safety_manual_content = ""
    try:
        with open(safety_manual_filepath, 'r', encoding='utf-8') as f:
            safety_manual_content = f.read()
        app.logger.info(f"Successfully loaded {safety_manual_filename}")
    except FileNotFoundError:
        app.logger.error(f"Error: {safety_manual_filename} not found at {safety_manual_filepath}. Safety information will be unavailable.")
        safety_manual_content = ""
    except Exception as e:
        app.logger.error(f"An error occurred reading {safety_manual_filename}: {e}")
        safety_manual_content = ""

    safety_manual_name = "WAREHOUSE SAFETY STANDARD OPERATING MANUAL - 2025 (WHS-SOP)"

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    safety_chunks_docs.clear()
    if safety_manual_content:
        safety_chunks_docs.extend(text_splitter.create_documents([safety_manual_content]))
        app.logger.info(f"Number of safety manual chunks created: {len(safety_chunks_docs)}")
    else:
        app.logger.warning("Skipping safety manual embedding due to empty content (confirmed).")

    safety_texts_with_metadata = []
    if safety_manual_content:
        for i, chunk_doc in enumerate(safety_chunks_docs):
            header_match = re.search(r"SECTION \d+: ([^\n]+)", chunk_doc.page_content)
            section_header = header_match.group(1).strip() if header_match else f"Chunk {i+1}"
            safety_texts_with_metadata.append(f"Source: {safety_manual_name}, Section: {section_header}, Content: {chunk_doc.page_content}")

    # Rebuild traditional_search_corpus
    traditional_search_corpus.clear()
    for _, row in shelf_df.iterrows():
        traditional_search_corpus.append(
            f"Item: {row['Item']}. Shelf Name: {row['Shelf Name']}. Shelf ID: {row['Shelf ID']}. Stock: {row['Stock']}."
        )
    for chunk_doc in safety_chunks_docs:
        traditional_search_corpus.append(chunk_doc.page_content)
    app.logger.info(f"Traditional Search Corpus rebuilt with {len(traditional_search_corpus)} entries.")

    # Prepare shelf_texts for vector store
    shelf_texts = [f"Item: {row['Item']}, Shelf: {row['Shelf Name']}, Shelf ID: {row['Shelf ID']}, Location: ({row['X Position']:.3f}, {row['Y Position']:.3f}, {row['Z Position']:.3f}), Stock: {row['Stock']}" for _, row in shelf_df.iterrows()]
    
    # Combine all texts for the unified vector store
    all_combined_texts.clear() # Clear before extending
    all_combined_texts.extend(shelf_texts + safety_texts_with_metadata)

    # Rebuild vector_store
    global vector_store # Declare global
    vector_store = None # Ensure it's reset before trying to build
    if all_combined_texts:
        try:
            vector_store = FAISS.from_texts(
                texts=all_combined_texts,
                embedding=SentenceTransformerEmbeddings(embeddings)
            )
            app.logger.info(f"FAISS vector store built with {len(all_combined_texts)} chunks.")
        except Exception as e:
            app.logger.error(f"Error building FAISS vector store: {e}")
    else:
        app.logger.error("No text content available to build the vector store. Check shelfinfo.csv and safety manual file. RAG will be unavailable.")


_load_shelf_data() # Initial load of data on app startup

def parse_rag_result(result_text: str) -> Dict[str, Any]:
    parsed_data = {
        'item': 'N/A', 'location': {'x': 'N/A', 'y': 'N/A', 'z': 'N/A'},
        'stock': 'N/A', 'shelf': 'N/A', 'shelf_id': 'N/A', 'safety_info': 'N/A'
    }

    item_match = re.search(r"Item:\s*([^\n,]+)", result_text, re.IGNORECASE)
    shelf_name_match = re.search(r"Shelf:\s*([^\n(]+)", result_text, re.IGNORECASE)
    shelf_id_match = re.search(r"ID:\s*([A-Z0-9]+)", result_text, re.IGNORECASE)
    location_match = re.search(r"Location:\s*\(([^)]+)\)", result_text, re.IGNORECASE)
    stock_match = re.search(r"Stock:\s*(\d+)", result_text, re.IGNORECASE)

    if item_match: parsed_data['item'] = item_match.group(1).strip()
    if shelf_name_match: parsed_data['shelf'] = shelf_name_match.group(1).strip()
    if shelf_id_match: parsed_data['shelf_id'] = shelf_id_match.group(1).strip()
    if location_match:
        try:
            coords = [float(c.strip()) for c in location_match.group(1).split(',')]
            if len(coords) == 3: parsed_data['location'] = {'x': coords[0], 'y': coords[1], 'z': coords[2]}
        except ValueError: app.logger.warning(f"Could not parse location coordinates from: {location_match.group(1)}")
    if stock_match:
        try: parsed_data['stock'] = int(stock_match.group(1))
        except ValueError: app.logger.warning(f"Could not parse stock quantity from: {stock_match.group(1)}")
    
    safety_keywords = ["safety", "ppe", "fire", "hazardous", "loto", "equipment", "manual", "procedure", "report", "training", "exit", "emergency"]
    is_safety_focused_query = any(keyword in result_text.lower() for keyword in safety_keywords)
    has_inventory_data_extracted = (parsed_data['item'] != 'N/A' or parsed_data['shelf'] != 'N/A' or parsed_data['shelf_id'] != 'N/A' or parsed_data['stock'] != 'N/A')

    if is_safety_focused_query and not has_inventory_data_extracted:
        parsed_data['safety_info'] = result_text 
    elif is_safety_focused_query and has_inventory_data_extracted:
        parsed_data['safety_info'] = "Relevant safety information available in message."

    return parsed_data

def traditional_keyword_search(query: str, texts: list[str]) -> Optional[str]:
    query_words = set(word.lower() for word in re.findall(r'\b\w+\b', query) if word.isalnum())
    if not query_words: return None
    best_match_content = None
    max_matches = 0
    for chunk_text in texts:
        chunk_words = set(word.lower() for word in re.findall(r'\b\w+\b', chunk_text) if word.isalnum())
        matches = len(query_words.intersection(chunk_words))
        if matches > max_matches:
            max_matches = matches
            best_match_content = chunk_text
        if matches == len(query_words) and len(query_words) > 0:
            return best_match_content
    if max_matches > 0: return best_match_content
    return None

@app.route('/')
def index():
    # Calculate real metrics from the DataFrame
    total_items = len(shelf_df)
    total_units = shelf_df['Stock'].sum() if 'Stock' in shelf_df.columns else 0
    total_value = (shelf_df['Stock'] * shelf_df['Value']).sum() if 'Value' in shelf_df.columns else 0
    print(total_value)
    low_stock_items = len(shelf_df[shelf_df['Stock'] < 5]) if 'Stock' in shelf_df.columns else 0
    categories = shelf_df['Category'].nunique() if 'Category' in shelf_df.columns else 1
    
    # Convert shelves to records for the template
    shelves_data = shelf_df.to_dict('records')
    
    # Pass all data to template
    return render_template('index.html', 
                         shelves=shelves_data,
                         total_items=total_items,
                         total_units=total_units,
                         total_value=total_value,
                         low_stock_items=low_stock_items,
                         categories=categories,
                         shelf_df=shelf_df) 

# New route for the Performance Test page
@app.route('/performance_test_page')
def performance_test_page():
    return render_template('comparision_tool.html')

@app.route('/assistance')
def assistance():
    return render_template('assistance.html', shelves=shelf_df.to_dict('records'))


@app.route('/robot_assistance')
def robot_assistance_page(): 
    return render_template('robot_assistance.html', shelves=shelf_df.to_dict('records'))

@app.route('/order_management')
def order_management():
    # Reload shelf data each time to ensure latest stock levels are displayed
    _load_shelf_data() 
    return render_template('order_management.html', all_items=all_items_for_order_page)

@app.route('/performance_dashboard')
def performance_dashboard():
    return render_template('performance_dashboard.html')

# New route for the Deep Thinking Assistant page
@app.route('/deep_thinking_assistant')
def deep_thinking_assistant():
    return render_template('deep_thinking_assistant.html')

# New route for the LLM Only Assistant page
@app.route('/llm_only_assistant')
def llm_only_assistant():
    return render_template('llm_only_assistant.html')


# New API endpoint for Deep Thinking queries
@app.route('/deep_query', methods=['POST'])
def deep_query():
    if not request.is_json:
        return jsonify({"status": "error", "message": "Request must be JSON"}), 415
    
    user_query = request.json.get('query')
    if not user_query:
        return jsonify({"status": "error", "message": "Query cannot be empty."}), 400

    app.logger.info(f"Received deep query: '{user_query}'")

    try:
        # Prepare comprehensive context for the LLM
        # Convert pandas DataFrame to a list of dictionaries for easier LLM consumption
        inventory_data_for_llm = shelf_df.to_dict(orient='records')
        
        # Format inventory data into a readable string for the LLM
        inventory_context_str = "Warehouse Inventory Data (Item, Shelf, Stock, Location):\n"
        for item_record in inventory_data_for_llm:
            inventory_context_str += (
                f"- Item: {item_record.get('Item', 'N/A')}, "
                f"Shelf: {item_record.get('Shelf Name', 'N/A')} (ID: {item_record.get('Shelf ID', 'N/A')}), "
                f"Stock: {item_record.get('Stock', 'N/A')}, "
                f"Location: (X:{item_record.get('X Position', 'N/A')}, Y:{item_record.get('Y Position', 'N/A')}, Z:{item_record.get('Z Position', 'N/A')})\n"
            )
        
        # Combine all relevant context
        full_context_for_llm = (
            f"Current Date: {time.strftime('%Y-%m-%d')}\n\n"
            f"{inventory_context_str}\n\n"
            f"Warehouse Safety Manual:\n{safety_manual_content}\n\n"
        )

        # Create a sophisticated prompt for deep thinking
        deep_thinking_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are an advanced Warehouse AI Analyst. Your task is to analyze the provided comprehensive "
             "WAREHOUSE INVENTORY DATA and WAREHOUSE SAFETY MANUAL to answer complex user queries. "
             "Perform filtering, aggregation, comparison, or summarization as required by the query. "
             "Strictly use only the information provided in the context. If the answer cannot be found "
             "or derived from the given context, state that clearly and politely. "
             "Present your answer in a clear, concise, and structured format. "
             "If the query involves multiple items or conditions, list them clearly. "
             "For numerical ranges, strictly adhere to the conditions (e.g., 'between 20 and 60 units' means >= 20 and <= 60). "
             "Do NOT invent any data or safety procedures that are not explicitly stated. "
             "Prioritize inventory data for inventory questions and safety manual for safety questions, "
             "but be prepared to combine information if a query spans both.\n\n" # Added instruction for structured list
             "COMPREHENSIVE CONTEXT:\n{context}"
            ),
            ("human", "User Query: {query}")
        ])

        # Invoke the LLM with the deep thinking prompt
        llm_start_time = time.time()
        llm_response = (
            {"context": RunnablePassthrough(), "query": RunnablePassthrough()}
            | deep_thinking_prompt
            | llm
            | parser
        ).invoke({"context": full_context_for_llm, "query": user_query}).strip()
        llm_end_time = time.time()
        llm_processing_time = llm_end_time - llm_start_time

        app.logger.info(f"Deep Query LLM processing time: {llm_processing_time:.4f}s")
        app.logger.info(f"Deep Query LLM Response: {llm_response[:500]}...") # Log beginning of response

        return jsonify({"status": "success", "answer": llm_response}), 200

    except Exception as e:
        app.logger.error(f"Error during deep query processing: {str(e)}")
        return jsonify({"status": "error", "message": f"Failed to process deep query: {str(e)}"}), 500


@app.route('/get_shelf_location', methods=['POST'])
def get_shelf_location():
    if not request.is_json:
        return jsonify({"status": "error", "message": "Request must be JSON"}), 415
    item_name = request.json.get('item')
    if not item_name:
        return jsonify({"status": "error", "message": "Item name required"}), 400
    item_info = shelf_locations.get(item_name)
    if not item_info:
        return jsonify({"status": "error", "message": "Item not found"}), 404
    return jsonify({
        "status": "success", "data": {
            "item": item_name, "location": {
                "x": item_info['x'], "y": item_info['y'], "z": item_info['z']
            }, "stock": item_info['stock'], "shelf": item_info['shelf_name']
        }
    })

@app.route('/send_to_robot', methods=['POST'])
def send_to_robot():
    if not request.is_json:
        return jsonify({"status": "error", "message": "Request must be JSON"}), 415
    item_or_shelf_id = request.json.get('item')
    if not item_or_shelf_id:
        return jsonify({"status": "error", "message": "Item or Shelf identifier required"}), 400

    target_info = None
    if item_or_shelf_id in shelf_locations: target_info = shelf_locations[item_or_shelf_id]
    elif item_or_shelf_id in shelf_name_to_info: target_info = shelf_name_to_info[item_or_shelf_id]
    elif item_or_shelf_id in shelf_id_to_info: target_info = shelf_id_to_info[item_or_shelf_id]
    
    if not target_info:
        return jsonify({"status": "error", "message": f"Target '{item_or_shelf_id}' not found for navigation."}), 404

    try:
        nav_to_shelf = NavToShelf()
        nav_to_shelf.send_goal(target_info['x'], target_info['y'], target_info['z'], 0.0, 0.0, 0.0, 1.0)
        rclpy.spin_once(nav_to_shelf)
        nav_to_shelf.destroy_node()
        
        display_shelf = target_info.get('shelf_name') or target_info.get('shelf', 'unknown shelf')
        message = f"Robot navigating to {target_info['item']} at shelf {display_shelf}" if 'item' in target_info and target_info['item'] != 'N/A' else f"Robot navigating to shelf {display_shelf}"
            
        return jsonify({"status": "success", "message": message})
    except Exception as e:
        app.logger.error(f"Navigation failed: {str(e)}")
        return jsonify({"status": "error", "message": f"Navigation failed: {str(e)}"}), 500
    

@app.route('/query', methods=['POST'])
def process_query():
    start_total_time = time.time()
    if not request.is_json: return jsonify({"status": "error", "message": "Request must be JSON"}), 415
    query = request.json.get('query')
    query_mode = request.json.get('mode', 'auto')
    app.logger.info(f"Received query: '{query}', Mode: '{query_mode}'")

    # --- FAILURE INJECTION POINTS ---
    if "cause internal server error" in query.lower():
        app.logger.error("Simulating an internal server error as requested.")
        raise ValueError("Simulated Internal Server Error: This is a test.")
    
    if "cause server timeout" in query.lower():
        app.logger.info("Simulating a server-side timeout as requested (sleeping for 60 seconds).")
        time.sleep(60) # Sleep longer than typical client timeout
        return jsonify({"status": "error", "message": "Simulated timeout completed, but client likely timed out first."}), 200 # This response likely won't be seen by client

    if not query: return jsonify({"status": "error", "message": "Query cannot be empty."}), 400

    result = None
    message = "I couldn't find an answer to that question."
    detection_method = "N/A"
    llm_extraction_time = 0.0
    rag_vector_search_time = 0.0
    llm_rag_synthesis_time = 0.0
    direct_lookup_time = 0.0
    traditional_search_time = 0.0

    # Convert pandas DataFrame to a list of dictionaries for easier LLM consumption
    # Limiting to first 10 items for the 'facts_prompt' in LLM-only mode
    inventory_data = shelf_df.to_dict(orient='records')
    
    # Mode: Direct Item Lookup Only or Auto (as a first attempt)
    if query_mode == 'auto' or query_mode == 'direct_item_lookup_only':
        start_direct_lookup_time = time.time()
        actual_item_info = None
        actual_item_key = next((item for item in shelf_locations.keys() if item.lower() == query.lower()), None)
        if actual_item_key: actual_item_info = shelf_locations.get(actual_item_key)
        if not actual_item_info:
            if query.lower().startswith("shelf ") and query in shelf_name_to_info: actual_item_info = shelf_name_to_info.get(query)
            elif len(query) == 1 and query.isalpha() and query.upper() in shelf_id_to_info: actual_item_info = shelf_id_to_info.get(query.upper())
        if actual_item_info:
            result = {
                'item': actual_item_info.get('item', 'N/A'), 'location': {'x': actual_item_info['x'], 'y': actual_item_info['y'], 'z': actual_item_info['z']},
                'stock': actual_item_info['stock'], 'shelf': actual_item_info['shelf_name'], 'shelf_id': actual_item_info['shelf_id'], 'safety_info': 'N/A'
            }
            message = f"Found {actual_item_key} on {actual_item_info['shelf_name']} with {actual_item_info['stock']} in stock." if actual_item_key else f"Identified {actual_item_info['shelf_name']} (ID: {actual_item_info['shelf_id']}) at X:{actual_item_info['x']:.3f}, Y:{actual_item_info['y']:.3f}, Z:{actual_item_info['z']:.3f}. Contains items like {actual_item_info.get('item', 'N/A')}."
            detection_method = 'Direct Lookup'
        direct_lookup_time = time.time() - start_direct_lookup_time

    # Mode: LLM Only (Context-Based Reasoning, then Entity Extraction as fallback) or Auto (if direct lookup fails)
    if not result and (query_mode == 'auto' or query_mode == 'llm_only'):
        start_llm_processing_time = time.time() # Start timer for this entire LLM phase
        
        try:
            # 1. Attempt LLM Context-Based Reasoning first
            facts_prompt = "\n".join([
                f"Item {row['Item']} is on {row['Shelf Name']} (ID: {row['Shelf ID']}) with {row['Stock']} units in stock."
                for row in inventory_data[:50] # Limiting to first 10 items for concise in-context learning
            ])
            
            
            # Use safety_manual_content directly for safety_info
            safety_info_for_llm = safety_manual_content

            # Invoke llm_answer_chain with the required inputs for its ChatPromptTemplate
            llm_answer = llm_answer_chain.invoke({
                "query": query,
                "facts_prompt": facts_prompt,
                "safety_info": safety_info_for_llm
            }).strip()

            print(llm_answer)
             
            # Check if LLM provided a meaningful answer
            if llm_answer and not ("cannot answer" in llm_answer.lower() or "not found" in llm_answer.lower() or "not mentioned" in llm_answer.lower()):
                # First try to parse structured data
                parsed_llm_answer_data = parse_rag_result(llm_answer)
                print(parsed_llm_answer_data)
                
                # Always create a result even if parsing fails, because the LLM gave a valid answer
                result = {
                    "llm_answer": llm_answer,
                    "item": parsed_llm_answer_data.get('item', 'N/A'), 
                    'location': parsed_llm_answer_data.get('location', {'x': 'N/A', 'y': 'N/A', 'z': 'N/A'}),
                    'stock': parsed_llm_answer_data.get('stock', 'N/A'), 
                    'shelf': parsed_llm_answer_data.get('shelf', 'N/A'), 
                    'shelf_id': parsed_llm_answer_data.get('shelf_id', 'N/A'), 
                    'safety_info': parsed_llm_answer_data.get('safety_info', 'N/A')
                }
                
                # Also try to extract Shelf A info directly if mentioned
                if 'shelf a' in llm_answer.lower() or 'shelf a' in llm_answer.lower():
                    if 'shelf A' in shelf_name_to_info:
                        shelf_info = shelf_name_to_info['shelf A']
                        result.update({
                            'item': 'Apple iPhone 14',  # From the response
                            'location': {'x': shelf_info['x'], 'y': shelf_info['y'], 'z': shelf_info['z']},
                            'stock': shelf_info['stock'],
                            'shelf': shelf_info['shelf_name'],
                            'shelf_id': shelf_info['shelf_id']
                        })

                message = f"LLM (Context-Based Reasoning) response: {llm_answer}"
                detection_method = "LLM Context-Based Reasoning"
                llm_extraction_time = time.time() - start_llm_processing_time # Time for this successful LLM call
            
            else:
                # 2. Fallback: LLM Entity Extraction
                start_entity_extraction_time = time.time()
                llm_extracted_entity = llm_shelf_extractor_chain.invoke(query).strip()
                llm_extraction_time = time.time() - start_entity_extraction_time
                
                info = None
                if llm_extracted_entity != 'NONE':
                    # Try multiple variations to match the shelf name
                    variations = [
                        llm_extracted_entity,
                        llm_extracted_entity.lower(),
                        llm_extracted_entity.capitalize(),
                        "shelf " + llm_extracted_entity.split()[-1] if "shelf" not in llm_extracted_entity.lower() else llm_extracted_entity
                    ]
                    
                    for var in variations:
                        if var in shelf_name_to_info:
                            info = shelf_name_to_info[var]
                            app.logger.info(f"DEBUG: Found match with variation '{var}'")
                            break
                    
                    # If still not found, try the original matching logic
                    if not info:
                        item_from_llm = next((item_name for item_name in shelf_locations.keys() if item_name.lower() == llm_extracted_entity.lower()), None)
                        if item_from_llm:
                            info = shelf_locations.get(item_from_llm)
                        elif llm_extracted_entity.lower().startswith("shelf ") and llm_extracted_entity in shelf_name_to_info:
                            info = shelf_name_to_info[llm_extracted_entity]
                        elif len(llm_extracted_entity) == 1 and llm_extracted_entity.isalpha() and llm_extracted_entity.upper() in shelf_id_to_info:
                            info = shelf_id_to_info[llm_extracted_entity.upper()]

                if info:
                    result = {
                        'item': info.get('item', 'N/A'),
                        'location': {'x': info['x'], 'y': info['y'], 'z': info['z']},
                        'stock': info.get('stock', 'N/A'),
                        'shelf': info.get('shelf_name', info.get('shelf')),
                        'shelf_id': info.get('shelf_id', 'N/A'),
                        'safety_info': 'N/A'
                    }
                    message = f"Identified {info['item']} on {info['shelf_name']} (ID: {info['shelf_id']}) with {info['stock']} in stock." \
                              if 'item' in info and info['item'] != 'N/A' and info['item'].lower() == llm_extracted_entity.lower() else \
                              f"Identified {info['shelf_name']} (ID: {info['shelf_id']}) at X:{info['x']:.3f}, Y:{info['y']:.3f}, Z:{info['z']:.3f}. Contains items like {info.get('item', 'N/A')}."
                    detection_method = 'LLM Entity Extraction (Fallback)'
                elif query_mode == 'llm_only':
                    message = "The LLM did not provide a direct answer from its context and also could not identify a specific item or shelf for direct lookup. Please rephrase your query for this mode."
                    detection_method = 'LLM Only (No Context/Entity)'
                    result = None
                else: # For 'auto' mode, if LLM returns NONE from extraction, it should try other methods
                    message = "The LLM did not identify a specific entity in your query. Trying other methods..."
                    detection_method = 'LLM Entity Extraction (Fallback - No Match)'

        except Exception as e:
            app.logger.error(f"LLM processing in 'llm_only' mode failed: {str(e)}")
            message = f"Error during LLM-only mode processing: {str(e)}"
            llm_extraction_time = time.time() - start_llm_processing_time

    if not result and (query_mode == 'auto' or query_mode == 'traditional_search_only'):
        start_traditional_search_time = time.time()
        traditional_context = traditional_keyword_search(query, traditional_search_corpus)
        traditional_search_time = time.time() - start_traditional_search_time
        if traditional_context:
            result = {
                'item': 'N/A', 'location': {'x': 'N/A', 'y': 'N/A', 'z': 'N/A'},
                'stock': 'N/A', 'shelf': 'N/A', 'shelf_id': 'N/A', 'safety_info': traditional_context
            }
            item_match = re.search(r"Item:\s*([^\n.]+)", traditional_context, re.IGNORECASE)
            shelf_name_match = re.search(r"Shelf Name:\s*([^\n.]+)", traditional_context, re.IGNORECASE)
            shelf_id_match = re.search(r"Shelf ID:\s*([^\n.]+)", traditional_context, re.IGNORECASE)
            stock_match = re.search(r"Stock:\s*(\d+)", traditional_context, re.IGNORECASE)
            if item_match: result['item'] = item_match.group(1).strip()
            if shelf_name_match: result['shelf'] = shelf_name_match.group(1).strip()
            if shelf_id_match: result['shelf_id'] = shelf_id_match.group(1).strip()
            if stock_match:
                try: result['stock'] = int(stock_match.group(1))
                except ValueError: pass
            message = f"Found relevant information using traditional keyword search:\n\n{traditional_context}"
            detection_method = 'Traditional Search'
        else: message = "Traditional keyword search did not find relevant exact matches."

    if not result and (query_mode == 'auto' or query_mode == 'rag_only'):
        app.logger.info("Attempting RAG process...")
        start_rag_time = time.time()
        context_text = None
        if vector_store: 
            try:
                query_embedding = get_embedding(query)
                # Ensure query_embedding is a 2D array (list of lists or numpy array)
                # FAISS expects the query embeddings to be a 2D array, even for a single query.
                if isinstance(query_embedding, np.ndarray) and query_embedding.ndim == 1:
                    query_embedding = query_embedding.reshape(1, -1)
                elif not isinstance(query_embedding, list): # handle case where it's a list from get_embedding but not 2D
                    query_embedding = [query_embedding] # Ensure it's a list of lists

                D, I = vector_store.index.search(np.array(query_embedding), k=3) 
                relevant_chunks_content = []
                # Ensure all_combined_texts is not empty before accessing by index
                if all_combined_texts:
                    for i_idx, dist in zip(I[0], D[0]):
                        if i_idx != -1 and dist < 100.0: relevant_chunks_content.append(all_combined_texts[i_idx])
                context_text = "\n\n".join(relevant_chunks_content) if relevant_chunks_content else None
                rag_vector_search_time = time.time() - start_rag_time
                if context_text:
                    start_llm_rag_synthesis_time = time.time()
                    llm_response_from_rag = rag_llm_chain.invoke({"query": query, "context": context_text}).strip()
                    llm_rag_synthesis_time = time.time() - start_llm_rag_synthesis_time
                    message = llm_response_from_rag
                    detection_method = 'RAG (Vector Search)'
                    parsed_rag_data = parse_rag_result(llm_response_from_rag)
                    result = {
                        'item': parsed_rag_data.get('item', 'N/A'), 'shelf': parsed_rag_data.get('shelf', 'N/A'), 'shelf_id': parsed_rag_data.get('shelf_id', 'N/A'),
                        'location': parsed_rag_data.get('location', {'x': 'N/A', 'y': 'N/A', 'z': 'N/A'}),
                        'stock': parsed_rag_data.get('stock', 'N/A'), 'safety_info': parsed_rag_data.get('safety_info', 'N/A')
                    }
                    if "i don't have enough information" in message.lower() or "i don't know" in message.lower() or "not mentioned in the provided context" in message.lower():
                        result = None 
                        message = "I couldn't find a precise answer to that question based on the available information."
                else: message = "RAG did not find relevant context for your query. Try rephrasing or a different query."
            except Exception as e:
                app.logger.error(f"RAG process failed: {str(e)}")
                message = f"An error occurred during RAG processing: {str(e)}"
        else: message = "Vector store is not initialized. Cannot perform RAG."

    total_time = time.time() - start_total_time
    response_data = {
        "status": "success" if result else "error", "data": result if result else {}, "message": message, "method": detection_method,
        "total_processing_time": total_time, "llm_time_without_rag": llm_extraction_time, "rag_time": rag_vector_search_time,
        "llm_rag_synthesis_time": llm_rag_synthesis_time, "direct_lookup_time": direct_lookup_time, "traditional_search_time": traditional_search_time
    }

    return jsonify(response_data), 200 if result else 404

@app.route('/process_order', methods=['POST'])
def process_order():
    global shelf_df # Declare global to modify the DataFrame
    if not request.is_json:
        return jsonify({"status": "error", "message": "Request must be JSON"}), 415

    order_items_raw: List[Dict[str, Any]] = request.json.get('order_items', [])
    if not order_items_raw:
        return jsonify({"status": "error", "message": "No items in the order."}), 400

    robot_log: List[str] = []
    shelves_to_visit: Dict[str, Dict[str, Any]] = {} # To store unique shelf info for navigation
    
    # --- Server-side Stock Validation and Item Grouping by Shelf ---
    items_to_update_stock = [] # Store (item_name, quantity, shelf_name) for stock decrement
    
    for item_in_order in order_items_raw:
        item_name = item_in_order.get('itemName')
        requested_quantity = item_in_order.get('quantity')

        if not item_name or not requested_quantity:
            robot_log.append(f"Skipping malformed order item: {item_in_order}")
            continue

        item_info_from_df = shelf_df[shelf_df['Item'] == item_name]
        if item_info_from_df.empty: # Check if item exists in DataFrame
            robot_log.append(f"Error: Item '{item_name}' not found in inventory. Cannot process order.")
            return jsonify({"status": "error", "message": f"Item '{item_name}' not found.", "log": robot_log}), 404
        
        # Access the stock from the first matching row (assuming unique item names)
        available_stock = item_info_from_df.iloc[0]['Stock']

        if requested_quantity > available_stock:
            robot_log.append(f"Error: Not enough stock for '{item_name}'. Requested: {requested_quantity}, Available: {available_stock}. Item skipped.")
            return jsonify({"status": "error", "message": f"Not enough stock for '{item_name}'. Available: {available_stock}.", "log": robot_log}), 400
        
        items_to_update_stock.append({
            'item_name': item_name,
            'quantity': requested_quantity,
            'shelf_name': item_info_from_df.iloc[0]['Shelf Name'] # Capture shelf name for consistent lookup
        })

        shelf_id = item_info_from_df.iloc[0]['Shelf ID']
        if shelf_id not in shelves_to_visit:
            shelves_to_visit[shelf_id] = {
                'x': item_info_from_df.iloc[0]['X Position'],
                'y': item_info_from_df.iloc[0]['Y Position'],
                'z': item_info_from_df.iloc[0]['Z Position'],
                'shelf_name': item_info_from_df.iloc[0]['Shelf Name'],
                'items_on_shelf': []
            }
        shelves_to_visit[shelf_id]['items_on_shelf'].append(f"{item_name} (Qty: {requested_quantity})")

    if not shelves_to_visit:
        return jsonify({"status": "error", "message": "No valid items to fulfill in the order.", "log": robot_log}), 400

    # Step 2: Decrement stock in DataFrame
    # Note: This loop assumes unique item names in the order and `shelf_df`.
    # If an item can be on multiple shelves, this logic would need refinement.
    for item_data in items_to_update_stock:
        item_name = item_data['item_name']
        quantity = item_data['quantity']
        
        # Find the row index for the item
        idx = shelf_df.index[shelf_df['Item'] == item_name].tolist()
        if idx:
            # Update the stock in the DataFrame
            shelf_df.loc[idx[0], 'Stock'] -= quantity
            app.logger.info(f"Decremented stock for {item_name} by {quantity}. New stock: {shelf_df.loc[idx[0], 'Stock']}")
    
    # Save updated DataFrame back to CSV
    # Use the globally defined script_dir for csv_path
    csv_path = os.path.join(script_dir, 'shelfinfo.csv')
    try:
        shelf_df.to_csv(csv_path, index=False)
        app.logger.info("Updated stock saved to shelfinfo.csv")
        # Reload all data structures to reflect new stock levels immediately after saving
        _load_shelf_data() 
    except Exception as e:
        app.logger.error(f"Error saving updated stock to CSV: {e}")
        robot_log.append(f"Critical error saving stock: {str(e)}")
        return jsonify({"status": "error", "message": "Failed to update stock permanently.", "log": robot_log}), 500

    # Step 3: Dispatch robot to each unique shelf
    try:
        nav_to_shelf = NavToShelf()
    
    # --- Add this log line for clarity ---
        robot_log.append("Order placed! Robot starting navigation to shelves.")
        app.logger.info("Order placed! Robot starting navigation sequence.")

        # Sort shelves to visit (e.g., by Shelf ID for consistency)
        sorted_shelves = sorted(shelves_to_visit.items(), key=lambda item: item[0])
        robot_log.append(f"Total unique shelves to visit: {len(sorted_shelves)}") # This is now safe to use

        for shelf_id, info in sorted_shelves:
            robot_log.append(f"Navigating to {info['shelf_name']} (ID: {shelf_id}) at X:{info['x']:.3f}, Y:{info['y']:.3f}, Z:{info['z']:.3f} to pick up: {', '.join(info['items_on_shelf'])}.")
            app.logger.info(f"Robot: Moving to Shelf {shelf_id} ({info['shelf_name']})")
            
            # Wrap send_goal in try-except to catch navigation-specific errors
            try:
                nav_to_shelf.send_goal(info['x'], info['y'], info['z'], 0.0, 0.0, 0.0, 1.0)
                rclpy.spin_once(nav_to_shelf)
            except AttributeError as e:
                error_msg = f"Navigation failed at Shelf {shelf_id} ({info['shelf_name']}): {e}"
                app.logger.error(error_msg)
                robot_log.append(f"CRITICAL ERROR during navigation: {str(e)}")
                return jsonify({"status": "error", "message": f"Order fulfillment failed during navigation: {str(e)}", "log": robot_log}), 500

            robot_log.append(f"Arrived at {info['shelf_name']}. Staying for 5 seconds for item pickup.")
            time.sleep(5)

            robot_log.append(f"Items picked from {info['shelf_name']}.")

        nav_to_shelf.destroy_node()
        robot_log.append("All items picked. Robot returning to base.")
        return jsonify({"status": "success", "message": "Order fulfillment completed!", "log": robot_log}), 200

    except Exception as e:
        app.logger.error(f"Error during robot navigation: {e}")
        robot_log.append(f"CRITICAL ERROR during robot navigation: {str(e)}")
        return jsonify({"status": "error", "message": f"Order fulfillment failed during navigation: {str(e)}", "log": robot_log}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)