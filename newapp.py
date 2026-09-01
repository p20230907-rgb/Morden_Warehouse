#!/usr/bin/env python3
"""
newapp.py – Warehouse assistant with Together AI, confidence‑aware routing,
enhanced safety, and evaluation support.
Uses the new router.py for routing decisions.
"""

import os
import sys
import re
import time
import logging
import json
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
import numpy as np
import rclpy
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv

# LangChain imports – Together AI
from langchain_together import ChatTogether
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Modular imports
from data_loader import load_data
from config import CONFIG
from router import Router                     # new unified router
from retriever import HybridRetriever
from safety import SafetyChecker
from robot_controller import RobotController
from logger import QueryLogger

# ======================================================================
# Flask app setup
# ======================================================================
app = Flask(__name__)
CORS(app)
app.logger.setLevel(logging.INFO)

# ======================================================================
# ROS2 and environment
# ======================================================================
if not rclpy.ok():
    rclpy.init()
ros_node = rclpy.create_node('warehouse_assistant_node')

load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    raise ValueError("TOGETHER_API_KEY not found in .env file.")

# ======================================================================
# Together AI model list and fallback
# ======================================================================
TOGETHER_MODELS = [
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
]
OVERRIDE_MODEL = os.getenv("TOGETHER_MODEL", None)

def get_llm(model_name=None, temperature=0):
    if OVERRIDE_MODEL:
        app.logger.info(f"Using override model: {OVERRIDE_MODEL}")
        model_name = OVERRIDE_MODEL
    if model_name is None:
        models_to_try = TOGETHER_MODELS
    else:
        models_to_try = [model_name] + [m for m in TOGETHER_MODELS if m != model_name]

    last_exception = None
    for model in models_to_try:
        try:
            app.logger.info(f"Attempting to load model: {model}")
            llm = ChatTogether(
                together_api_key=TOGETHER_API_KEY,
                model=model,
                temperature=temperature,
            )
            app.logger.info(f"✅ Successfully loaded model: {model}")
            return llm
        except Exception as e:
            app.logger.warning(f"Failed to load {model}: {e}")
            last_exception = e
            continue
    raise RuntimeError(f"All models failed. Last error: {last_exception}")

llm = get_llm()
parser = StrOutputParser()
app.logger.info(f"🚀 Using Together AI model: {llm.model_name}")

# ======================================================================
# Load data using the loader
# ======================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
data = load_data(script_dir, CONFIG)

shelf_df = data['shelf_df']
shelf_locations = data['shelf_locations']
shelf_name_to_info = data['shelf_name_to_info']
shelf_id_to_info = data['shelf_id_to_info']
all_items_for_order_page = data['all_items_for_order_page']
traditional_search_corpus = data['traditional_search_corpus']
safety_manual_content = data['safety_manual_content']
safety_chunks_docs = data['safety_chunks_docs']
vector_store = data['vector_store']
all_combined_texts = data['all_combined_texts']

# ======================================================================
# LLM chains
# ======================================================================
llm_shelf_extractor_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a warehouse assistant. Extract a shelf ID (e.g., 'A') or full shelf name (e.g., 'Shelf A') from the query. If none, respond 'NONE'."),
    ("human", "{query}")
])
llm_shelf_extractor_chain = {"query": RunnablePassthrough()} | llm_shelf_extractor_prompt | llm | parser

rag_llm_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful warehouse robot assistant. Use ONLY the provided context to answer. "
     "If the answer requires arithmetic (sum, count, average, comparison, etc.), perform the calculation "
     "using the numbers in the context. If the answer is not in the context, say so. "
     "Be precise and show the calculation if needed."),
    ("human", "Context: {context}\n\nQuestion: {query}")
])
rag_llm_chain = rag_llm_prompt | llm | parser

llm_answer_template_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a warehouse assistant. Use the facts and safety manual to answer.\n"
     "--- Inventory ---\n{facts_prompt}\n\n"
     "--- Safety Manual ---\n{safety_info}\n\n"
     "If you don't know, say so."
    ),
    ("human", "{query}")
])
llm_answer_chain = llm_answer_template_prompt | llm | parser

# ======================================================================
# Embedding model (for RAG)
# ======================================================================
embeddings = SentenceTransformer(CONFIG["embedding_model"])
embedding_cache = {}

def get_embedding(text: str) -> np.ndarray:
    if text not in embedding_cache:
        embedding_cache[text] = embeddings.encode(text)
    return embedding_cache[text]

# ======================================================================
# Helper functions
# ======================================================================
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
            if len(coords) == 3:
                parsed_data['location'] = {'x': coords[0], 'y': coords[1], 'z': coords[2]}
        except ValueError:
            app.logger.warning(f"Could not parse location coordinates from: {location_match.group(1)}")
    if stock_match:
        try:
            parsed_data['stock'] = int(stock_match.group(1))
        except ValueError:
            app.logger.warning(f"Could not parse stock quantity from: {stock_match.group(1)}")

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
    if not query_words:
        return None
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
    if max_matches > 0:
        return best_match_content
    return None

# ======================================================================
# Router instantiation
# ======================================================================
SAFETY_KEYWORDS = [
    "safety", "ppe", "fire", "hazardous", "loto", "manual",
    "procedure", "report", "training", "emergency", "exit",
    "first aid", "evacuation", "incident", "injury",
    "spill", "containment", "chemical", "hazard"
]

router = Router(
    mode=CONFIG.get("router_mode", "rule_based"),
    training_data_path=CONFIG.get("router_training_data", "router_training.csv"),
    model_path=CONFIG.get("router_model_path", "router_classifier_advanced.pkl"),
    confidence_threshold=CONFIG.get("router_confidence_threshold", 0.6),
    fallback_threshold=CONFIG.get("fallback_threshold", 0.3)
)

# ======================================================================
# Safety checker (placeholder)
# ======================================================================
class EnhancedSafetyChecker(SafetyChecker):
    def __init__(self, node, costmap_service='/global_costmap/get_costmap',
                 threshold=50, safety_buffer=0.5):
        super().__init__(node, costmap_service, threshold)
        self.safety_buffer = safety_buffer

    def is_goal_safe(self, x: float, y: float) -> Tuple[bool, str, str]:
        # Placeholder – implement real checks if costmap is available
        return True, "Safe (placeholder)", "low"

safety_checker = EnhancedSafetyChecker(ros_node, CONFIG["safety_costmap_service"],
                                       CONFIG["costmap_threshold"])

robot_controller = RobotController(ros_node, CONFIG["navigate_action"],
                                   CONFIG["goal_timeout_sec"], CONFIG["max_retries"])

query_logger = QueryLogger(CONFIG["log_file"])

# ======================================================================
# Flask Routes
# ======================================================================
@app.route('/')
def index():
    total_items = len(shelf_df)
    total_units = shelf_df['Stock'].sum() if 'Stock' in shelf_df.columns else 0
    total_value = (shelf_df['Stock'] * shelf_df['Value']).sum() if 'Value' in shelf_df.columns else 0
    low_stock_items = len(shelf_df[shelf_df['Stock'] < 5]) if 'Stock' in shelf_df.columns else 0
    categories = shelf_df['Category'].nunique() if 'Category' in shelf_df.columns else 1
    return render_template('index.html',
                           shelves=shelf_df.to_dict('records'),
                           total_items=total_items,
                           total_units=total_units,
                           total_value=total_value,
                           low_stock_items=low_stock_items,
                           categories=categories,
                           shelf_df=shelf_df)

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
    return render_template('order_management.html', all_items=all_items_for_order_page)

@app.route('/performance_dashboard')
def performance_dashboard():
    return render_template('performance_dashboard.html')

@app.route('/deep_thinking_assistant')
def deep_thinking_assistant():
    return render_template('deep_thinking_assistant.html')

@app.route('/llm_only_assistant')
def llm_only_assistant():
    return render_template('llm_only_assistant.html')

# ======================================================================
# Deep Query endpoint (unchanged)
# ======================================================================
@app.route('/deep_query', methods=['POST'])
def deep_query():
    if not request.is_json:
        return jsonify({"status": "error", "message": "Request must be JSON"}), 415
    user_query = request.json.get('query')
    if not user_query:
        return jsonify({"status": "error", "message": "Query cannot be empty."}), 400

    app.logger.info(f"Received deep query: '{user_query}'")

    try:
        inventory_data_for_llm = shelf_df.to_dict(orient='records')
        inventory_context_str = "Warehouse Inventory Data (Item, Shelf, Stock, Location):\n"
        for item_record in inventory_data_for_llm:
            inventory_context_str += (
                f"- Item: {item_record.get('Item', 'N/A')}, "
                f"Shelf: {item_record.get('Shelf Name', 'N/A')} (ID: {item_record.get('Shelf ID', 'N/A')}), "
                f"Stock: {item_record.get('Stock', 'N/A')}, "
                f"Location: (X:{item_record.get('X Position', 'N/A')}, Y:{item_record.get('Y Position', 'N/A')}, Z:{item_record.get('Z Position', 'N/A')})\n"
            )
        full_context_for_llm = (
            f"Current Date: {time.strftime('%Y-%m-%d')}\n\n"
            f"{inventory_context_str}\n\n"
            f"Warehouse Safety Manual:\n{safety_manual_content}\n\n"
        )

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
             "but be prepared to combine information if a query spans both.\n\n"
             "COMPREHENSIVE CONTEXT:\n{context}"
            ),
            ("human", "User Query: {query}")
        ])

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
        app.logger.info(f"Deep Query LLM Response: {llm_response[:500]}...")

        return jsonify({"status": "success", "answer": llm_response}), 200

    except Exception as e:
        app.logger.error(f"Error during deep query processing: {str(e)}")
        return jsonify({"status": "error", "message": f"Failed to process deep query: {str(e)}"}), 500

# ======================================================================
# Get Shelf Location
# ======================================================================
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
            "item": item_name,
            "location": {
                "x": item_info['x'],
                "y": item_info['y'],
                "z": item_info['z']
            },
            "stock": item_info['stock'],
            "shelf": item_info['shelf_name']
        }
    })

# ======================================================================
# NavToShelf import
# ======================================================================
try:
    from warehouse_sim.navigate_to_shelf import NavToShelf
except ImportError:
    app.logger.warning("Could not import NavToShelf. Using mock.")
    class NavToShelf:
        def __init__(self):
            app.logger.info("NavToShelf (mock) initialized.")
        def send_goal(self, x, y, z, ox, oy, oz, ow):
            app.logger.info(f"Mock goal: ({x}, {y}, {z})")
            time.sleep(1)
        def destroy_node(self):
            app.logger.info("Mock node destroyed.")

@app.route('/send_to_robot', methods=['POST'])
def send_to_robot():
    if not request.is_json:
        return jsonify({"status": "error", "message": "Request must be JSON"}), 415
    item_or_shelf_id = request.json.get('item')
    if not item_or_shelf_id:
        return jsonify({"status": "error", "message": "Item or Shelf identifier required"}), 400

    target_info = None
    if item_or_shelf_id in shelf_locations:
        target_info = shelf_locations[item_or_shelf_id]
    elif item_or_shelf_id in shelf_name_to_info:
        target_info = shelf_name_to_info[item_or_shelf_id]
    elif item_or_shelf_id in shelf_id_to_info:
        target_info = shelf_id_to_info[item_or_shelf_id]

    if not target_info:
        return jsonify({"status": "error", "message": f"Target '{item_or_shelf_id}' not found."}), 404

    safe, reason, risk = safety_checker.is_goal_safe(float(target_info['x']), float(target_info['x']))
    if not safe:
        app.logger.warning(f"Safety block: {reason} (risk: {risk})")
        return jsonify({"status": "error", "message": f"Goal unsafe: {reason}"}), 400
    
    try:
        nav = NavToShelf()
        x = float(target_info['x'])
        print("value ofx,", x)
        y = float(target_info['y'])
        z = float(target_info['z'])
        success = robot_controller.send_goal(x, y, z, orientation=(0.0, 0.0, 0.0, 1.0))
        #nav.send_goal(float(target_info['x']), float(target_info['y']), float(target_info['z']), 0,0,0,1)
        if hasattr(nav, 'subscriptions'):
            rclpy.spin_once(nav)
        nav.destroy_node()
        return jsonify({"status": "success", "message": f"Navigating to {item_or_shelf_id}"})
    except Exception as e:
        app.logger.error(f"Navigation failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# ======================================================================
# MAIN QUERY ENDPOINT (with all logic intact)
# ======================================================================
@app.route('/query', methods=['POST'])
def process_query():
    start_total_time = time.time()
    if not request.is_json:
        return jsonify({"status": "error", "message": "Request must be JSON"}), 415
    query = request.json.get('query')
    query_mode = request.json.get('mode', 'auto')
    app.logger.info(f"Received query: '{query}', Mode: '{query_mode}'")

    if "cause internal server error" in query.lower():
        raise ValueError("Simulated error")
    if "cause server timeout" in query.lower():
        time.sleep(60)
        return jsonify({"status": "error", "message": "Timeout simulated"}), 200
    if not query:
        return jsonify({"status": "error", "message": "Empty query"}), 400

    result = None
    message = "I couldn't find an answer."
    detection_method = "N/A"

    # --- Determine if special deterministic tiers are allowed ---
    use_special_tiers = (query_mode == "auto")

    # ==================================================================
    # AGGREGATION TIER – "how many X", "total X", etc. (only in auto mode)
    # ==================================================================
    if use_special_tiers:
        agg_patterns = [r"how many (\w+)", r"total (\w+)", r"count of (\w+)", r"sum of (\w+)"]
        agg_match = None
        for pat in agg_patterns:
            m = re.search(pat, query.lower())
            if m:
                agg_match = m.group(1)
                break

        if agg_match:
            category_map = {
                "phone": "Smartphones", "phones": "Smartphones",
                "laptop": "Laptops", "laptops": "Laptops",
                "headphone": "Headphones", "headphones": "Headphones",
                "accessory": "Accessories", "accessories": "Accessories",
                "gaming": "Gaming Consoles", "controller": "Controllers",
                "mouse": "Computer Mouse", "keyboard": "Keyboards",
            }
            cat = category_map.get(agg_match.lower())
            if cat:
                filtered = shelf_df[shelf_df['Category'].str.contains(cat, case=False, na=False)]
                total = filtered['Stock'].sum()
                if total > 0:
                    answer = f"We have {int(total)} {agg_match} in stock."
                else:
                    answer = f"No {agg_match} found in inventory."
                return jsonify({
                    "status": "success",
                    "message": answer,
                    "tier": "aggregation",
                    "confidence": 1.0,
                    "total_processing_time": time.time() - start_total_time
                })

    # ==================================================================
    # PRICE-FILTER TIER – "items with price > 500", etc. (only in auto mode)
    # ==================================================================
    if use_special_tiers:
        price_patterns = [
            r"list items? (?:with|where|whose) (?:price|value) (>|<|>=|<=|=|==) (\d+)",
            r"items? (?:with|where|whose) (?:price|value) (>|<|>=|<=|=|==) (\d+)",
            r"show (?:items?|products?) (?:with|where|whose) (?:price|value) (>|<|>=|<=|=|==) (\d+)",
            r"(?:price|value|cost)\s*(>|<|>=|<=|=)\s*\$?(\d+)",
            r"(?:more|less) than\s*\$?(\d+)",
        ]
        price_match = None
        for pat in price_patterns:
            m = re.search(pat, query.lower())
            if m:
                if len(m.groups()) == 2:
                    operator = m.group(1)
                    threshold = float(m.group(2))
                elif len(m.groups()) == 1:
                    if "more than" in query.lower():
                        operator = '>'
                        threshold = float(m.group(1))
                    elif "less than" in query.lower():
                        operator = '<'
                        threshold = float(m.group(1))
                    else:
                        continue
                else:
                    continue
                price_match = (operator, threshold)
                break

        if price_match:
            operator, threshold = price_match
            if operator == '>':
                filtered = shelf_df[shelf_df['Value'] > threshold]
            elif operator == '<':
                filtered = shelf_df[shelf_df['Value'] < threshold]
            elif operator == '>=':
                filtered = shelf_df[shelf_df['Value'] >= threshold]
            elif operator == '<=':
                filtered = shelf_df[shelf_df['Value'] <= threshold]
            elif operator in ['=', '==']:
                filtered = shelf_df[shelf_df['Value'] == threshold]
            else:
                filtered = shelf_df

            if not filtered.empty:
                items_list = filtered[['Item', 'Shelf Name', 'Value']].to_dict('records')
                lines = [f"- {item['Item']} (Shelf {item['Shelf Name']}) – ${item['Value']:.2f}" for item in items_list]
                answer = f"Items with price {operator} {threshold}:\n" + "\n".join(lines)
                answer += f"\n\nTotal items: {len(items_list)}"
            else:
                answer = f"No items found with price {operator} {threshold}."

            return jsonify({
                "status": "success",
                "message": answer,
                "tier": "price_filter",
                "confidence": 1.0,
                "total_processing_time": time.time() - start_total_time
            })

    # ==================================================================
    # ROUTING DECISION (using the new Router class)
    # ==================================================================
    if query_mode == "llm_only":
        tier = "llm"
        confidence = 1.0
        detection_method = "Forced LLM-only"
    elif query_mode == "rag_only":
        tier = "rag"
        confidence = 1.0
        detection_method = "Forced RAG-only"
    elif query_mode == "traditional_search_only":
        tier = "keyword"
        confidence = 1.0
        detection_method = "Forced Keyword-only"
    elif query_mode == "direct_only":
        tier = "direct"
        confidence = 1.0
    elif query_mode == "max_confidence":
        # Use the router's predict (it will handle max confidence if we add that method)
        tier, confidence = router.predict(query, shelf_locations, SAFETY_KEYWORDS)
        detection_method = "Max Confidence Router"
    else:
        tier, confidence = router.predict(query, shelf_locations, SAFETY_KEYWORDS)
        detection_method = f"Router ({router.mode})"

    app.logger.info(f"Routing decision: {tier} (conf: {confidence:.3f})")

    # ==================================================================
    # EXECUTE CHOSEN TIER
    # ==================================================================
    try:
        if tier == "direct":
            detection_method = "Direct Lookup"
            how_many_match = re.match(r"how many (.+)", query.lower())
            if how_many_match:
                item_candidate = how_many_match.group(1).strip()
                actual_item_key = next((item for item in shelf_locations.keys() if item.lower() == item_candidate), None)
            else:
                actual_item_key = next((item for item in shelf_locations.keys() if item.lower() == query.lower()), None)
            if actual_item_key:
                info = shelf_locations[actual_item_key]
                result = {
                    'item': actual_item_key,
                    'location': {'x': info['x'], 'y': info['y'], 'z': info['z']},
                    'stock': info['stock'],
                    'shelf': info['shelf_name'],
                    'shelf_id': info['shelf_id'],
                    'safety_info': 'N/A'
                }
                message = f"Found {actual_item_key} on {info['shelf_name']} with {info['stock']} in stock."
            else:
                message = "Item not found in direct lookup."

        elif tier == "keyword":
            detection_method = "Keyword Search"
            context = traditional_keyword_search(query, traditional_search_corpus)
            if context:
                result = {
                    'item': 'N/A', 'location': {'x': 'N/A', 'y': 'N/A', 'z': 'N/A'},
                    'stock': 'N/A', 'shelf': 'N/A', 'shelf_id': 'N/A', 'safety_info': context
                }
                message = f"Found relevant information using keyword search:\n{context}"
            else:
                message = "No keyword match found."

        elif tier == "rag":
            detection_method = "RAG"
            if vector_store:
                try:
                    query_embedding = get_embedding(query)
                    if isinstance(query_embedding, np.ndarray) and query_embedding.ndim == 1:
                        query_embedding = query_embedding.reshape(1, -1)
                    D, I = vector_store.index.search(np.array(query_embedding), k=10)
                    relevant_chunks = []
                    if all_combined_texts:
                        for i_idx in I[0]:
                            if i_idx != -1:
                                relevant_chunks.append(all_combined_texts[i_idx])
                    context = "\n\n".join(relevant_chunks)
                    if context:
                        llm_response = rag_llm_chain.invoke({"query": query, "context": context}).strip()
                        parsed = parse_rag_result(llm_response)
                        result = {
                            'item': parsed.get('item', 'N/A'),
                            'location': parsed.get('location', {'x': 'N/A', 'y': 'N/A', 'z': 'N/A'}),
                            'stock': parsed.get('stock', 'N/A'),
                            'shelf': parsed.get('shelf', 'N/A'),
                            'shelf_id': parsed.get('shelf_id', 'N/A'),
                            'safety_info': parsed.get('safety_info', 'N/A')
                        }
                        message = llm_response
                    else:
                        message = "No relevant context found."
                except Exception as e:
                    app.logger.warning(f"RAG failed: {e}. Falling back to LLM.")
                    detection_method = "LLM (fallback from RAG)"
                    inventory_lines = []
                    for _, row in shelf_df.iterrows():
                        inventory_lines.append(
                            f"Item: {row['Item']}, Shelf: {row['Shelf Name']} (ID: {row['Shelf ID']}), "
                            f"Stock: {row['Stock']}, Price: ${row['Value']:.2f}, Category: {row['Category']}"
                        )
                    facts_prompt = "\n".join(inventory_lines)
                    safety_info = safety_manual_content[:2000]
                    llm_answer = llm_answer_chain.invoke({
                        "query": query,
                        "facts_prompt": facts_prompt,
                        "safety_info": safety_info
                    }).strip()
                    parsed = parse_rag_result(llm_answer)
                    result = {
                        'llm_answer': llm_answer,
                        'item': parsed.get('item', 'N/A'),
                        'location': parsed.get('location', {'x': 'N/A', 'y': 'N/A', 'z': 'N/A'}),
                        'stock': parsed.get('stock', 'N/A'),
                        'shelf': parsed.get('shelf', 'N/A'),
                        'shelf_id': parsed.get('shelf_id', 'N/A'),
                        'safety_info': parsed.get('safety_info', 'N/A')
                    }
                    message = f"LLM fallback response: {llm_answer}"
            else:
                message = "Vector store not available."

        elif tier == "llm":
            detection_method = "LLM Reasoning"
            inventory_lines = []
            for _, row in shelf_df.iterrows():
                inventory_lines.append(
                    f"Item: {row['Item']}, Shelf: {row['Shelf Name']} (ID: {row['Shelf ID']}), "
                    f"Stock: {row['Stock']}, Price: ${row['Value']:.2f}, Category: {row['Category']}"
                )
            facts_prompt = "\n".join(inventory_lines)
            safety_info = safety_manual_content[:2000]
            llm_answer = llm_answer_chain.invoke({
                "query": query,
                "facts_prompt": facts_prompt,
                "safety_info": safety_info
            }).strip()
            parsed = parse_rag_result(llm_answer)
            result = {
                'llm_answer': llm_answer,
                'item': parsed.get('item', 'N/A'),
                'location': parsed.get('location', {'x': 'N/A', 'y': 'N/A', 'z': 'N/A'}),
                'stock': parsed.get('stock', 'N/A'),
                'shelf': parsed.get('shelf', 'N/A'),
                'shelf_id': parsed.get('shelf_id', 'N/A'),
                'safety_info': parsed.get('safety_info', 'N/A')
            }
            message = f"LLM response: {llm_answer}"

        else:
            message = "I'm not sure what you're asking. Could you rephrase?"
            result = None

    except Exception as e:
        app.logger.error(f"Query execution failed: {e}")
        return jsonify({"status": "error", "message": f"Processing error: {e}"}), 500

    total_time = time.time() - start_total_time
    query_logger.log(query, tier, tier, message, total_time, "")

    return jsonify({
        "status": "success" if result else "error",
        "data": result if result else {},
        "message": message,
        "method": detection_method,
        "total_processing_time": total_time,
        "confidence": confidence,
        "tier": tier
    }), 200 if result else 404

# ======================================================================
# Process Order (unchanged)
# ======================================================================
@app.route('/process_order', methods=['POST'])
def process_order():
    global shelf_df
    if not request.is_json:
        return jsonify({"status": "error", "message": "Request must be JSON"}), 415

    order_items_raw: List[Dict[str, Any]] = request.json.get('order_items', [])
    if not order_items_raw:
        return jsonify({"status": "error", "message": "No items in the order."}), 400

    robot_log: List[str] = []
    shelves_to_visit: Dict[str, Dict[str, Any]] = {}
    items_to_update_stock = []

    for item_in_order in order_items_raw:
        item_name = item_in_order.get('itemName')
        requested_quantity = item_in_order.get('quantity')

        if not item_name or not requested_quantity:
            robot_log.append(f"Skipping malformed order item: {item_in_order}")
            continue

        item_info_from_df = shelf_df[shelf_df['Item'] == item_name]
        if item_info_from_df.empty:
            robot_log.append(f"Error: Item '{item_name}' not found in inventory.")
            return jsonify({"status": "error", "message": f"Item '{item_name}' not found.", "log": robot_log}), 404

        available_stock = item_info_from_df.iloc[0]['Stock']
        if requested_quantity > available_stock:
            robot_log.append(f"Error: Not enough stock for '{item_name}'. Requested: {requested_quantity}, Available: {available_stock}.")
            return jsonify({"status": "error", "message": f"Not enough stock for '{item_name}'. Available: {available_stock}.", "log": robot_log}), 400

        items_to_update_stock.append({
            'item_name': item_name,
            'quantity': requested_quantity,
            'shelf_name': item_info_from_df.iloc[0]['Shelf Name']
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
        return jsonify({"status": "error", "message": "No valid items to fulfill.", "log": robot_log}), 400

    for item_data in items_to_update_stock:
        item_name = item_data['item_name']
        quantity = item_data['quantity']
        idx = shelf_df.index[shelf_df['Item'] == item_name].tolist()
        if idx:
            shelf_df.loc[idx[0], 'Stock'] -= quantity
            app.logger.info(f"Decremented stock for {item_name} by {quantity}. New stock: {shelf_df.loc[idx[0], 'Stock']}")

    csv_path = os.path.join(script_dir, 'shelfinfo.csv')
    try:
        shelf_df.to_csv(csv_path, index=False)
        app.logger.info("Updated stock saved to shelfinfo.csv")
    except Exception as e:
        app.logger.error(f"Error saving updated stock to CSV: {e}")
        robot_log.append(f"Critical error saving stock: {str(e)}")
        return jsonify({"status": "error", "message": "Failed to update stock permanently.", "log": robot_log}), 500

    try:
        nav_to_shelf = NavToShelf()
        robot_log.append("Order placed! Robot starting navigation.")
        app.logger.info("Order placed! Robot starting navigation sequence.")

        sorted_shelves = sorted(shelves_to_visit.items(), key=lambda item: item[0])
        robot_log.append(f"Total unique shelves to visit: {len(sorted_shelves)}")

        for shelf_id, info in sorted_shelves:
            robot_log.append(f"Navigating to {info['shelf_name']} (ID: {shelf_id}) to pick up: {', '.join(info['items_on_shelf'])}.")
            app.logger.info(f"Robot: Moving to Shelf {shelf_id} ({info['shelf_name']})")
            try:
                nav_to_shelf.send_goal(info['x'], info['y'], info['z'], 0.0, 0.0, 0.0, 1.0)
                if hasattr(nav_to_shelf, 'subscriptions'):
                    rclpy.spin_once(nav_to_shelf)
            except AttributeError as e:
                error_msg = f"Navigation failed at Shelf {shelf_id}: {e}"
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

# ======================================================================
# Run the app
# ======================================================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)