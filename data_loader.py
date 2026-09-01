# data_loader.py
import os
import re
import pandas as pd
import logging
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

# Custom embedding wrapper
class SentenceTransformerEmbeddings:
    def __init__(self, model):
        self.model = model
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    def embed_query(self, query):
        return self.model.encode([query])[0].tolist()

def load_data(script_dir: str, config: dict):
    """
    Load CSV, safety manual, build dictionaries, vector store, etc.
    Returns a dict with all data structures.
    """
    logger = logging.getLogger(__name__)
    
    # 1. Load CSV
    csv_path = os.path.join(script_dir, 'shelfinfo.csv')
    try:
        shelf_df = pd.read_csv(csv_path)
        logger.info("Shelf data loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        shelf_df = pd.DataFrame(columns=['Shelf ID','Shelf Name','X Position','Y Position','Z Position','Item','Stock','Category','Value'])

    # 2. Build lookup dictionaries
    shelf_locations = {}
    shelf_name_to_info = {}
    shelf_id_to_info = {}
    for _, row in shelf_df.iterrows():
        shelf_locations[row['Item']] = {
            'x': float(row['X Position']), 'y': float(row['Y Position']), 'z': float(row['Z Position']),
            'stock': row['Stock'], 'shelf_name': row['Shelf Name'], 'shelf_id': row['Shelf ID']
        }
        shelf_name_to_info[row['Shelf Name']] = {
            'shelf_id': row['Shelf ID'], 'x': row['X Position'], 'y': row['Y Position'],
            'z': row['Z Position'], 'item': row['Item'], 'stock': row['Stock']
        }
        shelf_id_to_info[row['Shelf ID']] = {
            'shelf_name': row['Shelf Name'], 'x': row['X Position'], 'y': row['Y Position'],
            'z': row['Z Position'], 'item': row['Item'], 'stock': row['Stock']
        }

    # 3. Unique items for order page
    all_items_for_order_page = []
    seen = set()
    for _, row in shelf_df.iterrows():
        if row['Item'] not in seen:
            all_items_for_order_page.append({
                'item_name': row['Item'],
                'shelf_name': row['Shelf Name'],
                'shelf_id': row['Shelf ID'],
                'x_position': row['X Position'],
                'y_position': row['Y Position'],
                'z_position': row['Z Position'],
                'stock': row['Stock']
            })
            seen.add(row['Item'])

    # 4. Safety manual
    safety_file = os.path.join(script_dir, 'data', 'WAREHOUSE_SAFETY_MANUAL.txt')
    safety_manual_content = ""
    try:
        with open(safety_file, 'r', encoding='utf-8') as f:
            safety_manual_content = f.read()
        logger.info("Safety manual loaded.")
    except Exception as e:
        logger.warning(f"Safety manual not loaded: {e}")

    # 5. Build safety chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        length_function=len,
    )
    safety_chunks_docs = []
    if safety_manual_content:
        safety_chunks_docs = text_splitter.create_documents([safety_manual_content])

    # 6. Traditional search corpus (shelf info + safety chunks)
    traditional_search_corpus = []
    for _, row in shelf_df.iterrows():
        traditional_search_corpus.append(
            f"Item: {row['Item']}. Shelf Name: {row['Shelf Name']}. Shelf ID: {row['Shelf ID']}. Stock: {row['Stock']}."
        )
    for chunk in safety_chunks_docs:
        traditional_search_corpus.append(chunk.page_content)

    # 7. Build vector store – include Price and Category
    shelf_texts = [
        f"Item: {row['Item']}, Shelf: {row['Shelf Name']}, Shelf ID: {row['Shelf ID']}, "
        f"Location: ({row['X Position']:.3f}, {row['Y Position']:.3f}, {row['Z Position']:.3f}), "
        f"Stock: {row['Stock']}, Price: ${row['Value']}, Category: {row['Category']}"
        for _, row in shelf_df.iterrows()
    ]
    safety_texts_with_metadata = []
    for i, chunk in enumerate(safety_chunks_docs):
        safety_texts_with_metadata.append(f"Source: Safety Manual, Chunk {i+1}\n{chunk.page_content}")

    all_combined_texts = shelf_texts + safety_texts_with_metadata
    vector_store = None
    if all_combined_texts:
        embed_model = SentenceTransformer(config["embedding_model"])
        vector_store = FAISS.from_texts(
            texts=all_combined_texts,
            embedding=SentenceTransformerEmbeddings(embed_model)
        )
        logger.info(f"Vector store built with {len(all_combined_texts)} chunks.")
    else:
        logger.warning("No texts for vector store.")

    return {
        "shelf_df": shelf_df,
        "shelf_locations": shelf_locations,
        "shelf_name_to_info": shelf_name_to_info,
        "shelf_id_to_info": shelf_id_to_info,
        "all_items_for_order_page": all_items_for_order_page,
        "traditional_search_corpus": traditional_search_corpus,
        "safety_manual_content": safety_manual_content,
        "safety_chunks_docs": safety_chunks_docs,
        "vector_store": vector_store,
        "all_combined_texts": all_combined_texts,
    }