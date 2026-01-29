# Modern Warehouse Management System (WMS) with AI Assistants & Robot Execution

A modern warehouse management system that combines **real-time inventory dashboards**, **order management**, and **AI-powered assistance** (Direct Lookup, LLM reasoning, and RAG over safety/SOP documents) with optional **robot navigation** (ROS 2) for shelf-to-item execution.

This project is designed for **warehouse-style environments** where operators need fast answers (inventory + location), safe operations (SOP compliance), and efficient fulfillment (multi-shelf order routing).

---

## ✨ Highlights

- **Inventory Dashboard**
  - Item list, shelf ID/name, coordinates (X,Y,Z), stock level, category, value
  - KPI tiles: total items, total units, low-stock count, total inventory value

- **Order Management**
  - Select items + quantities
  - Server-side stock validation and stock decrement
  - Groups items by shelf to minimize travel
  - Optional ROS 2 navigation to shelves

- **AI Assistants (Multi-tier Query Routing)**
  - **Direct Lookup**: deterministic and fast for exact item/shelf queries
  - **LLM-only reasoning**: context-based answers for more complex inventory questions
  - **RAG (Vector Search + LLM)**: grounded answers from inventory + safety manuals/SOPs

- **Safety Manual Integration**
  - Uses an SOP text file to answer questions like PPE, emergency exits, fire safety, training, etc.
  - RAG ensures answers are **evidence-based** (no hallucinations)

---

## 🧱 System Architecture (High-Level)

**UI Layer (Flask )**
- Inventory dashboard
- AI assistance pages
- Order management page
- Performance & comparison pages

**Data Layer**
- `shelfinfo.csv` for inventory + shelf coordinates + stock
- Safety SOP text file for compliance queries

**AI Layer**
- SentenceTransformer embeddings + FAISS vector store for retrieval
- Groq-hosted LLM (via LangChain) for response generation
- Routing logic selects best method: Direct → LLM → RAG

**Execution Layer (Optional)**
- ROS 2 navigation node (`NavToShelf`) to send goals (x,y,z + orientation)

---

DEmo URL: https://www.youtube.com/watch?v=n_hodmLCzXY
