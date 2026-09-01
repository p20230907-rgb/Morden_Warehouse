# Morden_Warehouse

## A Tiered Expert System for Evidence-Grounded Warehouse Assistance and Robot Dispatch

This repository contains the implementation, evaluation scripts, datasets, results, and ROS 2 integration used for the research work:

**A Tiered Expert System for Evidence-Grounded Warehouse Assistance and Robot Dispatch**

The system combines structured inventory lookup, keyword retrieval, LLM reasoning, retrieval-augmented generation (RAG), and ROS 2/Nav2-based robot dispatch within a unified warehouse-assistance framework.

---

## 1. System Overview

The proposed system uses a four-tier query-processing architecture:

| Tier | Method | Purpose |
|---|---|---|
| Tier 1 | Direct inventory lookup | Exact inventory and shelf queries |
| Tier 2 | Keyword retrieval | Lightweight factual and document queries |
| Tier 3 | LLM reasoning | Contextual reasoning without retrieval |
| Tier 4 | RAG | Evidence-grounded responses using warehouse documents |

A routing module determines the appropriate processing tier for each incoming query.

Three routing strategies are included:

- Rule-based routing
- TF-IDF + Logistic Regression classifier
- Sentence-BERT classifier with engineered query features

Low-confidence learned-router predictions can be redirected to the RAG tier.

---

## 2. Main Features

- Structured warehouse inventory lookup
- Natural-language warehouse assistance
- TF-IDF keyword retrieval
- FAISS semantic retrieval
- Retrieval-Augmented Generation
- Multiple LLM backends
- Confidence-aware query routing
- Structured robot-dispatch outputs
- ROS 2 / Nav2 robot navigation
- Order-driven robot dispatch
- Human-robot assistance dashboard
- Inventory dashboard
- Retrieval and routing ablation experiments
- Typographical-noise robustness evaluation
- Repeated latency evaluation
- Robot-dispatch evaluation

---

## 3. Repository Structure

```text
Morden_Warehouse/
├── newapp.py
├── config.py
├── data_loader.py
├── router.py
├── retriever.py
├── robot_controller.py
├── safety.py
├── logger.py
├── data/
│   ├── shelfinfo.csv
│   └── WAREHOUSE_SAFETY_MANUAL.txt
├── templates/
├── router_classifier.pkl
├── router_classifier_advanced.pkl
├── router_training_clean_160.csv
├── router_test_unseen_60_with_ground_truth_final.csv
├── train_router_classifier_final.py
├── train_router_classifier_advanced_final.py
├── baseline_comparison_final.py
├── run_full_ablation.py
├── run_chunk_ablation.py
├── run_model_ablation.py
├── run_noisy_ablation.py
├── embedding_retrieval_test.py
├── real_hybrid_fusion_test.py
├── repeat_api_3runs.py
├── robot_dispatch_60_test.py
├── robot_dispatch_60_challenging_test.py
├── results/
│   ├── final/
│   └── raw_runs/
├── supplementary/
└── ros2/
    └── warehouse_sim/
```

---

## 4. Data

The primary warehouse inventory data are stored in:

```text
data/shelfinfo.csv
```

Warehouse operational and safety information is stored in:

```text
data/WAREHOUSE_SAFETY_MANUAL.txt
```

These sources are used to construct the keyword-search corpus and the FAISS semantic retrieval index.

---

## 5. Software Requirements

- Ubuntu 22.04
- Python 3.10
- ROS 2 Humble
- Nav2
- Gazebo
- Flask
- Pandas
- NumPy
- Scikit-learn
- SentenceTransformers
- FAISS
- LangChain
- Together AI API

---

## 6. Python Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the main Python dependencies:

```bash
pip install flask flask-cors python-dotenv pandas numpy requests scikit-learn joblib
pip install sentence-transformers faiss-cpu
pip install langchain langchain-community langchain-text-splitters langchain-together
```

ROS 2 packages such as `rclpy`, `geometry_msgs`, and `nav2_msgs` should be installed through the ROS 2 Humble environment rather than through pip.

---

## 7. API Configuration

Create a `.env` file in the project root:

```text
TOGETHER_API_KEY=your_api_key_here
```

Do not upload `.env` or API keys to GitHub.

---

## 8. Run the Warehouse Assistant

```bash
source /opt/ros/humble/setup.bash
source venv/bin/activate
python3 newapp.py
```

The dashboard is typically available at:

```text
http://127.0.0.1:5000
```

---

## 9. Router Training

### TF-IDF Router

```bash
python3 train_router_classifier_final.py
```

### Sentence-BERT Router

```bash
python3 train_router_classifier_advanced_final.py
```

The router-development dataset contains 160 labelled training queries.

The independent evaluation set contains 60 warehouse queries:

```text
router_test_unseen_60_with_ground_truth_final.csv
```

---

## 10. Baseline Evaluation

The main baseline experiment compares:

- Direct-only
- Keyword-only
- LLM-only
- RAG-only
- Tiered architecture

Run:

```bash
python3 baseline_comparison_final.py
```

Final results are stored under:

```text
results/final/
```

---

## 11. Typographical Robustness

```bash
python3 typo_robustness_test.py
```

or:

```bash
python3 run_noisy_ablation.py
```

---

## 12. Embedding Evaluation

```bash
python3 embedding_retrieval_test.py
```

Evaluated embedding models include:

- `paraphrase-MiniLM-L6-v2`
- `all-MiniLM-L6-v2`
- `all-mpnet-base-v2`

---

## 13. Chunk-Size Ablation

| Chunk size | Overlap |
|---:|---:|
| 250 | 50 |
| 500 | 100 |
| 750 | 150 |
| 1000 | 200 |

Run:

```bash
python3 run_chunk_ablation.py
```

---

## 14. Hybrid Retrieval Experiment

```bash
python3 real_hybrid_fusion_test.py
```

Evaluated semantic/keyword configurations:

```text
1.0 / 0.0
0.0 / 1.0
0.5 / 0.5
0.6 / 0.4
0.7 / 0.3
0.8 / 0.2
```

---

## 15. Repeated Latency Evaluation

```bash
python3 repeat_api_3runs.py
python3 latency_stats.py
```

Reported metrics include mean latency, standard deviation, median, and P95 latency.

---

## 16. Robot Dispatch Evaluation

Standard dispatch evaluation:

```bash
python3 robot_dispatch_60_test.py
```

Challenging dispatch evaluation:

```bash
python3 robot_dispatch_60_challenging_test.py
```

The evaluation distinguishes direct item-to-shelf requests from more difficult reasoning-based destination requests.

---

## 17. ROS 2 / Nav2 Integration

ROS 2 simulation resources are located under:

```text
ros2/warehouse_sim/
```

The robot execution layer uses the Nav2 `NavigateToPose` action interface and `geometry_msgs/PoseStamped` goals.

Before running the robot system:

```bash
source /opt/ros/humble/setup.bash
```

If using a ROS workspace:

```bash
cd ~/ros2_ws
colcon build
source install/setup.bash
```

---

## 18. Safety Scope

The current implementation performs execution-validity checks before robot dispatch.

The current implementation should not be interpreted as a formal robotic safety-certification system.

The following are outside the scope of the present implementation:

- Human-aware collision-risk assessment
- Multi-robot traffic coordination
- Fleet-level collision management
- Aisle occupancy arbitration
- Formal runtime safety verification

---

## 19. Experimental Results

Final experimental results:

```text
results/final/
```

Raw experimental runs:

```text
results/raw_runs/
```

---

## 20. Supplementary Material

Additional reproducibility resources are available under:

```text
supplementary/
```

These include query sets, prompt templates, robot dispatch schema, inventory data, warehouse safety documents, and router training data.

---

## 21. Authors

**Kabita Choudhary**  
Department of Computer Science and Information Systems  
BITS Pilani, Dubai Campus

**Karthikeyan Ramanujam**  
Department of Mechanical Engineering  
BITS Pilani, Dubai Campus

**Sujala D. Shetty**  
Department of Computer Science and Information Systems  
BITS Pilani, Dubai Campus

**Kalaichelvi Venkatesan**  
Department of Electrical and Electronics Engineering  
BITS Pilani, Dubai Campus

---

## 22. Citation

If you use this repository, please cite the associated manuscript:

```text
K. Choudhary, K. Ramanujam, S. D. Shetty, and K. Venkatesan,
"A Tiered Expert System for Evidence-Grounded Warehouse Assistance
and Robot Dispatch."
```

The final journal citation and DOI can be added after publication.

---

## 23. Repository

https://github.com/p20230907-rgb/Morden_Warehouse

---

## License

This repository is provided for academic and research purposes.

A formal open-source license can be added depending on the intended reuse and distribution policy.
