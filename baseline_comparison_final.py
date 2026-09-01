#!/usr/bin/env python3

import os
import csv
import time
import statistics
import requests


SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

GROUND_TRUTH = os.path.join(
    SCRIPT_DIR,
    "router_test_unseen_60_with_ground_truth_final.csv"
)

OUTPUT = os.path.join(
    SCRIPT_DIR,
    "baseline_results_final.csv"
)

API_URL = "http://127.0.0.1:5000/query"


BASELINES = [

    ("Tiered", "auto"),

    ("Direct-only", "direct_only"),

    (
        "Keyword-only",
        "traditional_search_only"
    ),

    ("LLM-only", "llm_only"),

    ("RAG-only", "rag_only"),
]


with open(
    GROUND_TRUTH,
    "r",
    encoding="utf-8-sig"
) as f:

    queries = list(
        csv.DictReader(f)
    )


results = []


for baseline_name, mode in BASELINES:

    print(
        "\n=============================="
    )

    print(
        "Testing:",
        baseline_name
    )

    print(
        "=============================="
    )


    for i, row in enumerate(
        queries,
        1
    ):


        start = time.perf_counter()


        try:

            response = requests.post(

                API_URL,

                json={
                    "query":
                        row["query"],

                    "mode":
                        mode
                },

                timeout=180
            )


            latency = (
                time.perf_counter()
                - start
            )


            data = response.json()


        except Exception as e:

            latency = (
                time.perf_counter()
                - start
            )

            data = {
                "error": str(e)
            }


        message = data.get(
            "message",
            data.get(
                "response",
                ""
            )
        )


        results.append({

            "baseline":
                baseline_name,

            "query":
                row["query"],

            "expected_tier":
                row["expected_tier"],

            "returned_tier":
                data.get(
                    "tier",
                    ""
                ),

            "latency":
                latency,

            "correct_answer":
                row["correct_answer"],

            "message":
                message,

            "semantic_label":
                ""
        })


        print(
            f"[{i}/60] "
            f"{latency:.3f}s"
        )


with open(
    OUTPUT,
    "w",
    encoding="utf-8-sig",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=
        results[0].keys()
    )

    writer.writeheader()

    writer.writerows(
        results
    )


print(
    "\nSaved:",
    OUTPUT
)