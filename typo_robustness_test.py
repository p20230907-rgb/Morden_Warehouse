#!/usr/bin/env python3

import os
import csv
import re
import time
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
    "noisy_ablation_results_final.csv"
)

API_URL = "http://127.0.0.1:5000/query"


def create_typo(text):

    words = text.split()

    for i, word in enumerate(words):

        positions = [
            p
            for p, char in enumerate(word)
            if char.isalpha()
        ]

        if len(positions) >= 5:

            p1 = positions[1]
            p2 = positions[2]

            chars = list(word)

            chars[p1], chars[p2] = (
                chars[p2],
                chars[p1]
            )

            words[i] = "".join(chars)

            return " ".join(words)

    return text


with open(
    GROUND_TRUTH,
    "r",
    encoding="utf-8-sig"
) as f:

    queries = list(
        csv.DictReader(f)
    )


results = []


for i, row in enumerate(queries, 1):

    original_query = row["query"]

    noisy_query = create_typo(
        original_query
    )


    payload = {
        "query": noisy_query,
        "mode": "auto"
    }


    start = time.perf_counter()


    try:

        response = requests.post(
            API_URL,
            json=payload,
            timeout=120
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


    predicted_tier = str(
        data.get(
            "tier",
            data.get(
                "selected_tier",
                ""
            )
        )
    ).lower()


    message = data.get(
        "message",
        data.get(
            "response",
            data.get(
                "answer",
                ""
            )
        )
    )


    expected_tier = (
        row["expected_tier"]
        .strip()
        .lower()
    )


    routing_correct = (
        predicted_tier
        == expected_tier
    )


    results.append({

        "original_query":
            original_query,

        "noisy_query":
            noisy_query,

        "expected_tier":
            expected_tier,

        "predicted_tier":
            predicted_tier,

        "routing_correct":
            routing_correct,

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
        f"{noisy_query}"
    )

    print(
        f"    Expected={expected_tier}, "
        f"Predicted={predicted_tier}, "
        f"Correct={routing_correct}"
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


routing_accuracy = (
    sum(
        r["routing_correct"]
        for r in results
    )
    / len(results)
    * 100
)


print(
    f"\nRouting accuracy = "
    f"{routing_accuracy:.2f}%"
)

print(
    "Saved:",
    OUTPUT
)