#!/usr/bin/env python3

import os
import csv
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
    "repeat_api_3runs.csv"
)

API_URL = "http://127.0.0.1:5000/query"


with open(
    GROUND_TRUTH,
    "r",
    encoding="utf-8-sig"
) as f:

    queries = list(
        csv.DictReader(f)
    )


results = []


for run_number in range(
    1,
    4
):


    print(
        f"\n========== RUN "
        f"{run_number} =========="
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
                        "auto"
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


        predicted_tier = str(
            data.get(
                "tier",
                ""
            )
        ).lower()


        routing_correct = (

            predicted_tier

            == row[
                "expected_tier"
            ].lower()
        )


        message = data.get(
            "message",
            data.get(
                "response",
                ""
            )
        )


        results.append({

            "run":
                run_number,

            "query":
                row["query"],

            "expected_tier":
                row[
                    "expected_tier"
                ],

            "predicted_tier":
                predicted_tier,

            "routing_correct":
                routing_correct,

            "latency":
                latency,

            "correct_answer":
                row[
                    "correct_answer"
                ],

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