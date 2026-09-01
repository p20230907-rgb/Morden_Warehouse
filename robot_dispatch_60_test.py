#!/usr/bin/env python3

import os
import csv
import re
import time
import requests


SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

INVENTORY_FILE = os.path.join(
    SCRIPT_DIR,
    "shelfinfo.csv"
)

OUTPUT_FILE = os.path.join(
    SCRIPT_DIR,
    "robot_dispatch_60_results.csv"
)


QUERY_URL = "http://127.0.0.1:5000/query"
ROBOT_URL = "http://127.0.0.1:5000/send_to_robot"


# ============================================================
# Load inventory
# ============================================================

with open(
    INVENTORY_FILE,
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:

    inventory = list(
        csv.DictReader(f)
    )


# ============================================================
# Command templates
# ============================================================

templates = [

    "Navigate to {item}.",

    "Send the robot to {item}.",

    "Go to the shelf containing {item}.",

    "Please move the robot to {item}.",
]


# ============================================================
# Build exactly 60 requests
# ============================================================

requests_60 = []

i = 0

while len(requests_60) < 60:

    row = inventory[
        i % len(inventory)
    ]

    template_index = (
        i // len(inventory)
    ) % len(templates)

    query = templates[
        template_index
    ].format(
        item=row["Item"]
    )

    requests_60.append({

        "query":
            query,

        "item":
            row["Item"],

        "expected_shelf_id":
            row["Shelf ID"],

        "expected_shelf_name":
            row["Shelf Name"]
    })

    i += 1


# ============================================================
# Shelf extraction
# Mirrors robot_assistance.html
# ============================================================

def extract_shelf_from_query_response(data):

    # --------------------------------------------------------
    # METHOD 1:
    # Structured data
    # --------------------------------------------------------

    structured = data.get(
        "data",
        {}
    )

    if isinstance(
        structured,
        dict
    ):

        shelf = structured.get(
            "shelf"
        )

        shelf_id = structured.get(
            "shelf_id"
        )

        if shelf and str(shelf).strip().upper() != "N/A":

            return str(
                shelf
            ).strip()

        if shelf_id and str(shelf_id).strip().upper() != "N/A":

            return (
                "Shelf "
                + str(shelf_id).strip()
            )


    # --------------------------------------------------------
    # METHOD 2:
    # LLM answer / message
    # --------------------------------------------------------

    llm_response = ""

    if isinstance(
        structured,
        dict
    ):

        llm_response = (
            structured.get(
                "llm_answer"
            )
            or ""
        )


    if not llm_response:

        llm_response = (
            data.get(
                "message"
            )
            or ""
        )


    shelf_patterns = [

        r"Shelf\s+([A-G])",

        r"shelf\s+([A-G])",

        r"ID:\s*([A-G])",

        r"\(ID\s*:\s*([A-G])\)"
    ]


    for pattern in shelf_patterns:

        match = re.search(
            pattern,
            llm_response,
            re.IGNORECASE
        )

        if match:

            return (
                "Shelf "
                + match.group(1).upper()
            )


    return ""


# ============================================================
# Normalize shelf
# ============================================================

def normalize_shelf(value):

    value = str(
        value
    ).strip()


    match = re.search(
        r"Shelf\s*([A-G])",
        value,
        re.IGNORECASE
    )

    if match:

        return match.group(
            1
        ).upper()


    if (
        len(value) == 1
        and value.upper()
        in list("ABCDEFG")
    ):

        return value.upper()


    return ""


# ============================================================
# Run test
# ============================================================

results = []


for index, request_row in enumerate(
    requests_60,
    1
):


    # --------------------------------------------------------
    # STEP 1:
    # Send natural-language command to /query
    # Same as robot_assistance.html
    # --------------------------------------------------------

    query_payload = {

        "query":
            request_row[
                "query"
            ],

        "mode":
            "llm_only"
    }


    start = time.perf_counter()


    try:

        query_response = requests.post(

            QUERY_URL,

            json=query_payload,

            timeout=180
        )


        query_latency = (
            time.perf_counter()
            - start
        )


        query_response.raise_for_status()

        query_data = (
            query_response.json()
        )


    except Exception as e:

        query_latency = (
            time.perf_counter()
            - start
        )

        query_data = {
            "error": str(e)
        }


    # --------------------------------------------------------
    # STEP 2:
    # Extract target shelf
    # --------------------------------------------------------

    predicted_shelf_raw = (
        extract_shelf_from_query_response(
            query_data
        )
    )


    predicted_shelf_id = (
        normalize_shelf(
            predicted_shelf_raw
        )
    )


    expected_shelf_id = (
        request_row[
            "expected_shelf_id"
        ]
        .strip()
        .upper()
    )


    target_correct = (

        predicted_shelf_id
        == expected_shelf_id
    )


    # --------------------------------------------------------
    # STEP 3:
    # Send to robot endpoint
    # --------------------------------------------------------

    robot_called = False
    robot_success = False
    robot_message = ""
    robot_latency = 0.0


    if predicted_shelf_raw:

        robot_called = True


        robot_start = (
            time.perf_counter()
        )


        try:

            robot_response = (
                requests.post(

                    ROBOT_URL,

                    json={
                        "item":
                            predicted_shelf_raw
                    },

                    timeout=180
                )
            )


            robot_latency = (
                time.perf_counter()
                - robot_start
            )


            robot_data = (
                robot_response.json()
            )


            robot_success = (

                robot_data.get(
                    "status"
                )
                == "success"
            )


            robot_message = (
                robot_data.get(
                    "message",
                    ""
                )
            )


        except Exception as e:

            robot_latency = (
                time.perf_counter()
                - robot_start
            )

            robot_message = str(e)


    # --------------------------------------------------------
    # Total latency
    # --------------------------------------------------------

    total_latency = (
        query_latency
        + robot_latency
    )


    # --------------------------------------------------------
    # Original LLM message
    # --------------------------------------------------------

    query_message = (
        query_data.get(
            "message",
            ""
        )
    )


    # --------------------------------------------------------
    # Save row
    # --------------------------------------------------------

    results.append({

        "query":
            request_row[
                "query"
            ],

        "item":
            request_row[
                "item"
            ],

        "expected_shelf":
            expected_shelf_id,

        "predicted_shelf_raw":
            predicted_shelf_raw,

        "predicted_shelf":
            predicted_shelf_id,

        "target_correct":
            target_correct,

        "robot_called":
            robot_called,

        "robot_success":
            robot_success,

        "query_latency":
            query_latency,

        "robot_latency":
            robot_latency,

        "total_latency":
            total_latency,

        "query_message":
            query_message,

        "robot_message":
            robot_message
    })


    print(
        f"[{index}/60] "
        f"Expected={expected_shelf_id} "
        f"Predicted={predicted_shelf_id} "
        f"TargetCorrect={target_correct} "
        f"RobotSuccess={robot_success}"
    )


# ============================================================
# Save CSV
# ============================================================

with open(
    OUTPUT_FILE,
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


# ============================================================
# Summary
# ============================================================

target_accuracy = (

    sum(
        r[
            "target_correct"
        ]
        for r in results
    )

    / len(results)

    * 100
)


robot_called_count = sum(

    r[
        "robot_called"
    ]

    for r in results
)


robot_success_count = sum(

    r[
        "robot_success"
    ]

    for r in results
)


robot_success_rate = (

    robot_success_count
    / robot_called_count
    * 100

    if robot_called_count > 0

    else 0.0
)


print("\n" + "=" * 60)

print("FINAL ROBOT DISPATCH RESULTS")

print("=" * 60)

print(
    f"Target shelf accuracy: "
    f"{target_accuracy:.2f}%"
)

print(
    f"Robot endpoint called: "
    f"{robot_called_count}/60"
)

print(
    f"Robot endpoint success: "
    f"{robot_success_count}/"
    f"{robot_called_count}"
)

print(
    f"Robot success rate: "
    f"{robot_success_rate:.2f}%"
)

print(
    "\nSaved:",
    OUTPUT_FILE
)