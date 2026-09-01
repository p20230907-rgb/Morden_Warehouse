#!/usr/bin/env python3

import csv
import os
import re
import time
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_FILE = os.path.join(
    SCRIPT_DIR,
    "robot_dispatch_60_challenging_results.csv"
)

QUERY_URL = "http://127.0.0.1:5000/query"


# ============================================================
# 30 SIMPLE COMMANDS
# ============================================================

simple_commands = [

    ("Navigate to Apple iPhone 14.", "A"),
    ("Navigate to Apple iPhone 14 Pro.", "A"),
    ("Navigate to Apple iPhone 13.", "A"),
    ("Navigate to iPhone Charging Cable.", "A"),
    ("Navigate to iPhone Screen Protector.", "A"),

    ("Send the robot to Samsung Galaxy S23.", "B"),
    ("Send the robot to Samsung Galaxy S24.", "B"),
    ("Send the robot to Samsung Galaxy Tab S9.", "B"),
    ("Send the robot to Samsung Wireless Earbuds.", "B"),
    ("Send the robot to Samsung Fast Charger.", "B"),

    ("Go to Sony WH-1000XM5.", "C"),
    ("Go to Sony WH-CH720N.", "C"),
    ("Go to Sony WF-1000XM5.", "C"),
    ("Go to Sony Extra Bass Headphones.", "C"),
    ("Go to Sony Carrying Case.", "C"),

    ("Move to Dell XPS 13.", "D"),
    ("Move to Dell XPS 15.", "D"),
    ("Move to Dell Inspiron 14.", "D"),
    ("Move to Dell Laptop Charger.", "D"),
    ("Move to Dell Laptop Bag.", "D"),

    ("Navigate to Bose QuietComfort 45.", "E"),
    ("Navigate to Bose QuietComfort Ultra.", "E"),
    ("Navigate to Bose SoundLink Flex.", "E"),
    ("Navigate to Bose Soundbar 600.", "E"),
    ("Navigate to Bose Replacement Ear Pads.", "E"),

    ("Send the robot to Logitech MX Master 3.", "F"),
    ("Send the robot to Logitech MX Keys.", "F"),
    ("Send the robot to Logitech G Pro X Superlight.", "F"),
    ("Send the robot to Logitech Webcam C920.", "F"),
    ("Send the robot to Logitech Mouse Pad.", "F"),
]


# ============================================================
# 30 DIFFICULT / REASONING COMMANDS
# ============================================================

difficult_commands = [

    (
        "Send the robot to the shelf that stores the gaming console with 99 units in stock.",
        "G"
    ),

    (
        "Navigate to the location of the laptop model whose stock is 45 units.",
        "D"
    ),

    (
        "Go to the shelf containing the Bose product with the lowest stock level.",
        "E"
    ),

    (
        "Dispatch the robot to the shelf where the most heavily stocked Samsung phone is stored.",
        "B"
    ),

    (
        "Move to the shelf containing the Logitech product with the highest quantity.",
        "F"
    ),

    (
        "Take the robot to the shelf that contains the product priced at 549.99 dollars.",
        "G"
    ),

    (
        "Navigate to the shelf containing the Sony headphones with 23 units remaining.",
        "C"
    ),

    (
        "Go to the shelf that contains the Apple phone with the lowest stock.",
        "A"
    ),

    (
        "Send the robot to the shelf whose total stock is the highest in the warehouse.",
        "B"
    ),

    (
        "Navigate to the shelf with the second-highest total number of units.",
        "G"
    ),

    (
        "Move the robot to the shelf with the lowest aggregate inventory quantity.",
        "E"
    ),

    (
        "Go to the shelf whose total stock is closest to the average stock per shelf.",
        "F"
    ),

    (
        "Send the robot to the shelf that has two more units than Shelf C.",
        "F"
    ),

    (
        "Navigate to the shelf that has 196 fewer units than Shelf B.",
        "C"
    ),

    (
        "Take the robot to the shelf containing both the PlayStation VR2 and the DualSense Controller.",
        "G"
    ),

    (
        "Go to the shelf where both Dell XPS models are stored.",
        "D"
    ),

    (
        "Dispatch the robot to the shelf containing both Bose QuietComfort models.",
        "E"
    ),

    (
        "Navigate to the location that stores both the MX Keys and MX Master 3.",
        "F"
    ),

    (
        "Send the robot to the shelf containing both Galaxy S23 and Galaxy S24.",
        "B"
    ),

    (
        "Take the robot to the shelf holding the product tied for the minimum stock level that belongs to Logitech.",
        "F"
    ),

    (
        "Go to the shelf holding the other minimum-stock product, excluding Logitech.",
        "E"
    ),

    (
        "Send the robot to the shelf with the highest total stock, but not the shelf containing PlayStation products.",
        "B"
    ),

    (
        "Go to the shelf containing headphones, but not Sony headphones; choose the shelf with the lowest-stock headphone product.",
        "E"
    ),

    (
        "Navigate to the shelf containing the smartphone with 196 units, not the shelf containing Apple phones.",
        "B"
    ),

    (
        "Send the robot to the shelf containing the 19.99-dollar accessory with the highest stock.",
        "F"
    ),

    (
        "Move to the shelf containing the 29.99-dollar accessory with the greatest stock.",
        "B"
    ),

    (
        "Navigate to the shelf containing the most expensive Dell product.",
        "D"
    ),

    (
        "Go to the shelf containing the most expensive Bose product.",
        "E"
    ),

    (
        "I need the 699.99-dollar product that is not a tablet. Navigate to its shelf.",
        "D"
    ),

    (
        "Dispatch to the shelf containing the 8-unit product that is a laptop rather than software.",
        "D"
    ),
]


# ============================================================
# Combine = exactly 60
# ============================================================

test_cases = []

for command, shelf in simple_commands:
    test_cases.append({
        "difficulty": "simple",
        "query": command,
        "expected_shelf": shelf
    })

for command, shelf in difficult_commands:
    test_cases.append({
        "difficulty": "difficult",
        "query": command,
        "expected_shelf": shelf
    })


assert len(test_cases) == 60, f"Expected 60 tests, got {len(test_cases)}"


# ============================================================
# Extract shelf
# Mirrors robot_assistance.html logic
# ============================================================

def extract_shelf(data):

    structured = data.get("data", {})

    # 1. Structured shelf
    if isinstance(structured, dict):

        shelf = structured.get("shelf")

        if shelf and str(shelf).strip().upper() != "N/A":

            m = re.search(
                r"Shelf\s*([A-G])",
                str(shelf),
                re.IGNORECASE
            )

            if m:
                return m.group(1).upper()


        shelf_id = structured.get("shelf_id")

        if shelf_id and str(shelf_id).strip().upper() != "N/A":

            value = str(shelf_id).strip().upper()

            if value in list("ABCDEFG"):
                return value


    # 2. LLM answer
    text = ""

    if isinstance(structured, dict):
        text = structured.get("llm_answer") or ""

    if not text:
        text = data.get("message") or ""


    patterns = [
        r"Shelf\s+([A-G])",
        r"shelf\s+([A-G])",
        r"ID:\s*([A-G])",
        r"\(ID\s*:\s*([A-G])\)"
    ]

    for pattern in patterns:

        m = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if m:
            return m.group(1).upper()


    return ""


# ============================================================
# Run tests
# ============================================================

results = []


for i, test in enumerate(test_cases, 1):

    print("\n" + "=" * 70)
    print(f"[{i}/60] {test['difficulty'].upper()}")
    print(test["query"])
    print("=" * 70)

    payload = {
        "query": test["query"],
        "mode": "llm_only"
    }

    start = time.perf_counter()

    try:

        response = requests.post(
            QUERY_URL,
            json=payload,
            timeout=120
        )

        latency = time.perf_counter() - start

        response.raise_for_status()

        data = response.json()

    except Exception as e:

        latency = time.perf_counter() - start

        data = {
            "error": str(e)
        }


    predicted_shelf = extract_shelf(data)

    expected_shelf = test["expected_shelf"]

    correct = (
        predicted_shelf == expected_shelf
    )

    message = data.get(
        "message",
        ""
    )


    print(
        f"Expected={expected_shelf} "
        f"Predicted={predicted_shelf} "
        f"Correct={correct} "
        f"Latency={latency:.3f}s"
    )


    results.append({

        "difficulty":
            test["difficulty"],

        "query":
            test["query"],

        "expected_shelf":
            expected_shelf,

        "predicted_shelf":
            predicted_shelf,

        "dispatch_correct":
            correct,

        "latency":
            latency,

        "message":
            message
    })


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
        fieldnames=results[0].keys()
    )

    writer.writeheader()

    writer.writerows(results)


# ============================================================
# Summary
# ============================================================

simple_results = [
    r for r in results
    if r["difficulty"] == "simple"
]

difficult_results = [
    r for r in results
    if r["difficulty"] == "difficult"
]


simple_correct = sum(
    r["dispatch_correct"]
    for r in simple_results
)

difficult_correct = sum(
    r["dispatch_correct"]
    for r in difficult_results
)

overall_correct = sum(
    r["dispatch_correct"]
    for r in results
)


simple_acc = (
    simple_correct
    / len(simple_results)
    * 100
)

difficult_acc = (
    difficult_correct
    / len(difficult_results)
    * 100
)

overall_acc = (
    overall_correct
    / len(results)
    * 100
)


print("\n" + "=" * 70)

print("FINAL ROBOT DISPATCH RESULTS")

print("=" * 70)

print(
    f"Simple commands: "
    f"{simple_correct}/30 "
    f"= {simple_acc:.2f}%"
)

print(
    f"Difficult commands: "
    f"{difficult_correct}/30 "
    f"= {difficult_acc:.2f}%"
)

print(
    f"Overall: "
    f"{overall_correct}/60 "
    f"= {overall_acc:.2f}%"
)

print(
    "\nSaved:",
    OUTPUT_FILE
)