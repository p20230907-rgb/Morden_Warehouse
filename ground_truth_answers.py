#!/usr/bin/env python3
"""
ground_truth_answers.py – Defines the correct answer for each of the 60 queries
with expected key details for partial/full correctness.
"""

GROUND_TRUTH = {
    # === DIRECT (15) ===
    "Apple iPhone 14": {
        "expected": "Apple iPhone 14 is on Shelf A with stock 10 and price $799.99",
        "must_have": ["iPhone 14", "Shelf A", "10"],
        "should_have": ["799.99", "Smartphones"],
        "type": "direct"
    },
    "Samsung Galaxy S23": {
        "expected": "Samsung Galaxy S23 is on Shelf B with stock 196 and price $749.99",
        "must_have": ["Galaxy S23", "Shelf B", "196"],
        "should_have": ["749.99", "Smartphones"],
        "type": "direct"
    },
    "Sony WH-1000XM5": {
        "expected": "Sony WH-1000XM5 is on Shelf C with stock 16 and price $399.99",
        "must_have": ["WH-1000XM5", "Shelf C", "16"],
        "should_have": ["399.99", "Headphones"],
        "type": "direct"
    },
    "Dell XPS 13": {
        "expected": "Dell XPS 13 is on Shelf D with stock 45 and price $1199.99",
        "must_have": ["XPS 13", "Shelf D", "45"],
        "should_have": ["1199.99", "Laptops"],
        "type": "direct"
    },
    "Bose QuietComfort 45": {
        "expected": "Bose QuietComfort 45 is on Shelf E with stock 11 and price $329.99",
        "must_have": ["QuietComfort 45", "Shelf E", "11"],
        "should_have": ["329.99", "Headphones"],
        "type": "direct"
    },
    "Logitech MX Master 3": {
        "expected": "Logitech MX Master 3 is on Shelf F with stock 7 and price $99.99",
        "must_have": ["MX Master 3", "Shelf F", "7"],
        "should_have": ["99.99", "Computer Mouse"],
        "type": "direct"
    },
    "Sony PlayStation 5": {
        "expected": "Sony PlayStation 5 is on Shelf G with stock 99 and price $499.99",
        "must_have": ["PlayStation 5", "Shelf G", "99"],
        "should_have": ["499.99", "Gaming Consoles"],
        "type": "direct"
    },
    "Samsung Galaxy S24": {
        "expected": "Samsung Galaxy S24 is on Shelf B with stock 22 and price $899.99",
        "must_have": ["Galaxy S24", "Shelf B", "22"],
        "should_have": ["899.99", "Smartphones"],
        "type": "direct"
    },
    "Apple iPhone 14 Pro": {
        "expected": "Apple iPhone 14 Pro is on Shelf A with stock 10 and price $999.99",
        "must_have": ["iPhone 14 Pro", "Shelf A", "10"],
        "should_have": ["999.99", "Smartphones"],
        "type": "direct"
    },
    "Bose SoundLink Flex": {
        "expected": "Bose SoundLink Flex is on Shelf E with stock 25 and price $149.99",
        "must_have": ["SoundLink Flex", "Shelf E", "25"],
        "should_have": ["149.99", "Speakers"],
        "type": "direct"
    },
    "Shelf A": {
        "expected": "Shelf A contains: iPhone 14 (10), iPhone 14 Pro (10), iPhone 13 (8), Charging Cable (45), Screen Protector (32)",
        "must_have": ["Shelf A", "iPhone"],
        "should_have": ["iPhone 14", "iPhone 14 Pro"],
        "type": "direct"
    },
    "Shelf B": {
        "expected": "Shelf B contains: Galaxy S23 (196), Galaxy S24 (22), Galaxy Tab S9 (15), Wireless Earbuds (27), Fast Charger (66)",
        "must_have": ["Shelf B", "Galaxy"],
        "should_have": ["Galaxy S23", "196"],
        "type": "direct"
    },
    "Shelf C": {
        "expected": "Shelf C contains: WH-1000XM5 (16), WH-CH720N (24), WF-1000XM5 (18), Extra Bass Headphones (31), Carrying Case (42)",
        "must_have": ["Shelf C", "Sony"],
        "should_have": ["WH-1000XM5", "16"],
        "type": "direct"
    },
    "Shelf D": {
        "expected": "Shelf D contains: XPS 13 (45), XPS 15 (8), Inspiron 14 (14), Laptop Charger (23), Laptop Bag (19)",
        "must_have": ["Shelf D", "Dell"],
        "should_have": ["XPS 13", "45"],
        "type": "direct"
    },
    "How many Samsung Galaxy S23 are there?": {
        "expected": "There are 196 units of Samsung Galaxy S23 in stock.",
        "must_have": ["196"],
        "should_have": ["Samsung", "Galaxy S23"],
        "type": "direct"
    },

    # === KEYWORD (15) ===
    "Safety Helmets": {
        "expected": "Safety Helmets are required at all times inside loading areas.",
        "must_have": ["Safety Helmets", "loading areas"],
        "should_have": ["required"],
        "type": "keyword"
    },
    "Fire Extinguishers": {
        "expected": "Fire extinguishers: one every 15 meters, monthly inspection.",
        "must_have": ["Fire Extinguishers", "15 meters"],
        "should_have": ["monthly"],
        "type": "keyword"
    },
    "forklift operators": {
        "expected": "Forklift operators must be certified and recertified every 2 years.",
        "must_have": ["forklift", "certified"],
        "should_have": ["2 years"],
        "type": "keyword"
    },
    "steel-toed boots": {
        "expected": "Steel-toed boots must conform to ISO 20345:2011.",
        "must_have": ["steel-toed", "ISO 20345"],
        "should_have": ["2011"],
        "type": "keyword"
    },
    "cut-resistant gloves": {
        "expected": "Cut-resistant gloves are mandatory when handling sharp materials.",
        "must_have": ["cut-resistant", "sharp"],
        "should_have": ["gloves"],
        "type": "keyword"
    },
    "PPE requirements": {
        "expected": "PPE includes: Safety Helmets, High-Visibility Vests, Steel-Toed Boots, Cut-Resistant Gloves, Hearing Protection.",
        "must_have": ["PPE", "Helmets", "Vests"],
        "should_have": ["Gloves"],
        "type": "keyword"
    },
    "LOTO procedure": {
        "expected": "Lockout/Tagout procedures must be reviewed every 12 months.",
        "must_have": ["Lockout", "Tagout"],
        "should_have": ["12 months"],
        "type": "keyword"
    },
    "emergency exits": {
        "expected": "At least two emergency exits per zone, clearly marked.",
        "must_have": ["emergency exits", "two"],
        "should_have": ["clearly marked"],
        "type": "keyword"
    },
    "assembly points": {
        "expected": "Assembly points are 50 meters from the building, marked with green signage.",
        "must_have": ["assembly", "50 meters"],
        "should_have": ["green"],
        "type": "keyword"
    },
    "hazardous chemicals": {
        "expected": "Hazardous chemicals must be stored in locked ventilated cabinets.",
        "must_have": ["hazardous", "locked ventilated cabinets"],
        "should_have": ["chemicals"],
        "type": "keyword"
    },
    "locked ventilated cabinets": {
        "expected": "Locked ventilated cabinets are used to store hazardous chemicals.",
        "must_have": ["locked ventilated cabinets", "hazardous"],
        "should_have": ["chemicals"],
        "type": "keyword"
    },
    "incident report": {
        "expected": "Incidents reported using Form WHS-207 within 1 hour (2024) or Form WHS-101 within 2 hours (2025).",
        "must_have": ["incident", "WHS"],
        "should_have": ["207", "101"],
        "type": "keyword"
    },
    "first aid kit": {
        "expected": "First aid kits are available every 30 meters and checked weekly.",
        "must_have": ["first aid", "30 meters"],
        "should_have": ["weekly"],
        "type": "keyword"
    },
    "warehouse training": {
        "expected": "Safety training required annually for all employees, and immediately after a safety violation or accident.",
        "must_have": ["training", "annually"],
        "should_have": ["safety violation"],
        "type": "keyword"
    },
    "spill containment": {
        "expected": "Spills must be reported immediately and cleaned with absorbent pads.",
        "must_have": ["spill", "report", "absorbent"],
        "should_have": ["immediately"],
        "type": "keyword"
    },

    # === RAG (15) ===
    "What PPE is required for handling chemicals?": {
        "expected": "The manual does not specify specific PPE for chemicals, but general PPE includes helmets, vests, boots, gloves, and hearing protection.",
        "must_have": ["PPE", "chemicals"],
        "should_have": ["helmet", "gloves"],
        "type": "rag"
    },
    "What are the rules for emergency exits?": {
        "expected": "At least two emergency exits per zone, clearly marked.",
        "must_have": ["emergency exits", "two"],
        "should_have": ["clearly marked"],
        "type": "rag"
    },
    "What is the LOTO procedure?": {
        "expected": "Lockout/Tagout procedures must be reviewed every 12 months.",
        "must_have": ["Lockout", "Tagout"],
        "should_have": ["12 months"],
        "type": "rag"
    },
    "How often do we inspect fire extinguishers?": {
        "expected": "Monthly.",
        "must_have": ["monthly"],
        "should_have": ["inspect"],
        "type": "rag"
    },
    "Who can operate forklifts?": {
        "expected": "Only trained, certified staff. Recertified every 2 years.",
        "must_have": ["trained", "certified"],
        "should_have": ["2 years"],
        "type": "rag"
    },
    "Where should I store hazardous materials?": {
        "expected": "In locked ventilated cabinets.",
        "must_have": ["locked ventilated cabinets"],
        "should_have": ["hazardous"],
        "type": "rag"
    },
    "What kind of gloves do I need for sharp objects?": {
        "expected": "Cut-resistant gloves.",
        "must_have": ["cut-resistant"],
        "should_have": ["sharp"],
        "type": "rag"
    },
    "Tell me about the evacuation protocol.": {
        "expected": "Fire extinguishers every 15m, monthly inspection; at least two exits per zone; drills every 6 months; assembly points 50m away; never use elevators.",
        "must_have": ["evacuation", "exits"],
        "should_have": ["15 meters", "50 meters", "6 months"],
        "type": "rag"
    },
    "How often are fire drills conducted?": {
        "expected": "Every 6 months.",
        "must_have": ["6 months"],
        "should_have": ["drills"],
        "type": "rag"
    },
    "What is the procedure for chemical spills?": {
        "expected": "Report immediately, clean with absorbent pads.",
        "must_have": ["spill", "report", "absorbent"],
        "should_have": ["immediately"],
        "type": "rag"
    },
    "What standard must steel-toed boots conform to?": {
        "expected": "ISO 20345:2011.",
        "must_have": ["ISO 20345"],
        "should_have": ["2011"],
        "type": "rag"
    },
    "What's the replacement frequency for safety helmets?": {
        "expected": "Not specified in the manual.",
        "must_have": ["not specified", "not in the context"],
        "should_have": ["replacement"],
        "type": "rag"
    },
    "Explain the emergency evacuation procedure.": {
        "expected": "Use emergency exits, do not use elevators, gather at assembly points 50 meters away.",
        "must_have": ["evacuation", "exits"],
        "should_have": ["50 meters", "assembly"],
        "type": "rag"
    },
    "What should I do during a warehouse fire?": {
        "expected": "Follow evacuation protocol: use emergency exits, don't use elevators, go to assembly point.",
        "must_have": ["fire", "evacuation", "exits"],
        "should_have": ["assembly"],
        "type": "rag"
    },
    "What are the requirements for emergency exits?": {
        "expected": "At least two per zone, clearly marked.",
        "must_have": ["emergency exits", "two"],
        "should_have": ["clearly marked"],
        "type": "rag"
    },

    # === LLM (15) ===
    "Which iPhone model has the highest price?": {
        "expected": "iPhone 14 Pro at $999.99.",
        "must_have": ["iPhone 14 Pro", "999.99"],
        "should_have": ["highest"],
        "type": "llm"
    },
    "Compare the stock on Shelf A and Shelf B.": {
        "expected": "Shelf A has ~105 units, Shelf B has ~326 units. Shelf B has more stock.",
        "must_have": ["Shelf A", "Shelf B", "105", "326"],
        "should_have": ["more"],
        "type": "llm"
    },
    "Which shelf has the most items?": {
        "expected": "Shelf B has the most items (326 units).",
        "must_have": ["Shelf B", "326"],
        "should_have": ["most"],
        "type": "llm"
    },
    "What is the total value of all smartphones?": {
        "expected": "Total value of smartphones is approximately $203,799.68.",
        "must_have": ["203799.68", "smartphone"],
        "should_have": ["value"],
        "type": "llm"
    },
    "Compare the prices of Dell laptops.": {
        "expected": "Dell XPS 13: $1199.99, XPS 15: $1799.99, Inspiron 14: $699.99.",
        "must_have": ["1199.99", "1799.99", "699.99"],
        "should_have": ["Dell", "XPS"],
        "type": "llm"
    },
    "Which item is the most expensive?": {
        "expected": "Dell XPS 15 at $1799.99.",
        "must_have": ["Dell XPS 15", "1799.99"],
        "should_have": ["most expensive"],
        "type": "llm"
    },
    "Which shelf should I visit first to get the most expensive products?": {
        "expected": "Shelf D (contains Dell XPS 15 at $1799.99).",
        "must_have": ["Shelf D", "Dell XPS 15"],
        "should_have": ["1799.99"],
        "type": "llm"
    },
    "What is the difference between Shelf C and Shelf D?": {
        "expected": "Shelf C has Sony audio products; Shelf D has Dell laptops and accessories.",
        "must_have": ["Shelf C", "Sony", "Shelf D", "Dell"],
        "should_have": ["audio", "laptops"],
        "type": "llm"
    },
    "Compare the inventory levels across shelves.": {
        "expected": "Shelf B has the most (326), Shelf E has the least (88).",
        "must_have": ["Shelf B", "326"],
        "should_have": ["Shelf E", "88"],
        "type": "llm"
    },
    "Which product has the lowest stock?": {
        "expected": "Bose QuietComfort Ultra (7) or Logitech MX Master 3 (7).",
        "must_have": ["7"],
        "should_have": ["Bose", "Logitech"],
        "type": "llm"
    },
    "Which shelf has the highest total inventory value?": {
        "expected": "Shelf B has the highest total inventory value.",
        "must_have": ["Shelf B", "highest"],
        "should_have": ["value"],
        "type": "llm"
    },
    "Summarize the inventory on Shelf F.": {
        "expected": "Shelf F has Logitech products: MX Master 3 (7), MX Keys (14), G Pro X (22), Webcam (31), Mouse Pad (58). Total 132.",
        "must_have": ["Shelf F", "Logitech"],
        "should_have": ["132"],
        "type": "llm"
    },
    "What is the average stock per shelf?": {
        "expected": "Average stock per shelf is approximately 154.43 units.",
        "must_have": ["154.43", "average"],
        "should_have": ["units"],
        "type": "llm"
    },
    "Which two shelves have similar stock levels?": {
        "expected": "Shelf A (105) and Shelf D (109) have similar stock levels.",
        "must_have": ["Shelf A", "105", "Shelf D", "109"],
        "should_have": ["similar"],
        "type": "llm"
    },
    "Give me a short comparison of shelves A, B, and C.": {
        "expected": "Shelf A: 105 units (Apple), Shelf B: 326 units (Samsung, highest), Shelf C: 130 units (Sony audio).",
        "must_have": ["Shelf A", "Shelf B", "Shelf C"],
        "should_have": ["105", "326", "130"],
        "type": "llm"
    },
}