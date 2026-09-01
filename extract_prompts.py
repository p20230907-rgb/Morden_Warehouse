#!/usr/bin/env python3
"""
extract_prompts.py

Reads newapp.py and extracts the actual prompt strings for the LLM chains.
Outputs prompt_templates.json with full content.
"""

import re
import json
import sys

def extract_prompt_from_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    prompts = {}

    # Search for ChatPromptTemplate.from_messages definitions.
    # We look for patterns like:
    #   rag_llm_prompt = ChatPromptTemplate.from_messages([
    #       ("system", "You are a helpful warehouse..."),
    #       ("human", "Context: {context}\n\nQuestion: {query}")
    #   ])
    # We'll capture the variable name and the list content.

    # Pattern: variable_name = ChatPromptTemplate.from_messages([ ... ])
    pattern = r'(\w+)\s*=\s*ChatPromptTemplate\.from_messages\(\s*\[([\s\S]*?)\]\s*\)'

    matches = re.findall(pattern, content)
    for var_name, body in matches:
        # Now parse the body to extract system and human messages
        # The body is a Python list of tuples: ("role", "text")
        # We'll extract all tuples.
        tuple_pattern = r'\(\s*"([^"]+)"\s*,\s*"([^"]*)"\s*\)'
        tuples = re.findall(tuple_pattern, body)
        if tuples:
            prompts[var_name] = tuples

    return prompts

if __name__ == "__main__":
    if len(sys.argv) > 1:
        newapp_file = sys.argv[1]
    else:
        newapp_file = "newapp.py"

    try:
        prompts = extract_prompt_from_file(newapp_file)
        with open("prompt_templates.json", "w") as f:
            json.dump(prompts, f, indent=2)
        print(f"✅ Extracted prompts to prompt_templates.json")
    except Exception as e:
        print(f"❌ Error: {e}")