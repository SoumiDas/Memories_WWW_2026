import os
import pandas as pd
import json
from openai import OpenAI

# --- Configuration ---
try:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY_HERE"))
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    print("Please make sure you have set your OPENAI_API_KEY environment variable or replaced 'YOUR_API_KEY_HERE' in the script.")
    exit()

# --- Prompt Definitions ---

PROMPT_1_CATEGORIZATION = """
You are given a note about a person:

"{memory_entry}"

Task: Separate the information into two categories without interpreting beyond the text.

1. **Factual Information** – External, objective facts explicitly stated in the entry. Examples: name, age, location, job, family members, education, hobbies, explicit events.
2. **Psychological Information** – Internal, subjective states or attitudes explicitly described. Examples: emotions, beliefs, aspirations, intentions.

Final instruction: You MUST Return the result strictly in this JSON format:

{{
  "factual_information": [ "...", "..." ],
  "psychological_information": [ "...", "..." ]
}}
"""

PROMPT_2_GDPR_CLASSIFICATION = """
You are given a note about a person.

"{memory_entry}"

Identify all personal information present in this note, and classify it according to the rules below.

### 1. Main Categories
Classify each item into one of the following main categories for the "category" field:

1.  **personal_data (GDPR Article 4(1))**
    - Examples: name, an identification number, address, phone number, email, IP address, or to one or more factors specific to the physical, physiological, genetic, mental, economic, cultural or social identity of the person.
2.  **special_category_data (GDPR Article 9(1))**
    - Examples: racial or ethnic origin, political opinions, religious/philosophical beliefs, trade union membership, genetic data, biometric data, health data, sex life, sexual orientation.
3.  **non_personal_information**
    - Example: general facts that do not identify a person (e.g., "language preference").

### 2. Specific Sub-Type
You must also populate the "data_type" field based on the following logic:

- **IF** the "category" is `personal_data`, then the "data_type" MUST be one of the following specific types: `name`, `identification number`, `address`, `phone number`, `email`, `IP address`, `physical identity`, `physiological identity`, `genetic identity`, `economic identity`, `cultural identity`, `social identity`.
- **ELSE IF** the "category" is `special_category_data` or `non_personal_information`, the "data_type" MUST be one of the following specific types: `race`, `ethnicity`, `political opinion`, `religion`, `philosophical belief`, `trade union membership`, `genetic data`, `biometric data`, `health data`, `sex life`, `sexual orientation`.
- **ELSE IF** the "category" is `non_personal_information`, the "data_type" MUST be the literal string "NA".

### 3. Citation Rule
For the "citation" field, you MUST provide the exact, verbatim part of the original text that contains the identified information.

### Required Output Format
Final instruction: You MUST Return the result strictly in this JSON format:

[
  {{
    "item": "...",
    "category": "personal_data | special_category_data | non_personal_information",
    "data_type": "name | address | school | NA",
    "justification": "short explanation",
    "citation": "exact text from the note"
  }}
]
"""


def get_llm_response(prompt_text, model="gpt-4o"):
    """A helper function to make a single API call and return the content."""
    print(f"\n--- Sending API Request (Model: {model}) ---")
    try:
        response = client.chat.completions.create(
            model=model,
            
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a helpful assistant that only responds in JSON format."},
                {"role": "user", "content": prompt_text}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"An API error occurred: {e}")
        return None


def process_memory_entry(memory_entry: str):
    """
    Runs the full conditional workflow for a given memory entry.
    """
    print(f"==========================================================")
    print(f"Processing New Memory Entry:\n'{memory_entry}'")
    print(f"==========================================================")
    
    final_results = {}

    # === STEP 1: Categorization (Prompt 1) ===
    print("\n[Step 1] Calling Categorization API (Prompt 1)...")
    #print("Check1")
    try:
        prompt1_filled = PROMPT_1_CATEGORIZATION.format(memory_entry=memory_entry)
    except Exception as e:
        print(f"[ERROR] Could not format Prompt 1: {e}")
        return None
    print(prompt1_filled)
    #print("Check2")
    categorization_response_str = get_llm_response(prompt1_filled)

    if not categorization_response_str:
        print("Halting workflow due to error in Step 1.")
        return None

    try:
        categorization_data = json.loads(categorization_response_str)
        final_results["categorization"] = categorization_data
        print("[Step 1] Successfully parsed categorization data.")
    except json.JSONDecodeError:
        print(f"[ERROR] Could not parse JSON from Step 1 response:\n{categorization_response_str}")
        return None
    
    factual_info = categorization_data.get("factual_information", [])

    # === STEP 2: GDPR Classification (Prompt 2) ===

    # Case: Factual Only (Option A)
    if factual_info:
        print("\n[Decision] Factual info found. Running GDPR Classification (Prompt 2).")
        prompt2_filled = PROMPT_2_GDPR_CLASSIFICATION.format(memory_entry=memory_entry)
        gdpr_response_str = get_llm_response(prompt2_filled)
        if gdpr_response_str:
            try:
                final_results["gdpr_analysis"] = json.loads(gdpr_response_str)
                print("[Step 2] Successfully parsed GDPR analysis.")
            except json.JSONDecodeError:
                final_results["gdpr_analysis_raw"] = gdpr_response_str
                print(f"[Warning] Could not parse JSON from GDPR response.")
    else:
        print("\n[Decision] No factual information found in the entry. Halting workflow.")

    return final_results


if __name__ == "__main__":
    
    df_memories = pd.read_csv("../dummy_data/chatgpt_memories_msg_id.csv")

    for idx, row in df_memories.iterrows():
        print(idx)
        USER_ID = row['user_id']
        CONV_ID = row['conversation_id']
        TEXT_DATA = row['Updated Memory']
        MEMORY_ID = row['memory_id']

        output_file = f"../dummy_data/gdpr_annotated_data/{USER_ID}/{CONV_ID}/{MEMORY_ID}.json"

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        try:
            response = process_memory_entry(TEXT_DATA)
            #print(response)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(response, f, indent=2)

        except Exception as e:
            print(f"[ERROR] Error processing entry {MEMORY_ID}: {e}")
            print(f"Output file: {output_file}")
        #exit(0)