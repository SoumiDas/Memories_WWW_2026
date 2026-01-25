import os
import pandas as pd
import json
from openai import OpenAI

# --- Configuration ---
try:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"))
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

PROMPT_3_ATTRIBUTE_MAPPING = """
You are given a list of psychological information.

{psychological_list_as_string}

For each item, identify which attribute it expresses:

- Beliefs: person's model of reality, self-concept, or values.
- Emotions: direct felt experience or mood.
- Aspirations: desired future or specific goal.
- Intentions: reasons for acting or commitment to a plan.

Final instruction: You MUST Return the result strictly in this JSON format:

[
  {{
    "item": "...",
    "attribute": "belief | emotion | aspiration | intention",
    "justification": "short explanation"
  }}
]
"""

PROMPT_4_VERIFICATION = """
I have the following NOTE about a person:
"{memory_entry}"

I also have a psychological PATTERN:
"{psychological_pattern}"

Please evaluate whether the PATTERN directly follows from the NOTE by generating a 1-sentence RATIONALE of your thought process and providing a resulting ANSWER of ONE of the following multiple-choice options, including just the letter: 
    - A: Strongly agree
    - B: Agree
    - C: Neither agree nor disagree
    - D: Disagree
    - E: Strongly disagree

Final instruction: You MUST Return the result strictly in this JSON format:

{{
"rationale": "<rationale>",
"answer": "<answer>"
}}
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

def verify_psychological_pattern(memory_entry: str, psychological_pattern: dict):
    """
    Verifies if the psychological pattern follows from the memory entry using Prompt 4.
    """

    print(f"==========================================================")
    print(f"Verifies psychological pattern:")
    print(f"==========================================================")

    try:
        pattern_str = json.dumps(psychological_pattern, ensure_ascii=False)
    except Exception as e:
        print(f"[ERROR] Could not serialize psychological pattern: {e}")
        return None

    prompt4_filled = PROMPT_4_VERIFICATION.format(
        memory_entry=memory_entry,
        psychological_pattern=pattern_str
    )
    verification_response_str = get_llm_response(prompt4_filled)

    if not verification_response_str:
        print("Halting verification due to error in API call.")
        return None

    try:
        verification_data = json.loads(verification_response_str)
        print("[Verification] Successfully parsed verification data.")
        return verification_data
    except json.JSONDecodeError:
        print(f"[ERROR] Could not parse JSON from verification response:\n{verification_response_str}")
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
    psychological_info = categorization_data.get("psychological_information", [])

    #print(factual_info)
    #print(psychological_info)


    # === STEP 2: Conditional Logic and Follow-up API Calls ===

    # Case: Factual Only (Option A)
    if factual_info and not psychological_info:
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


    # # Case: Psychological Only (Option B)
    # elif psychological_info and not factual_info:
    #     print("\n[Decision] Psychological info found. Running Attribute Mapping (Prompt 3).")
    #     # Format the list of items for the prompt
    #     psych_list_str = "\n".join([f"- {item}" for item in psychological_info])
    #     prompt3_filled = PROMPT_3_ATTRIBUTE_MAPPING.format(psychological_list_as_string=psych_list_str)
    #     attribute_response_str = get_llm_response(prompt3_filled)
    #     if attribute_response_str:
    #         try:
    #             final_results["attribute_mapping"] = json.loads(attribute_response_str)
    #             print("[Step 2] Successfully parsed attribute mapping.")
    #         except json.JSONDecodeError:
    #             final_results["attribute_mapping_raw"] = attribute_response_str
    #             print(f"[Warning] Could not parse JSON from attribute response.")
    
    # Case: Both Factual and Psychological
    elif factual_info and psychological_info:
        print("\n[Decision] Both Factual and Psychological info found. Running both follow-up prompts.")
        
        # --- Run Prompt 2 ---
        print("\n  -> Running GDPR Classification (Prompt 2)...")
        prompt2_filled = PROMPT_2_GDPR_CLASSIFICATION.format(memory_entry=memory_entry)
        gdpr_response_str = get_llm_response(prompt2_filled)
        if gdpr_response_str:
            try:
                final_results["gdpr_analysis"] = json.loads(gdpr_response_str)
                print("  -> Successfully parsed GDPR analysis.")
            except json.JSONDecodeError:
                final_results["gdpr_analysis_raw"] = gdpr_response_str
                print(f"  -> [Warning] Could not parse JSON from GDPR response.")
        
        # --- Run Prompt 3 ---
        # print("\n  -> Running Attribute Mapping (Prompt 3)...")
        # psych_list_str = "\n".join([f"- {item}" for item in psychological_info])
        # prompt3_filled = PROMPT_3_ATTRIBUTE_MAPPING.format(psychological_list_as_string=psych_list_str)
        # attribute_response_str = get_llm_response(prompt3_filled)
        # if attribute_response_str:
        #     try:
        #         final_results["attribute_mapping"] = json.loads(attribute_response_str)
        #         print("  -> Successfully parsed attribute mapping.")
        #     except json.JSONDecodeError:
        #         final_results["attribute_mapping_raw"] = attribute_response_str
        #         print(f"  -> [Warning] Could not parse JSON from attribute response.")

    else:
        print("\n[Decision] No actionable information found in the entry. Halting workflow.")

    return final_results


if __name__ == "__main__":
    
    df_memories = pd.read_csv(".../data/ICL_rephrased_queries_with_memory_info.csv")
    # all the user ids
    user_id_list = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 17, 19, 21, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 51, 53, 54, 55, 56, 57, 59, 60, 61, 63, 64, 65, 68, 70, 71, 72, 73, 74, 75, 76, 77, 78
]
    sampled_users = [
        f'user_{user_id}' for user_id in user_id_list
    ]
    df_memories = df_memories[df_memories["user_id"].isin(sampled_users)]


    for idx, row in df_memories.iterrows():
        print(idx)
        USER_ID = row['user_id']
        conv_id = row['conversation_id']
        text_data = row['original_memory']
        message_id = row['message_id']

        output_file = f".../result/annotation/original_memory/{USER_ID}/{conv_id}/{message_id}.json"

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        try:
            response = process_memory_entry(text_data)
            #print(response)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(response, f, indent=2)

        except Exception as e:
            #print("Check")
            print(f"{output_file}, {e}")
        
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                response_check = json.load(f)
            print(f"Successfully read back the output file: {output_file}")
            verification_results = None
            if response and "attribute_mapping" in response:
                psychological_patterns = response["attribute_mapping"]
                verification_results = verify_psychological_pattern(text_data, psychological_patterns)
                if verification_results:
                    print(f"Verification Results: {verification_results}")
                    response["verification"] = verification_results
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(response, f, indent=2)
        
        except Exception as e:
            print(f"[ERROR] Could not read back the output file: {output_file}. Error: {e}")
        #exit(0)