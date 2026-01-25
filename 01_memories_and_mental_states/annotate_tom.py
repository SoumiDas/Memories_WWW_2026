import os
import pandas as pd
import json
import argparse
from openai import OpenAI
from tqdm import tqdm
# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument("--model_name", type=str, default="gpt-4o", help="OpenAI model name to use (default: gpt-4o)")

args = argparser.parse_args()
MODEL_NAME = args.model_name

print(f"Model Name: {MODEL_NAME}")

# --- Configuration ---
try:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY_HERE"))
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    print("Please make sure you have set your OPENAI_API_KEY environment variable or replaced 'YOUR_API_KEY_HERE' in the script.")
    exit()

# --- Prompt Definitions ---

PROMPT_UNIFIED_CATEGORIZATION = """
You are given a text snippet. Your task is to determine whether it contains explicit Theory of Mind (ToM) content and, if so, which categories are present.
A snippet contains explicit Theory of Mind (ToM) content if it references internal mental states of a person.

ToM categories:
  * Belief: references what a person believes or their model of reality, self-concept, or values
  * Desire: references what a person wants, wishes, or prefers or their goals
  * Intention: references what a person explicitly intends or commits to doing (must indicate a mental commitment, not just a behavioral plan)
  * Emotion: references what a person feels emotionally (e.g., sadness, frustration, excitement, fascination)
  * Percept: references how a person subjectively experiences or perceives things (e.g., "feels like", "seems to", "experiences as")
  * Knowledge: makes inferences about what a person knows or does not know based on their access to information (what they perceived or were told)
  * Mentalistic: contains a non-literal phrase (metaphors, irony, sarcasm, idioms, etc.)

Important: 
Do not infer mental states unless they are explicitly stated or linguistically implied (e.g., "feels", "believes", "wants", "is motivated", "is anxious", etc.). Phrases like "plans to", "is doing", or "went to" must involve a clearly expressed internal stance to count as ToM.

The text snippet is:
{memory_entry}

Classify whether the snippet contains ToM content. If ToM is present, mark all applicable categories and give a short justification for each true category.
"""

PROMPT_VERIFICATION = """
You are verifying whether proposed Theory of Mind (ToM) categories are directly supported by a NOTE about a person.

NOTE:
"{memory_entry}"

PROPOSED CATEGORIES:
"{psychological_pattern}"

For EACH category listed, decide whether the NOTE directly supports it based only on explicit or clearly implied language. Do NOT infer missing mental states or intentions.

Answer choices:
- A: Strongly agree
- B: Agree
- C: Neither agree nor disagree
- D: Disagree
- E: Strongly disagree

For each category, return your answer choice and a brief rationale for your answer.
"""

# JSON Schema for structured outputs - Categorization
CLASSIFICATION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "tom_classification_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "ToM": {"type": "boolean"},
                "belief": {"type": "boolean"},
                "belief_rationale": {"type": "string"},
                "desire": {"type": "boolean"},
                "desire_rationale": {"type": "string"},
                "intention": {"type": "boolean"},
                "intention_rationale": {"type": "string"},
                "emotion": {"type": "boolean"},
                "emotion_rationale": {"type": "string"},
                "percept": {"type": "boolean"},
                "percept_rationale": {"type": "string"},
                "knowledge": {"type": "boolean"},
                "knowledge_rationale": {"type": "string"},
                "mentalistic": {"type": "boolean"},
                "mentalistic_rationale": {"type": "string"}
            },
            "required": ["ToM", "belief", "belief_rationale", "desire", "desire_rationale", 
                        "intention", "intention_rationale", "emotion", "emotion_rationale",
                        "percept", "percept_rationale", "knowledge", "knowledge_rationale",
                        "mentalistic", "mentalistic_rationale"],
            "additionalProperties": False
        }
    }
}

# JSON Schema for structured outputs - Verification
VERIFICATION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "verification_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "enum": ["belief", "emotion", "desire", "intention", "percept", "knowledge", "mentalistic"]
                            },
                            "rationale": {"type": "string"},
                            "answer": {
                                "type": "string",
                                "enum": ["A", "B", "C", "D", "E"]
                            }
                        },
                        "required": ["category", "rationale", "answer"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["results"],
            "additionalProperties": False
        }
    }
}

def get_llm_response(prompt_text, json_schema=None, model=None):
    """A helper function to make a single API call and return the content."""
    if model is None:
        model = MODEL_NAME
    print(f"\n--- Sending API Request (Model: {model}) ---")
    try:
        response = client.chat.completions.create(
            model=model,
            response_format=json_schema if json_schema else {"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a helpful assistant that responds in JSON format."},
                {"role": "user", "content": prompt_text}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"An API error occurred: {e}")
        return None

def verify_psychological_pattern(memory_entry: str, psychological_pattern: dict):
    """
    Verifies if the ToM category patterns follow from the memory entry.
    Handles all ToM categories including belief, desire, intention, emotion, percept, knowledge, and mentalistic.
    """
    print(f"==========================================================")
    print(f"Verifying ToM categories:")
    print(f"==========================================================")

    try:
        pattern_str = json.dumps(psychological_pattern, ensure_ascii=False)
    except Exception as e:
        print(f"[ERROR] Could not serialize psychological pattern: {e}")
        return None

    prompt_filled = PROMPT_VERIFICATION.format(
        memory_entry=memory_entry,
        psychological_pattern=pattern_str
    )
    
    print(f"\n[Verification] Sending request to LLM...")
    verification_response_str = get_llm_response(prompt_filled, json_schema=VERIFICATION_SCHEMA)

    if not verification_response_str:
        print("[ERROR] Halting verification due to error in API call.")
        return None

    try:
        verification_data = json.loads(verification_response_str)
        print("[Verification] Successfully parsed verification data.")
        return verification_data
    except json.JSONDecodeError as e:
        print(f"[ERROR] Could not parse JSON from verification response:\n{verification_response_str}")
        print(f"JSON decode error: {e}")
        return None

def process_memory_entry(memory_entry: str):
    """
    Runs the full conditional workflow for a given memory entry.
    """
    print(f"==========================================================")
    print(f"Processing New Memory Entry:\n'{memory_entry}'")
    print(f"==========================================================")
    
    # === STEP 1: Unified Categorization ===
    print("\n[Step 1] Calling Unified Categorization...")
    try:
        prompt_filled = PROMPT_UNIFIED_CATEGORIZATION.format(memory_entry=memory_entry)
    except Exception as e:
        print(f"[ERROR] Could not format unified prompt: {e}")
        return None
    
    categorization_response_str = get_llm_response(prompt_filled, json_schema=CLASSIFICATION_SCHEMA)

    if not categorization_response_str:
        print("Halting workflow due to error in Step 1.")
        return None

    try:
        categorization_data = json.loads(categorization_response_str)
        print("[Step 1] Successfully parsed categorization data.")
        
        # Validate that ToM field is present
        if "ToM" not in categorization_data:
            print("[WARNING] ToM field missing from response, but continuing...")
        
        return categorization_data
    except json.JSONDecodeError as e:
        print(f"[ERROR] Could not parse JSON from Step 1 response:\n{categorization_response_str}")
        print(f"JSON decode error: {e}")
        return None



def save_categorization_result(response: dict, output_file: str):
    """
    Save categorization result to JSON file.
    
    Args:
        response: The categorization response dictionary
        output_file: Path to the output file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(response, f, indent=2)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save categorization result to {output_file}: {e}")
        return False


def is_tom_true(tom_value):
    """
    Check if ToM value is True, handling both boolean and string formats.
    
    Args:
        tom_value: ToM value (can be bool or str)
    
    Returns:
        True if ToM is true, False otherwise
    """
    if isinstance(tom_value, bool):
        return tom_value
    elif isinstance(tom_value, str):
        return "yes" in tom_value.lower()
    return False


def extract_tom_categories(response_check: dict):
    """
    Extract all ToM categories that are True from the response.
    
    Args:
        response_check: The categorization response dictionary
    
    Returns:
        List of category dictionaries with 'category' and 'justification' keys
    """
    tom_categories = []
    category_fields = ["belief", "desire", "intention", "emotion", "percept", "knowledge", "mentalistic"]
    
    for category in category_fields:
        if response_check.get(category) is True:
            category_data = {
                "category": category,
                "justification": response_check.get(f"{category}_rationale", "")
            }
            tom_categories.append(category_data)
    
    return tom_categories


def add_verification_to_response(response_check: dict, memory_entry: str, memory_id: str):
    """
    Add verification results to the response if ToM is true.
    
    Args:
        response_check: The categorization response dictionary
        memory_entry: The original memory text
        memory_id: Memory ID for error reporting
    
    Returns:
        True if verification was added, False otherwise
    """
    if not response_check or "ToM" not in response_check:
        return False
    
    tom_value = response_check["ToM"]
    tom_is_true = is_tom_true(tom_value)
    
    if not tom_is_true:
        return False
    
    # Extract ToM categories
    tom_categories = extract_tom_categories(response_check)
    
    if not tom_categories:
        return False
    
    # Run verification
    tom_verification = verify_psychological_pattern(memory_entry, tom_categories)
    
    if tom_verification:
        print(f"ToM Verification Results: {tom_verification}")
        if "verification" not in response_check:
            response_check["verification"] = {}
        response_check["verification"]["ToM_categories"] = tom_verification
        return True
    else:
        print(f"[Warning] ToM verification failed for entry {memory_id}")
        return False


def process_single_memory(row, output_file: str):
    """
    Process a single memory entry: categorize and verify.
    
    Args:
        row: DataFrame row containing memory data
        output_file: Path to the output JSON file
    
    Returns:
        True if processing was successful, False otherwise
    """
    user_id = row['user_id']
    conv_id = row['conversation_id']
    text_data = row['Updated Memory']
    memory_id = row['memory_id']
    
    # Step 1: Categorize memory entry
    try:
        response = process_memory_entry(text_data)
        if response is None:
            print(f"[ERROR] Failed to categorize memory {memory_id}")
            return False
        
        # Save categorization result
        if not save_categorization_result(response, output_file):
            return False
        
    except Exception as e:
        print(f"[ERROR] Error processing memory {memory_id} at {output_file}: {e}")
        return False
    
    # Step 2: Verify ToM categories if applicable
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            response_check = json.load(f)
        print(f"Successfully read back the output file: {output_file}")
        
        verification_added = add_verification_to_response(response_check, text_data, memory_id)
        
        # Save updated response with verification results if any were added
        if verification_added:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(response_check, f, indent=2)
            print(f"Verification results added and saved to: {output_file}")
        
        return True
        
    except FileNotFoundError:
        print(f"[Warning] Could not read output file for verification: {output_file}")
        return False
    except json.JSONDecodeError as e:
        print(f"[ERROR] Could not parse JSON from output file: {output_file}. Error: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Error during verification for entry {memory_id}: {e}")
        return False


if __name__ == "__main__":

    BASE_DIR = "../dummy_data"
    # Step 1: Load and filter memories
    csv_path = f"{BASE_DIR}/chatgpt_memories_msg_id.csv"
    df_memories = pd.read_csv(csv_path)
    
    # Step 2: Process each memory entry
    for idx, row in tqdm(df_memories.iterrows(), total=len(df_memories)):
        print(f"Processing row {idx}")
        
        # Generate output file path
        output_file = f"{BASE_DIR}/tom_annotated_data/{row['user_id']}/{row['conversation_id']}/{row['memory_id']}.json"
        
        # Process memory (categorization + verification)
        process_single_memory(row, output_file)