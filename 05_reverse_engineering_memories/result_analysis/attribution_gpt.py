import os
import json
from openai import OpenAI
import time

# --- Configuration ---
try:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY_HERE"))
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    print("Please make sure you have set your OPENAI_API_KEY environment variable or replaced 'YOUR_API_KEY_HERE' in the script.")
    exit()

# --- Prompt Definitions ---

PROMPT_PART3 = """
This is a conversation:

Query A: {text_data[0]}

Query B: {text_data[1]}
Which of them is more privacy preserving? We define privacy preserving in terms of less attribution to personal actions.
---

### Required Answer Format

Your final output MUST be a single, valid JSON object in this exact format. Do not add any text or explanation outside of the JSON. Choose ONLY one query that seems privacy preserving to you based on the above definition. Also provide 1-2 line explanation for your choice.

{{
    "Privacy preserving query": {{ Query A or Query B
    }},
    "Explanation": {{...
    }}
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

def process_file_entry(text_data: str):
    """
    Runs the full workflow for a given conversation entry.
    """
    print(f"==========================================================")
    print(f"Processing New Memory Entry:\n'{text_data}'")
    print(f"==========================================================")
    
    #final_results = {}

    # === STEP 1: Categorization (Prompt 1) ===
    print("\n[Step 1] Calling API (Prompt Part3)...")
    print("Check1")
    try:
        prompt_filled = PROMPT_PART3.format(text_data=text_data)
    except Exception as e:
        print(f"[ERROR] Could not format Prompt Part3: {e}")
        return None
    #print(prompt_filled)
    print("Check2")
    response_str = get_llm_response(prompt_filled)

    if not response_str:
        print("Halting workflow due to error in Step 1.")
        return None

    try:
        resultInfo = json.loads(response_str)
        
    except json.JSONDecodeError:
        print(f"[ERROR] Could not parse JSON from Step 1 response:\n{response_str}")
        return None

    return resultInfo


if __name__ == "__main__":

    query_dict = [("I've been off my medication, which is 20mg of methylphenidate, for around 2-3 weeks. I've just started back on it today. Can you tell me what effects I might experience and how long they may last?","What are the potential effects and duration of restarting a medication after a period of not taking it?"),("I really need to quit smoking cannabis. It's sapping my motivation, money and causes me to eat shit junk food and stay fat. I know it's something I need to do but in afraid of all the time alone clear headed","What are the common challenges and strategies for quitting cannabis, particularly in terms of motivation, finances, and diet?"),("I know a part of it is that I feel fat and out of shape, my diet is poor but I'm trying to be better now I'm working. Would more exercise help?","How can lifestyle changes such as diet and exercise impact sex drive and overall health?")]
    
    output_file = f".../results/Attribution/result.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)


    responses = []
    for query in query_dict:

        try:
            responses.append(process_file_entry(query))
        except Exception as e:
            print("Error")
        
    with open(output_file, "a", encoding="utf-8") as f:
        json.dump(responses, f, indent=2)
        
    '''for suser in sampled_users:
        start = time.time()
        print(suser)
        USER_ID = suser
        df_user_memories = df_memories[df_memories["user_id"]==suser]
        df_grouped = df_user_memories.groupby("conversation_id")["User Message"].apply(list).reset_index()
        df_grouped["all_messages"] = df_grouped["User Message"].apply(lambda msgs: "\n".join(f"User A: {msg}" for msg in msgs))
        print(len(df_grouped["conversation_id"].unique()))
        conv_ids = list(df_grouped["conversation_id"].unique())
        print(conv_ids)

        for conv_id in conv_ids:
            print("Now working on "+str(conv_id))
            text_data = df_grouped[df_grouped["conversation_id"] == conv_id]["all_messages"].iloc[0]
            #print(text_data)
            output_file = f"/NS/chatgpt/work/Soumi/Conversations_Users_Remain/{USER_ID}/{conv_id}-finalise.json"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            try:
                response = process_file_entry(text_data)
            except Exception as e:
                print(f"{output_file}, {e}")
        
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(response, f, indent=2)

        print(f"{suser} done in {time.time()-start} seconds")'''
