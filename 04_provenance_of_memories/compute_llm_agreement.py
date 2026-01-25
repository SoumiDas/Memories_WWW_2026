"""
Model-based Analysis for Context-Memory Agreement

Evaluate whether a generated memory logically follows from 
the user context using a agreement rating.

"""

import pandas as pd
import json
import os
from typing import Optional
from tqdm import tqdm
import time
import openai
from openai import OpenAI
from dotenv import load_dotenv



class ModelBasedAnalyzer:
    """Analyzer for evaluating context-memory agreement using GPT-4o."""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o", max_tokens: int = 500):
        """
        Initialize the model-based analyzer.
        
        Args:
            api_key: OpenAI API key (if None, will try to get from environment)
            model: Model to use for evaluation
            max_tokens: Maximum tokens for model responses
        """
        self.model = model
        self.max_tokens = max_tokens
        
        # Initialize OpenAI client
        if api_key:
            openai.api_key = api_key
        else:
            load_dotenv("../.env")
            openai.api_key = os.getenv('OPENAI_API_KEY')
            
        if not openai.api_key:
            raise ValueError("OpenAI API key not provided.")

        self.client =  OpenAI(api_key=openai.api_key)
    
    def get_user_messages(self, df: pd.DataFrame, current_idx: int, conversation_id: str) -> str:
        """Get user messages from conversation up to current index."""
        conversation_messages = df[
            (df['conversation_id'] == conversation_id) & 
            (df.index <= current_idx)
        ].sort_index()
        
        if len(conversation_messages) == 0:
            return ""
        
        context_messages = conversation_messages['User Message'].tolist()
        return " ".join(context_messages)
    
    def get_past_memories(self, df: pd.DataFrame, current_idx: int, conversation_id: str) -> str:
        """Get past memories from conversation up to current index."""
        previous_memories = df[
            (df['conversation_id'] == conversation_id) & 
            (df.index < current_idx) &
            (df['Updated Memory'] != 'No memory')
        ]['Updated Memory'].tolist()
        
        return " ".join(previous_memories) if previous_memories else ""
    
    def get_combination_1_context(self, df: pd.DataFrame, current_idx: int) -> str:
        """Combination 1: Current prompt only."""
        current_row = df.loc[current_idx]
        return current_row['User Message']
    
    def get_combination_2_context(self, df: pd.DataFrame, current_idx: int, conversation_id: str) -> str:
        """Combination 2: Conversation context (prior queries + current query)."""
        return self.get_user_messages(df, current_idx, conversation_id)
    
    def get_combination_3_context(self, df: pd.DataFrame, current_idx: int, conversation_id: str) -> str:
        """Combination 3: Conversation context with memories (prior queries + generated memories)."""
        user_messages = self.get_user_messages(df, current_idx, conversation_id)
        past_memories = self.get_past_memories(df, current_idx, conversation_id)
        
        if not past_memories:
            return user_messages
        
        return user_messages + " " + past_memories
    
    def get_combination_4_context(self, df: pd.DataFrame, current_idx: int, conversation_id: str, 
                                 user_id: str, cross_conversation_memories: str = "") -> str:
        """Combination 4: Full cross-conversation context."""
        combo3_context = self.get_combination_3_context(df, current_idx, conversation_id)
        
        if not cross_conversation_memories:
            return combo3_context
        
        return cross_conversation_memories + " " + combo3_context
    
    def create_evaluation_prompt(self, context: str, memory: str, past_memories: str = "") -> str:
        """Create a prompt for evaluating context-memory agreement."""
        
        # Build the prompt with clear separation between user context and past memories
        prompt_parts = [
            """You are evaluating whether a stored memory entry about a user can be logically derived from the user's past conversation(s).
Your goal is to judge how strongly the memory is grounded in or logically follows from the user's conversation, without introducing assumptions or speculative reasoning.

Consider only what the user explicitly said or clearly implied in the provided user messages (and past memories if provided).""",
            "",
            "**USER MESSAGES:**",
            context,
            ""
        ]
        
        # Add past memories section only if they exist
        if past_memories and past_memories.strip():
            prompt_parts.extend([
                "**PREVIOUS MEMORIES:**",
                past_memories,
                ""
            ])
        
        prompt_parts.extend([
            "**GENERATED MEMORY:**",
            memory,
            "",
            """Evaluation Scale (5-point):

- 5 (Directly Stated): The memory exactly restates something the user explicitly said.
- 4 (Paraphrased): The memory rephrases or condenses information that is clearly present in the conversation.
- 3 (Logically Inferred): The memory is not stated verbatim but can be reasonably inferred from the conversation.
- 2 (Weakly Supported): The memory could be loosely consistent with the conversation but lacks clear grounding. Inference is speculative or uncertain.
- 1 (Unsupported): The memory is unsupported or contradicted by the user's conversation.

Your task:
- Assign a score from 1–5 on this scale.
- Quote or paraphrase the specific parts of the conversation that justify your judgment.
- Briefly explain your reasoning.

Only respond with strict JSON with the following format (no additional text):
{{
  "rating": <integer from 1 to 5>,
  "classification": "<text label corresponding to rating>",
  "justification": "<quote or paraphrase of user text that supports your decision>",
  "reasoning": "<short explanation of why the memory deserves this rating>"
}}

IMPORTANT: Respond ONLY with the JSON object above. Do not include any other text."""
        ])
        
        return "\n".join(prompt_parts)

    def evaluate_context_memory_agreement(self, context: str, memory: str, past_memories: str = "") -> dict:
        """
        Evaluate whether memory follows from context using GPT-4o.
        
        Args:
            context: User context string
            memory: Generated memory string
            past_memories: Past memories string (optional)
            
        Returns:
            Dictionary with agreement_score and reasoning
        """
        try:
            prompt = self.create_evaluation_prompt(context, memory, past_memories)
            
            response = self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are an expert evaluator. Always respond with valid JSON format as requested."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=0
            )
            
            # Parse the JSON response
            result_text = response.choices[0].message.content.strip()
            
            try:
                # Clean the response text
                result_text = result_text.strip()
                
                # Find JSON in the response
                start_idx = result_text.find('{')
                end_idx = result_text.rfind('}') + 1
                
                if start_idx == -1 or end_idx == 0:
                    raise ValueError("No JSON object found in response")
                
                json_text = result_text[start_idx:end_idx].strip()
                result_dict = json.loads(json_text)
                
                # Extract rating and convert to int
                rating = result_dict.get('rating')
                if rating is None:
                    raise KeyError("'rating' field not found in response")
                
                agreement_score = int(rating)
                if not (1 <= agreement_score <= 5):
                    raise ValueError(f"Rating {agreement_score} is not in valid range 1-5")
                
                return {
                    'agreement_score': agreement_score,
                    'classification': result_dict.get('classification', ''),
                    'reasoning': result_dict.get('reasoning', '')
                }
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"Error parsing model response: {e}")
                print(f"Raw response: {result_text}")
                return {
                    'agreement_score': 0,  # Default
                    'classification': 'Error',
                    'reasoning': f"Error parsing response: {e}"
                }
                
        except Exception as e:
            print(f"Error in model evaluation: {e}")
            return {
                'agreement_score': 0,  # Default
                'classification': 'Error',
                'reasoning': f"Error: {e}"
            }
    
    def analyze_agreement(self, df: pd.DataFrame, max_samples: Optional[int] = None, 
                         delay_between_requests: float = 1.0) -> pd.DataFrame:
        """
        Analyze context-memory agreement using model evaluation.
        
        Args:
            df: DataFrame with conversation data
            max_samples: Maximum number of samples to process (for testing)
            delay_between_requests: Delay between API requests in seconds
            
        Returns:
            DataFrame with evaluation results
        """
        # Filter out rows with "No memory"
        df_filtered = df[df['Updated Memory'] != 'No memory'].copy()
        
        if max_samples:
            df_filtered = df_filtered.head(max_samples)
        
        print(f"Analyzing {len(df_filtered)} memory entries with model-based agreement evaluation...")
        
        results = []
        api_calls_saved = 0
        
        for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Processing"):
            memory = row['Updated Memory']
            conversation_id = row['conversation_id']
            user_id = row['user_id']
            
            # Get contexts for all combinations
            combo1_context = self.get_combination_1_context(df, idx)
            combo2_context = self.get_combination_2_context(df, idx, conversation_id)
            
            # For combo3, get user messages and past memories separately
            combo3_user_messages = self.get_user_messages(df, idx, conversation_id)
            combo3_past_memories = self.get_past_memories(df, idx, conversation_id)
            
            # Evaluate each combination, avoiding redundant API calls
            combo1_result = self.evaluate_context_memory_agreement(combo1_context, memory)
            time.sleep(delay_between_requests)
            
            # Check if combo2 is same as combo1
            if combo2_context == combo1_context:
                combo2_result = combo1_result
                api_calls_saved += 1
            else:
                combo2_result = self.evaluate_context_memory_agreement(combo2_context, memory)
                time.sleep(delay_between_requests)
            
            # For combo3, check if user messages are same as combo1 or combo2
            if combo3_user_messages == combo1_context:
                combo3_result = combo1_result
                api_calls_saved += 1
            elif combo3_user_messages == combo2_context:
                combo3_result = combo2_result
                api_calls_saved += 1
            else:
                combo3_result = self.evaluate_context_memory_agreement(combo3_user_messages, memory, combo3_past_memories)
                time.sleep(delay_between_requests)
            
            # Store results
            result = {
                'row_index': idx,
                'conversation_id': conversation_id,
                'user_id': user_id,
                'message_id': row.get('message_id', idx),
                'memory': memory,
                
                # Agreement scores for each combination
                'combo1_agreement_score': combo1_result['agreement_score'],
                'combo1_classification': combo1_result.get('classification', ''),
                'combo1_reasoning': combo1_result['reasoning'],
                'combo1_context_length': len(combo1_context.split()) if combo1_context else 0,
                'combo1_user_messages': combo1_context,
                'combo1_past_memories': '',  # Combo1 only uses current message
                
                'combo2_agreement_score': combo2_result['agreement_score'],
                'combo2_classification': combo2_result.get('classification', ''),
                'combo2_reasoning': combo2_result['reasoning'],
                'combo2_context_length': len(combo2_context.split()) if combo2_context else 0,
                'combo2_user_messages': combo2_context,
                'combo2_past_memories': '',  # Combo2 only uses user messages
                
                'combo3_agreement_score': combo3_result['agreement_score'],
                'combo3_classification': combo3_result.get('classification', ''),
                'combo3_reasoning': combo3_result['reasoning'],
                'combo3_context_length': len(combo3_user_messages.split()) + len(combo3_past_memories.split()) if combo3_user_messages or combo3_past_memories else 0,
                'combo3_user_messages': combo3_user_messages,
                'combo3_past_memories': combo3_past_memories,
                    }
            
            results.append(result)
        
        print(f"\nAPI calls saved: {api_calls_saved} (out of {len(df_filtered) * 3} total possible calls)")
        return pd.DataFrame(results)
    
    def print_summary(self, results_df: pd.DataFrame):
        """Print summary of model-based agreement evaluation results."""
        print("\n" + "="*60)
        print("MODEL-BASED CONTEXT-MEMORY AGREEMENT ANALYSIS")
        print("="*60)
        
        print(f"\nDataset: {len(results_df)} memory entries")
        print(f"Average context lengths:")
        print(f"- Combo 1 (current only): {results_df['combo1_context_length'].mean():.1f} words")
        print(f"- Combo 2 (conversation context): {results_df['combo2_context_length'].mean():.1f} words")
        print(f"- Combo 3 (context + memories): {results_df['combo3_context_length'].mean():.1f} words")
        
        combinations = [
            ("COMBO 1: Current prompt only", "combo1"),
            ("COMBO 2: Conversation context", "combo2"),
            ("COMBO 3: Context + memories", "combo3"),
        ]
        
        # Print results for each combination
        for desc, prefix in combinations:
            agreement_mean = results_df[f'{prefix}_agreement_score'].mean()
            agreement_std = results_df[f'{prefix}_agreement_score'].std()
            
            print(f"\n{desc}:")
            print(f"- Agreement Score: {agreement_mean:.2f} ± {agreement_std:.2f} (1-5 scale)")
            
            # Show distribution
            score_counts = results_df[f'{prefix}_agreement_score'].value_counts().sort_index()
            print(f"- Score Distribution:")
            for score in range(1, 6):
                count = score_counts.get(score, 0)
                percentage = (count / len(results_df)) * 100
                print(f"  * {score}: {count} ({percentage:.1f}%)")
        
        # Improvement analysis
        print(f"\nIMPROVEMENT ANALYSIS:")
        base_agreement = results_df['combo1_agreement_score'].mean()
        
        for i, (desc, prefix) in enumerate(combinations[1:], 2):
            improvement = results_df[f'{prefix}_agreement_score'].mean() - base_agreement
            print(f"- Combo {i} vs Combo 1: {improvement:+.2f}")
        
        # Best performing combination
        best_combo = None
        best_score = -1
        for prefix in ['combo1', 'combo2', 'combo3']:
            score = results_df[f'{prefix}_agreement_score'].mean()
            if score > best_score:
                best_score = score
                best_combo = prefix
        
        print(f"\nBEST PERFORMING COMBINATION: {best_combo.upper()} (Agreement Score: {best_score:.2f})")
    
    def save_results(self, results_df: pd.DataFrame, output_dir: str = "../dummy_data/provenance_of_memories/"):
        """Save results to CSV and JSON."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save CSV
        csv_path = os.path.join(output_dir, "model_based_agreement_results.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        
        # Save detailed JSON with reasoning
        json_path = os.path.join(output_dir, "model_based_agreement_results_detailed.json")
        detailed_results = []
        
        for _, row in results_df.iterrows():
            detailed_result = {
                'row_index': row['row_index'],
                'conversation_id': row['conversation_id'],
                'user_id': row['user_id'],
                'message_id': row['message_id'],
                'memory': row['memory'],
                'combinations': {}
            }
            
            for combo in ['combo1', 'combo2', 'combo3']:
                detailed_result['combinations'][combo] = {
                    'agreement_score': row[f'{combo}_agreement_score'],
                    'classification': row[f'{combo}_classification'],
                    'reasoning': row[f'{combo}_reasoning'],
                    'context_length': row[f'{combo}_context_length'],
                    'user_messages': row[f'{combo}_user_messages'],
                    'past_memories': row[f'{combo}_past_memories']
                }
            
            detailed_results.append(detailed_result)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        print(f"Detailed results saved to: {json_path}")


def main():
    """Main function to run the simplified model-based agreement analysis."""
    try:
        # Initialize analyzer
        analyzer = ModelBasedAnalyzer(
            model="gpt-4o",
            max_tokens=500,
        )
        
        # Load data
        df_memories = pd.read_csv("../dummy_data/chatgpt_memories_msg_id.csv")
        # sampled_users = ["user_0"] # FOR TESTING, we can use a subset of users
        # df_memories = df_memories[df_memories["user_id"].isin(sampled_users)]
        
        print("Dataset loaded successfully!")
        print(f"Total rows: {len(df_memories)}")
        print(f"Memory entries: {(df_memories['Updated Memory'] != 'No memory').sum()}")
        
        # Analyze agreement
        print("\nStarting model-based agreement analysis...")
        
        results_df = analyzer.analyze_agreement(
            df_memories, 
            # max_samples=5,  # FOR TESTING, we can use a smaller sample
            delay_between_requests=1.0
        )
        
        # Print summary
        analyzer.print_summary(results_df)
        
        # Save results
        analyzer.save_results(results_df)
        
        return results_df
        
    except FileNotFoundError:
        print("Dataset file not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")    
        return None


if __name__ == "__main__":
    main()
