import pandas as pd
from tqdm import tqdm
from utils import normalize_string, tokenize_string, get_combination_1_context, get_combination_2_context, get_combination_3_context, get_combination_4_context



def exact_match_rate(message_tokens, memory_tokens):
    """Calculate percentage of memory tokens that appear verbatim in the user message."""
    if not memory_tokens:
        return 1.0  # If no memory tokens
    if not message_tokens:
        return 0.0  # If no message tokens, no match
    
    message_set = set(message_tokens)
    memory_set = set(memory_tokens)
    
    # Count how many memory tokens appear in the message
    matching_tokens = len(memory_set.intersection(message_set))
    
    # Return percentage of memory tokens that match
    return matching_tokens / len(memory_set)


def calculate_overlap(message, memory):
    """Calculate basic overlap metrics between message and memory."""
    # Normalize strings
    norm_message = normalize_string(message)
    norm_memory = normalize_string(memory)
    
    # Tokenize
    message_tokens = tokenize_string(norm_message)
    memory_tokens = tokenize_string(norm_memory)
    
    # Calculate metrics
    exact_match = 1.0 if norm_message == norm_memory else 0.0
    exact_match_rate_score = exact_match_rate(message_tokens, memory_tokens)
    
    return {
        'exact_match': exact_match,
        'exact_match_rate': exact_match_rate_score
    }




def analyze_exact_match(df):
    """Analyze message-memory overlap for the dataset using 4 combinations."""
    # Filter out rows with "No memory"
    df_filtered = df[df['Updated Memory'] != 'No memory'].copy()
    
    print(f"Analyzing {len(df_filtered)} memory entries with 4 combinations...")
    
    # Process by user
    results = []
    sampled_users = list(df_filtered['user_id'].unique())
    
    for user_id in tqdm(sampled_users, desc="Processing users"):
        df_user_memories = df_filtered[df_filtered['user_id'] == user_id]
        
        # Track cross-conversation memories
        cross_conversation_memories = ""
        
        # Process each conversation for this user
        for conversation_id in df_user_memories['conversation_id'].unique():
            df_conv = df_user_memories[df_user_memories['conversation_id'] == conversation_id].sort_index()
            
            # Track conversation context and memories
            conversation_context = ""
            conversation_context_with_memory = ""
            current_memories = ""
            
            for idx, row in df_conv.iterrows():
                memory = row['Updated Memory']
                
                # Build conversation context
                conversation_context = conversation_context + " " + row['User Message']
                conversation_context_with_memory = conversation_context_with_memory + " " + row['User Message']
                
                if memory != "No memory":
                    # Get contexts for all 4 combinations
                    combo1_context = get_combination_1_context(df, idx)
                    combo2_context = get_combination_2_context(df, idx, conversation_id)
                    combo3_context = get_combination_3_context(df, idx, conversation_id)
                    combo4_context = get_combination_4_context(df, idx, conversation_id, user_id, cross_conversation_memories)
                    
                    # Calculate overlaps for each combination
                    combo1_overlap = calculate_overlap(combo1_context, memory)
                    combo2_overlap = calculate_overlap(combo2_context, memory)
                    combo3_overlap = calculate_overlap(combo3_context, memory)
                    combo4_overlap = calculate_overlap(combo4_context, memory)
                    
                    # Combine results
                    overlap_metrics = {
                        'row_index': idx,
                        'conversation_id': conversation_id,
                        'user_id': user_id,
                        'message_id': row.get('message_id', idx),  # Use message_id if available
                        # Combination 1: Current prompt only
                        'combo1_exact_match': combo1_overlap['exact_match'],
                        'combo1_exact_match_rate': combo1_overlap['exact_match_rate'],
                        # Combination 2: Conversation context
                        'combo2_exact_match': combo2_overlap['exact_match'],
                        'combo2_exact_match_rate': combo2_overlap['exact_match_rate'],
                        # Combination 3: Conversation context with memories
                        'combo3_exact_match': combo3_overlap['exact_match'],
                        'combo3_exact_match_rate': combo3_overlap['exact_match_rate'],
                        # Combination 4: Full cross-conversation context
                        'combo4_exact_match': combo4_overlap['exact_match'],
                        'combo4_exact_match_rate': combo4_overlap['exact_match_rate'],
                        # Context lengths for reference
                        'combo1_length': len(combo1_context.split()) if combo1_context else 0,
                        'combo2_length': len(combo2_context.split()) if combo2_context else 0,
                        'combo3_length': len(combo3_context.split()) if combo3_context else 0,
                        'combo4_length': len(combo4_context.split()) if combo4_context else 0
                    }
                    
                    results.append(overlap_metrics)
                
                # Update conversation context with memory
                if memory != "No memory":
                    current_memories = current_memories + " " + memory
                    conversation_context_with_memory = conversation_context_with_memory + " " + memory
            
            # Update cross-conversation memories
            cross_conversation_memories = cross_conversation_memories + " " + current_memories
    
    return pd.DataFrame(results)


def print_summary(exact_match_df):
    """Print simple summary of overlap statistics for all 4 combinations."""
    print("\n" + "="*80)
    print("MESSAGE-MEMORY OVERLAP SUMMARY - 4 COMBINATIONS")
    print("="*80)
    
    print(f"\nDataset: {len(exact_match_df)} memory entries")
    print(f"Average context lengths:")
    print(f"- Combo 1 (current only): {exact_match_df['combo1_length'].mean():.1f} words")
    print(f"- Combo 2 (conversation context): {exact_match_df['combo2_length'].mean():.1f} words")
    print(f"- Combo 3 (conversation context + memories): {exact_match_df['combo3_length'].mean():.1f} words")
    print(f"- Combo 4 (full cross-conversation context): {exact_match_df['combo4_length'].mean():.1f} words")
    
    # Define combination descriptions
    combinations = [
        ("COMBO 1: Current prompt only", "combo1"),
        ("COMBO 2: Conversation context (prior queries + current query)", "combo2"),
        ("COMBO 3: Conversation context with memories (prior queries + generated memories)", "combo3"),
        ("COMBO 4: Full cross-conversation context (prior queries + generated memories across conversations)", "combo4")
    ]
    
    # Print results for each combination
    for desc, prefix in combinations:
        print(f"\n{desc}:")
        print(f"- Exact matches: {exact_match_df[f'{prefix}_exact_match'].mean():.1%}")
        print(f"- Exact match rate: {exact_match_df[f'{prefix}_exact_match_rate'].mean():.1%}")
    


def save_results(exact_match_df, output_dir="../dummy_data/provenance_of_memories/"):
    """Save results to CSV."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "exact_match_results.csv")
    exact_match_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


def main():
    try:
        # Load data
        df_memories = pd.read_csv("../dummy_data/chatgpt_memories_msg_id.csv")

        print("Dataset loaded successfully!")
        print(f"Total rows: {len(df_memories)}")
        print(f"Memory entries: {(df_memories['Updated Memory'] != 'No memory').sum()}")
        
        # Analyze overlap
        exact_match_df = analyze_exact_match(df_memories)
        
        # Print summary
        print_summary(exact_match_df)
        
        # Save results
        save_results(exact_match_df)
        
        return exact_match_df
        
    except FileNotFoundError:
        print("Dataset file not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    main()