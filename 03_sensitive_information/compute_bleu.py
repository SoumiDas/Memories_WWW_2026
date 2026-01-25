import pandas as pd
from tqdm import tqdm
import evaluate
from utils import get_combination_1_context, get_combination_2_context, get_combination_3_context, get_combination_4_context

# Load BLEU metric
_bleu_metric = evaluate.load('bleu')

def calculate_bleu_unigram_precision(reference, candidate):
    """Calculate BLEU unigram precision between reference and candidate."""
    results = _bleu_metric.compute(predictions=[candidate], references=[[reference]])
    return results['precisions'][0]


def analyze_bleu(df):
    """Analyze message-memory overlap for the dataset using 4 combinations."""
    df_filtered = df[df['Updated Memory'] != 'No memory'].copy()
    print(f"Analyzing {len(df_filtered)} memory entries with 4 combinations...")
    
    results = []
    cross_conversation_memories = {}  # Track per user
    
    for user_id in tqdm(df_filtered['user_id'].unique(), desc="Processing users"):
        user_df = df_filtered[df_filtered['user_id'] == user_id]
        cross_conversation_memories[user_id] = ""
        
        for conversation_id in user_df['conversation_id'].unique():
            conv_df = user_df[user_df['conversation_id'] == conversation_id].sort_index()
            current_memories = []
            
            for idx, row in conv_df.iterrows():
                memory = row['Updated Memory']
                
                # Get contexts for all 4 combinations
                contexts = {
                    'combo1': get_combination_1_context(df, idx),
                    'combo2': get_combination_2_context(df, idx, conversation_id),
                    'combo3': get_combination_3_context(df, idx, conversation_id),
                    'combo4': get_combination_4_context(df, idx, conversation_id, user_id, 
                                                         cross_conversation_memories[user_id])
                }
                
                # Calculate BLEU scores and lengths
                result = {
                    'row_index': idx,
                    'conversation_id': conversation_id,
                    'user_id': user_id,
                    'message_id': row.get('message_id', idx),
                }
                
                for combo in ['combo1', 'combo2', 'combo3', 'combo4']:
                    context = contexts[combo]
                    result[f'{combo}_bleu_unigram_precision'] = calculate_bleu_unigram_precision(context, memory)
                    result[f'{combo}_length'] = len(context.split()) if context else 0
                
                results.append(result)
                current_memories.append(memory)
            
            # Update cross-conversation memories
            cross_conversation_memories[user_id] += " " + " ".join(current_memories)
    
    return pd.DataFrame(results)


def print_summary(bleu_df):
    """Print summary of overlap statistics for all 4 combinations."""
    combinations = [
        ("COMBO 1: Current prompt only", "combo1", "current only"),
        ("COMBO 2: Conversation context (prior queries + current query)", "combo2", "conversation context"),
        ("COMBO 3: Conversation context with memories", "combo3", "conversation context + memories"),
        ("COMBO 4: Full cross-conversation context", "combo4", "full cross-conversation context")
    ]
    
    print("\n" + "="*80)
    print("MESSAGE-MEMORY OVERLAP SUMMARY - 4 COMBINATIONS")
    print("="*80)
    print(f"\nDataset: {len(bleu_df)} memory entries")
    
    # Context lengths
    print(f"\nAverage context lengths:")
    for _, prefix, desc in combinations:
        mean_len = bleu_df[f'{prefix}_length'].mean()
        print(f"- Combo {prefix[-1]} ({desc}): {mean_len:.1f} words")
    
    # BLEU scores
    bleu_means = {}
    for desc, prefix, _ in combinations:
        mean_bleu = bleu_df[f'{prefix}_bleu_unigram_precision'].mean()
        bleu_means[prefix] = mean_bleu
        print(f"\n{desc}:")
        print(f"- BLEU Unigram Precision: {mean_bleu:.3f}")
    
    # Improvement analysis
    print(f"\nIMPROVEMENT ANALYSIS (BLEU Unigram Precision):")
    base_bleu = bleu_means['combo1']
    for i in range(2, 5):
        improvement = bleu_means[f'combo{i}'] - base_bleu
        print(f"- Combo {i} vs Combo 1 improvement: {improvement:+.3f}")
    
    # Cross-comparison improvements
    print(f"\nCROSS-COMBINATION IMPROVEMENTS (BLEU Unigram Precision):")
    for i in range(2, 5):
        improvement = bleu_means[f'combo{i}'] - bleu_means[f'combo{i-1}']
        print(f"- Combo {i} vs Combo {i-1}: {improvement:+.3f}")
    
    # Best performing combination
    best_num = max(range(1, 5), key=lambda x: bleu_means[f'combo{x}'])
    best_prefix = f'combo{best_num}'
    best_bleu = bleu_means[best_prefix]
    print(f"\nBEST PERFORMING COMBINATION: Combo {best_num} (BLEU Unigram Precision: {best_bleu:.3f})")
    
    # Similarity distribution
    best_col = f'{best_prefix}_bleu_unigram_precision'
    scores = bleu_df[best_col]
    high_sim = (scores >= 0.8).sum()
    med_sim = ((scores >= 0.5) & (scores < 0.8)).sum()
    low_sim = (scores < 0.5).sum()
    
    print(f"\nCOMBO {best_num} BLEU UNIGRAM PRECISION DISTRIBUTION:")
    print(f"- High similarity (≥0.8): {high_sim} ({high_sim/len(bleu_df):.1%})")
    print(f"- Medium similarity (0.5-0.8): {med_sim} ({med_sim/len(bleu_df):.1%})")
    print(f"- Low similarity (<0.5): {low_sim} ({low_sim/len(bleu_df):.1%})")


def save_results(bleu_df, output_dir="../dummy_data/provenance_of_memories/"):
    """Save results to CSV."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "bleu_results.csv")
    bleu_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


def main():
    """Main execution function."""
    data_path = "../dummy_data/chatgpt_memories_msg_id.csv"
    
    try:
        df_memories = pd.read_csv(data_path)
        print("Dataset loaded successfully!")
        print(f"Total rows: {len(df_memories)}")
        print(f"Memory entries: {(df_memories['Updated Memory'] != 'No memory').sum()}")
        
        bleu_df = analyze_bleu(df_memories)
        print_summary(bleu_df)
        save_results(bleu_df)
        
        return bleu_df
    except FileNotFoundError:
        print(f"Dataset file not found: {data_path}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    main()