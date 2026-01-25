"""
Semantic Similarity Analysis for Context-Memory Agreement

Compute cosine similarity between context embeddings and memory embeddings
for different input combinations to evaluate semantic similarity.
"""

import pandas as pd
import numpy as np
import os
from typing import List, Optional
from tqdm import tqdm
import time
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken
from utils import get_combination_1_context, get_combination_2_context, get_combination_3_context, get_combination_4_context


# OpenAI embedding model
EMBEDDING_MODEL = "text-embedding-3-large"

# Initialize tiktoken encoder for OpenAI models (cl100k_base encoding)
_encoding = tiktoken.get_encoding("cl100k_base")



def truncate_context(context: str, max_tokens: int = 8000) -> str:
    """
    Truncate a context string to keep only the most recent part within the
    token budget using tiktoken.
    
    Args:
        context: The context string to truncate
        max_tokens: Maximum number of tokens (default 8000, slightly under 8192 limit)
    
    Returns:
        Truncated context string that fits within max_tokens
    """
    if not context:
        return context
    
    
    # Count tokens in the context
    tokens = _encoding.encode(context)
    
    # If within limit, return as-is
    if len(tokens) <= max_tokens:
        return context
    
    # Truncate to max_tokens, keeping the most recent tokens
    truncated_tokens = tokens[-max_tokens:]
    
    # Decode back to string
    truncated = _encoding.decode(truncated_tokens)
    
    return truncated


def get_embedding(text: str, client: OpenAI, model: str = EMBEDDING_MODEL) -> List[float]:
    """
    Get embedding for a text using OpenAI API.
    
    Args:
        text: Text to embed
        client: OpenAI client
        model: Embedding model name
    
    Returns:
        List of embedding values
    """
    if not text or not text.strip():
        # Return zero vector for empty text (standard dim.)
        return [0.0] * 1536
    
    try:
        response = client.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise


def compute_cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
    
    Returns:
        Cosine similarity score
    """
    if not embedding1 or not embedding2:
        return 0.0
    
    # Convert to numpy arrays
    emb1 = np.array(embedding1)
    emb2 = np.array(embedding2)
    
    # Compute norms
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    
    # Handle zero vectors
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # Compute cosine similarity: dot product / (norm1 * norm2)
    cosine_sim = np.dot(emb1, emb2) / (norm1 * norm2)
    
    # Clamp to [-1, 1] range (safety)
    return float(np.clip(cosine_sim, -1.0, 1.0))


def analyze_semantic_similarity(df: pd.DataFrame, client: OpenAI, 
                                delay_between_requests: float = 0.1,
                                max_samples: Optional[int] = None) -> pd.DataFrame:
    """
    Analyze semantic similarity between contexts and memories for the dataset using 4 combinations.
    
    Args:
        df: DataFrame with conversation data
        client: OpenAI client for generating embeddings
        delay_between_requests: Delay between API requests in seconds
        max_samples: Maximum number of samples to process (for testing)
    
    Returns:
        DataFrame with semantic similarity results
    """
    # Filter out rows with "No memory"
    df_filtered = df[df['Updated Memory'] != 'No memory'].copy()
    
    if max_samples:
        df_filtered = df_filtered.head(max_samples)
    
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
            current_memories = ""
            
            for idx, row in df_conv.iterrows():
                memory = row['Updated Memory']
                
                if memory != "No memory":
                    # Update current_memories
                    current_memories = current_memories + " " + memory
                    
                    # Get contexts for all 4 combinations
                    combo1_context_raw = get_combination_1_context(df, idx)
                    combo2_context_raw = get_combination_2_context(df, idx, conversation_id)
                    combo3_context_raw = get_combination_3_context(df, idx, conversation_id)
                    combo4_context_raw = get_combination_4_context(df, idx, conversation_id, user_id, cross_conversation_memories)

                    # Truncate contexts to stay within approximate model context window,
                    # keeping the most recent portion of each.
                    combo1_context = truncate_context(combo1_context_raw)
                    combo2_context = truncate_context(combo2_context_raw)
                    combo3_context = truncate_context(combo3_context_raw)
                    combo4_context = truncate_context(combo4_context_raw)
                    
                    # Generate embeddings for contexts and memory
                    try:
                        combo1_emb = get_embedding(combo1_context, client)
                        time.sleep(delay_between_requests)
                        
                        # Check if combo2 is same as combo1 to avoid redundant API call
                        if combo2_context == combo1_context:
                            combo2_emb = combo1_emb
                        else:
                            combo2_emb = get_embedding(combo2_context, client)
                            time.sleep(delay_between_requests)
                        
                        # Check if combo3 is same as combo1 or combo2
                        if combo3_context == combo1_context:
                            combo3_emb = combo1_emb
                        elif combo3_context == combo2_context:
                            combo3_emb = combo2_emb
                        else:
                            combo3_emb = get_embedding(combo3_context, client)
                            time.sleep(delay_between_requests)
                        
                        # Check if combo4 is same as any previous
                        if combo4_context == combo1_context:
                            combo4_emb = combo1_emb
                        elif combo4_context == combo2_context:
                            combo4_emb = combo2_emb
                        elif combo4_context == combo3_context:
                            combo4_emb = combo3_emb
                        else:
                            combo4_emb = get_embedding(combo4_context, client)
                            time.sleep(delay_between_requests)
                        
                        # Generate embedding for memory (truncate if needed)
                        memory_truncated = truncate_context(memory, max_tokens=8000)
                        memory_emb = get_embedding(memory_truncated, client)
                        time.sleep(delay_between_requests)
                        
                        # Compute cosine similarities
                        combo1_similarity = compute_cosine_similarity(combo1_emb, memory_emb)
                        combo2_similarity = compute_cosine_similarity(combo2_emb, memory_emb)
                        combo3_similarity = compute_cosine_similarity(combo3_emb, memory_emb)
                        combo4_similarity = compute_cosine_similarity(combo4_emb, memory_emb)
                        
                        # Combine results
                        similarity_metrics = {
                            'row_index': idx,
                            'conversation_id': conversation_id,
                            'user_id': user_id,
                            'message_id': row.get('message_id', idx),  # Use message_id if available
                            'memory': memory,
                            # Combination 1: Current prompt only
                            'combo1_semantic_similarity': combo1_similarity,
                            # Combination 2: Conversation context
                            'combo2_semantic_similarity': combo2_similarity,
                            # Combination 3: Conversation context with memories
                            'combo3_semantic_similarity': combo3_similarity,
                            # Combination 4: Full cross-conversation context
                            'combo4_semantic_similarity': combo4_similarity,
                            # Context lengths for reference
                            'combo1_length': len(combo1_context.split()) if combo1_context else 0,
                            'combo2_length': len(combo2_context.split()) if combo2_context else 0,
                            'combo3_length': len(combo3_context.split()) if combo3_context else 0,
                            'combo4_length': len(combo4_context.split()) if combo4_context else 0
                        }
                        
                        results.append(similarity_metrics)
                        
                    except Exception as e:
                        print(f"Error processing row {idx}: {e}")
                        continue
            
            # Update cross-conversation memories
            cross_conversation_memories = cross_conversation_memories + " " + current_memories
    
    return pd.DataFrame(results)


def print_summary(similarity_df: pd.DataFrame):
    """Print simple summary of semantic similarity statistics for all 4 combinations."""
    print("\n" + "="*80)
    print("SEMANTIC SIMILARITY SUMMARY - 4 COMBINATIONS")
    print("="*80)
    
    print(f"\nDataset: {len(similarity_df)} memory entries")
    print(f"Average context lengths:")
    print(f"- Combo 1 (current only): {similarity_df['combo1_length'].mean():.1f} words")
    print(f"- Combo 2 (conversation context): {similarity_df['combo2_length'].mean():.1f} words")
    print(f"- Combo 3 (conversation context + memories): {similarity_df['combo3_length'].mean():.1f} words")
    print(f"- Combo 4 (full cross-conversation context): {similarity_df['combo4_length'].mean():.1f} words")
    
    # Define combination descriptions
    combinations = [
        ("COMBO 1: Current prompt only", "combo1"),
        ("COMBO 2: Conversation context (prior queries + current query)", "combo2"),
        ("COMBO 3: Conversation context with memories (prior queries + generated memories)", "combo3"),
        ("COMBO 4: Full cross-conversation context (prior queries + generated memories across conversations)", "combo4")
    ]
    
    # Print results for each combination
    for desc, prefix in combinations:
        similarity_mean = similarity_df[f'{prefix}_semantic_similarity'].mean()
        similarity_std = similarity_df[f'{prefix}_semantic_similarity'].std()
        
        print(f"\n{desc}:")
        print(f"- Semantic Similarity: {similarity_mean:.4f} ± {similarity_std:.4f} (cosine similarity)")
    
    # Improvement analysis
    print(f"\nIMPROVEMENT ANALYSIS:")
    base_similarity = similarity_df['combo1_semantic_similarity'].mean()
    
    for i, (desc, prefix) in enumerate(combinations[1:], 2):
        improvement = similarity_df[f'{prefix}_semantic_similarity'].mean() - base_similarity
        print(f"- Combo {i} vs Combo 1: {improvement:+.4f}")
    
    # Best performing combination
    best_combo = None
    best_score = -1
    for prefix in ['combo1', 'combo2', 'combo3', 'combo4']:
        score = similarity_df[f'{prefix}_semantic_similarity'].mean()
        if score > best_score:
            best_score = score
            best_combo = prefix
    
    print(f"\nBEST PERFORMING COMBINATION: {best_combo.upper()} (Similarity: {best_score:.4f})")


def save_results(similarity_df: pd.DataFrame, output_dir: str = "../dummy_data/provenance_of_memories/"):
    """Save results to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "semantic_similarity_results.csv")
    similarity_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


def main():
    """Main function to run the semantic similarity analysis."""
    try:
        # Initialize OpenAI client
        load_dotenv("../.env")
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables.")
        
        client = OpenAI(api_key=api_key)
        
        # Load data
        df_memories = pd.read_csv("../dummy_data/chatgpt_memories_msg_id.csv")
        # sampled_users = ["user_1"] # FOR TESTING, we can use a smaller sample
        # df_memories = df_memories[df_memories['user_id'].isin(sampled_users)]
        
        print("Dataset loaded successfully!")
        print(f"Total rows: {len(df_memories)}")
        print(f"Memory entries: {(df_memories['Updated Memory'] != 'No memory').sum()}")
        
        # Analyze semantic similarity
        print("\nStarting semantic similarity analysis...")
        
        similarity_df = analyze_semantic_similarity(
            df_memories,
            client,
            delay_between_requests=0.1,
            # max_samples=100,  # FOR TESTING, we can use a smaller sample
        )
        
        # Print summary
        print_summary(similarity_df)
        
        # Save results
        save_results(similarity_df)
        
        return similarity_df
        
    except FileNotFoundError:
        print("Dataset file not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
