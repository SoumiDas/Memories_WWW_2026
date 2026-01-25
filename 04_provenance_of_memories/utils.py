import pandas as pd
import re


def normalize_string(string):
    """Normalize string for comparison."""
    if pd.isna(string) or string == "No memory":
        return ""
    return re.sub(r'\s+', ' ', str(string).lower().strip())


def tokenize_string(string):
    """Tokenize string into words."""
    if not string:
        return []
    return re.findall(r'\b\w+\b', string.lower())



def get_combination_1_context(df, current_idx):
    """Combination 1: Current prompt only."""
    current_row = df.loc[current_idx]
    return current_row['User Message']


def get_combination_2_context(df, current_idx, conversation_id):
    """Combination 2: Conversation context (prior conversation queries + current query)."""
    # Get all messages in the same conversation up to and including current message
    conversation_messages = df[
        (df['conversation_id'] == conversation_id) & 
        (df.index <= current_idx)
    ].sort_index()
    
    if len(conversation_messages) == 0:
        return ""
    
    # Combine all messages in the conversation context
    context_messages = conversation_messages['User Message'].tolist()
    return " ".join(context_messages)


def get_combination_3_context(df, current_idx, conversation_id):
    """Combination 3: Conversation context with memories (prior conversation queries + generated memories)."""
    # Get all messages in the same conversation up to and including current message
    conversation_messages = df[
        (df['conversation_id'] == conversation_id) & 
        (df.index <= current_idx)
    ].sort_index()
    
    if len(conversation_messages) == 0:
        return ""
    
    # Get all messages in conversation
    context_messages = conversation_messages['User Message'].tolist()
    
    # Get all previous memories in the same conversation (excluding current)
    previous_memories = df[
        (df['conversation_id'] == conversation_id) & 
        (df.index < current_idx) &
        (df['Updated Memory'] != 'No memory')
    ]['Updated Memory'].tolist()
    
    # Combine messages and memories
    all_context = context_messages + previous_memories
    return " ".join(all_context)


def get_combination_4_context(df, current_idx, conversation_id, user_id, cross_conversation_memories=""):
    """Combination 4: Full cross-conversation context (prior conversation queries + generated memories across conversations)."""
    # Get combination 3 context first
    combo3_context = get_combination_3_context(df, current_idx, conversation_id)
    
    # If no cross-conversation memories, return combo3 context
    if not cross_conversation_memories:
        return combo3_context
    
    # Combine with cross-conversation memories
    full_context = combo3_context + " " + cross_conversation_memories
    return full_context