import streamlit as st
import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd

RESULTS_DIR = Path("/path/to/npc-rl/results")  # Relative to the script location

def find_result_files() -> List[Tuple[str, str]]:
    """Find all merge_results_*.json files in the results directory.
    
    Returns:
        List of (display_name, file_path) tuples
    """
    result_files = []
    for file_path in glob.glob(str(RESULTS_DIR / "merge_results_*.json")):
        # Extract the model name from the filename
        # e.g. "merge_results_Qwen3-8B_with_rag.json" -> "Qwen3-8B with rag"
        display_name = Path(file_path).stem.replace("merge_results_", "").replace("_", " ")
        result_files.append((display_name, file_path))
    
    # Sort by display name
    return sorted(result_files, key=lambda x: x[0])

def load_conversations(file_path: str) -> List[Dict[str, Any]]:
    """Load conversations data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # If it's a single conversation, wrap it in a list
            if isinstance(data, dict):
                return [data]
            return data
    except Exception as e:
        st.error(f"Error loading file {file_path}: {str(e)}")
        return []

def get_turns_from_conversation(conversation: Dict[str, Any]) -> Dict[str, Any]:
    """Extract turns from conversation data where turns are stored as turn_1, turn_2, etc."""
    turns = {}
    for key, value in conversation.get("turns", {}).items():
        if key.startswith('turn_') and isinstance(value, dict):
            turns[key] = value
    return turns

def display_conversation_info(conversation: Dict[str, Any]):
    """Display common conversation information."""
    st.sidebar.header("Conversation Info")
    
    # Display basic conversation info
    st.sidebar.json({
        "conversation_id": conversation.get("data_id", "N/A"),
        "worldview": conversation.get("worldview", "N/A"),
        "total_turns": len(get_turns_from_conversation(conversation))
    })
    
    # Display player and NPC profiles if available
    if "player" in conversation:
        with st.sidebar:
            st.subheader("Player Profile")
            st.json({
                "name": conversation["player"].get("name", "Unknown"),
                "level": conversation["player"].get("level", "N/A"),
                "inventory": conversation["player"].get("inventory", [])
            })
    
    if "npc" in conversation:
        with st.sidebar:
            st.subheader("NPC Profile")
            st.json({
                "name": conversation["npc"].get("name", "Unknown"),
                "role": conversation["npc"].get("role", "N/A"),
                "personality": conversation["npc"].get("personality", "N/A")
            })

def format_tool_calls(tool_calls: List[Dict]) -> str:
    """Format tool calls for display."""
    if not tool_calls:
        return "No tool calls"
    return "\n".join([
        f"- {call.get('name', 'unnamed')}: {call.get('arguments', {})}"
        for call in tool_calls
    ])

def display_dialogue(dialogue: List[Dict[str, Any]]) -> None:
    """Display the conversation dialogue."""
    st.markdown("**Dialogue**")
    for msg in dialogue:
        speaker = msg.get('speaker', 'unknown').title()
        text = msg.get('text', '')
        with st.chat_message(speaker.lower()):
            st.write(text)

def display_functions(functions: List[Dict[str, Any]], title: str) -> None:
    """Display function calls with their parameters and returns."""
    if not functions:
        return
        
    st.markdown(f"**{title}**")
    for i, func in enumerate(functions, 1):
        with st.expander(f"{i}. {func.get('name', 'unnamed')}"):
            st.json(func)

def display_turn(turn_data: Dict[str, Any], turn_key: str):
    """Display a single turn's information."""
    st.markdown(f"### {turn_key.replace('_', ' ').title()}")
    
    # Display dialogue if present
    dialogue = turn_data.get("dialogue", [])
    if dialogue:
        display_dialogue(dialogue)
    
    # Display gold response and functions
    if "gold" in turn_data:
        gold = turn_data["gold"]
        if "response" in gold:
            with st.chat_message("assistant"):
                st.write("**Gold Response:**")
                st.write(gold["response"])
        
        if "functions" in gold and gold["functions"]:
            display_functions(gold["functions"], "Gold Function Calls")
    
    # Display model response and functions
    if "model" in turn_data:
        model = turn_data["model"]
        if "response" in model and model["response"] is not None:
            with st.chat_message("assistant"):
                st.write("**Model Response:**")
                st.write(model["response"])
        
        if "functions" in model and model["functions"]:
            display_functions(model["functions"], "Model Function Calls")

def display_file_selector() -> Optional[str]:
    """Display file selector for merge_results_*.json files."""
    st.sidebar.header("Select Result File")
    
    # Find all result files
    result_files = find_result_files()
    
    if not result_files:
        st.sidebar.warning(f"No merge_results_*.json files found in {RESULTS_DIR}")
        return None
    
    # Create a dropdown to select a file
    display_names = [name for name, _ in result_files]
    selected_display = st.sidebar.selectbox(
        "Select a result file:", 
        display_names
    )
    
    # Find the corresponding file path
    selected_file = next(path for name, path in result_files if name == selected_display)
    
    # Show file path
    st.sidebar.caption(f"Selected: {selected_file}")
    
    return selected_file

def main():
    st.set_page_config(
        page_title="Conversation Visualizer",
        page_icon="💬",
        layout="wide"
    )
    
    st.title("💬 Conversation Visualizer")
    
    # Display file selector
    selected_file = display_file_selector()
    
    if selected_file is None:
        st.warning(f"No merge_results_*.json files found in {RESULTS_DIR}. Please check the directory.")
        return
    
    # Load the conversations from the selected file
    conversations = load_conversations(selected_file)
    
    if not conversations:
        st.error("No conversations found in the selected file.")
        return
    
    # Add conversation selector if there are multiple conversations
    if len(conversations) > 1:
        conversation_index = st.sidebar.slider(
            "Select Conversation", 
            0, 
            len(conversations) - 1, 
            0,
            help="Select which conversation to view"
        )
    else:
        conversation_index = 0
    
    # Get the selected conversation
    conversation = conversations[conversation_index]
    
    # Display conversation info
    display_conversation_info(conversation)
    
    # Display turns
    st.header("Conversation Turns")
    
    # Extract turns from the conversation
    turns = get_turns_from_conversation(conversation)
    
    if not turns:
        st.warning("No turns found in the conversation.")
        return
    
    # Sort turns by their numeric suffix (turn_1, turn_2, etc.)
    turn_keys = sorted(turns.keys(), key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0)
    
    # Create a selectbox to navigate between turns
    selected_turn = st.selectbox("Select Turn", turn_keys)
    
    if selected_turn and selected_turn in turns:
        display_turn(turns[selected_turn], selected_turn)
    
    # Add an expander with raw JSON data
    with st.expander("View Raw JSON"):
        st.json(conversation)
        
    # Show conversation navigation info if there are multiple conversations
    if len(conversations) > 1:
        st.sidebar.info(f"Conversation {conversation_index + 1} of {len(conversations)}")

if __name__ == "__main__":
    main()
