from sentence_transformers import SentenceTransformer, util
import torch
import os
from typing import List, Tuple

class ToolClassifier:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the ToolClassifier with a specified sentence transformer model.

        Parameters
        ----------
        model_name : str, optional
            The name of the sentence transformer model to use, by default 'all-MiniLM-L6-v2'
        """
        if os.getenv('FORCE_CPU', '0').lower() in ['1', 'true', 'yes']:
            self.device = 'cpu'
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"ToolClassifier: Using device: {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        print(f"ToolClassifier: Model {model_name} loaded successfully.")

    def recall_top_n(self, query: str, tool_names: List[str], top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Recalls the top N tool names most similar to the query.

        Parameters
        ----------
        query : str
            The input query string.
        tool_names : List[str]
            A list of tool names to compare against.
        top_n : int, optional
            The number of top tool names to return, by default 5.

        Returns
        -------
        List[Tuple[str, float]]
            A list of tuples, where each tuple contains a tool name and its similarity score,
            sorted by score in descending order.
        """
        if not tool_names:
            return []

        if top_n <= 0:
            raise ValueError("top_n must be a positive integer")
        
        if top_n > len(tool_names):
            top_n = len(tool_names)

        # Encode the query and tool names
        query_embedding = self.model.encode(query, convert_to_tensor=True, device=self.device)
        tool_embeddings = self.model.encode(tool_names, convert_to_tensor=True, device=self.device)

        # Compute cosine similarities
        # cosine_scores is a tensor of shape (1, len(tool_names))
        cosine_scores = util.cos_sim(query_embedding, tool_embeddings)[0]

        # Pair tool names with scores
        results = []
        for i, tool_name in enumerate(tool_names):
            results.append((tool_name, cosine_scores[i].item()))

        # Sort by score in descending order
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_n]

if __name__ == '__main__':
    # Example Usage
    print("Running ToolClassifier example...")
    classifier = ToolClassifier()

    sample_query = "I want to find information about the weather in London."
    sample_tool_names = [
        "get_current_weather",
        "search_web",
        "send_email",
        "calculate_distance",
        "find_restaurants",
        "translate_text",
        "get_stock_price"
    ]

    top_tools = classifier.recall_top_n(sample_query, sample_tool_names, top_n=3)

    print(f"\nQuery: {sample_query}")
    print(f"Top {len(top_tools)} recalled tools:")
    for tool, score in top_tools:
        print(f"  - {tool} (Score: {score:.4f})")

    sample_query_2 = "book a flight to New York"
    sample_tool_names_2 = [
        "find_flight_options",
        "reserve_hotel_room",
        "check_flight_status",
        "get_travel_insurance_quote"
    ]
    top_tools_2 = classifier.recall_top_n(sample_query_2, sample_tool_names_2, top_n=2)
    print(f"\nQuery: {sample_query_2}")
    print(f"Top {len(top_tools_2)} recalled tools:")
    for tool, score in top_tools_2:
        print(f"  - {tool} (Score: {score:.4f})")

    sample_query_3 = "what's the capital of France?"
    sample_tool_names_3 = [
        "get_capital_city",
        "general_knowledge_query",
        "lookup_country_info"
    ]
    top_tools_3 = classifier.recall_top_n(sample_query_3, sample_tool_names_3, top_n=3)
    print(f"\nQuery: {sample_query_3}")
    print(f"Top {len(top_tools_3)} recalled tools:")
    for tool, score in top_tools_3:
        print(f"  - {tool} (Score: {score:.4f})")
