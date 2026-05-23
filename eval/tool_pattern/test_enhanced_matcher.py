import sys
import os

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tool_pattern.tool_pattern_matcher_new import ToolPatternMatcher

def run_tests():
    # Initialize the matcher
    matcher = ToolPatternMatcher()
    
    # Test cases
    test_cases = [
        # Price queries
        ("How much does a sword cost?", "check_price for 'sword'"),
        ("What's the price of a health potion?", "check_price for 'health potion'"),
        ("Price of a bow", "check_price for 'bow'"),
        
        # Attack queries
        ("What's the attack power of a sword?", "check_attack for 'sword'"),
        ("How strong is the dragon slayer?", "check_attack for 'dragon slayer'"),
        ("Attack value of a bow", "check_attack for 'bow'"),
        
        # Type queries
        ("What type of item is a sword?", "check_type for 'sword'"),
        ("Is a health potion consumable?", "check_type for 'health potion'"),
        ("Category of a bow", "check_type for 'bow'"),
        
        # Description queries
        ("Tell me about the sword", "check_description for 'sword'"),
        ("Describe the health potion", "check_description for 'health potion'"),
        ("What can you tell me about the bow?", "check_description for 'bow'"),
        
        # Quest queries
        ("Tell me about the main quest", "check_quest_info for 'main quest'"),
        ("What's the dragon slayer quest about?", "check_quest_info for 'dragon slayer quest'"),
        ("Description of the lost treasure quest", "check_quest_description for 'lost treasure quest'"),
        
        # Equip intents
        ("Equip the sword", "equip for 'sword'"),
        ("I want to use the bow", "equip for 'bow'"),
        ("Wield the dragon slayer", "equip for 'dragon slayer'"),
        
        # Confirmations
        ("Yes", "confirm"),
        ("That's right", "confirm"),
        ("Confirm", "confirm"),
    ]
    
    # Run the tests
    passed = 0
    for query, expected in test_cases:
        try:
            # Get the function call
            func_call = matcher.match_function(query)
            
            # Format the result for display
            if not func_call:
                result = "No function matched"
            else:
                func_name = func_call['name']
                params = func_call.get('parameters', {})
                
                if func_name == 'search_item':
                    desc = params.get('item_description', '')
                    result = f"search_item for '{desc}'"
                elif func_name == 'search_quest':
                    desc = params.get('quest_description', '')
                    result = f"search_quest for '{desc}'"
                else:
                    # Get the first parameter value
                    param_value = next(iter(params.values())) if params else ''
                    result = f"{func_name} for '{param_value}'"
            
            # Check if the result matches the expected output
            if expected in result:
                status = "PASSED"
                passed += 1
            else:
                status = f"FAILED (expected: {expected})"
            
            print(f"Test: '{query}'")
            print(f"  Result: {result}")
            print(f"  Status: {status}")
            print()
            
        except Exception as e:
            print(f"Error processing query: '{query}'")
            print(f"  Error: {str(e)}")
            print()
    
    # Print summary
    total = len(test_cases)
    print(f"\nTest Summary: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

if __name__ == "__main__":
    run_tests()
