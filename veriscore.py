import requests
import json
import os
from typing import Dict, Any

# Get URL from environment variable with fallback
DEFAULT_VERISCORE_URL = "http://10.137.193.1:36981/veriscore"
VERISCORE_URL = os.getenv("VERISCORE_URL", DEFAULT_VERISCORE_URL)

def verify_response(verification_result: str) -> bool:
    """
    Verify if the response is supported based on the verification result.
    
    Args:
        verification_result (str): The verification result from the API
        
    Returns:
        bool: True if supported, False otherwise
    """
    return verification_result.lower() == "supported"

def validate_input(question: str, response: str) -> None:
    """
    Validate input parameters to ensure they are valid strings.
    
    Args:
        question (str): The question to validate
        response (str): The response to validate
        
    Raises:
        ValueError: If any input is None or empty after stripping
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be None or empty")
    if not response or not response.strip():
        raise ValueError("Response cannot be None or empty")

def run_veriscore(question: str, response: str, last_sentence_only: bool = True) -> Dict[str, Any]:
    """
    Test the veriscore API endpoint.
    
    Args:
        question (str): The question being asked
        response (str): The response to be evaluated
        last_sentence_only (bool): Whether to only evaluate the last sentence. Defaults to True.
    
    Returns:
        Dict[str, Any]: The API response containing the veriscore evaluation
    """
    # Validate inputs
    validate_input(question, response)
    
    url = VERISCORE_URL
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    # Ensure all values are properly formatted
    payload = {
        "question": str(question).strip(),
        "response": str(response).strip(),
        "last_sentence_only": bool(last_sentence_only)
    }
    
    # Print request details for debugging
    print(f"\nMaking request to: {url}")
    print("Headers:", json.dumps(headers, indent=2))
    print("Payload:", json.dumps(payload, indent=2))
    
    # Make the request
    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=30
    )
    response.raise_for_status()
    return response.json()


def veriscore(question: str, response: str, last_sentence_only: bool = True) -> bool:
    """
    Verify if the response is supported based on the verification result.
    
    Args:
        question (str): The question being asked
        response (str): The response to be evaluated
        last_sentence_only (bool): Whether to only evaluate the last sentence. Defaults to True.
        
    Returns:
        bool: True if supported, False otherwise
    """
    verification_result = run_veriscore(question, response, last_sentence_only)["verification_results"]
    
    correctness = [int(result['verification_result'] == "supported") for result in verification_result]
    try:
        score = sum(correctness) / len(correctness)
    except ZeroDivisionError:
        print(f"Error: {verification_result}")
        score = 0.0
    return score



if __name__ == "__main__":
    print(f"Using Veriscore URL: {VERISCORE_URL}")
    
    # Test case 1: Simple supported case
    question1 = "What is the capital of France?"
    response1 = "The capital of France is Paris."
    
    print("\n=== Test Case 1 (Simple Supported) ===")
    result1 = run_veriscore(question1, response1)
    print("API Response:", json.dumps(result1, indent=2))
    if "verification_result" in result1:
        print("Is Supported:", verify_response(result1["verification_result"]))
    
    # Test case 2: Simple unsupported case
    question2 = "What is the population of Paris?"
    response2 = "The population of Paris is 15 million people."
    
    print("\n=== Test Case 2 (Simple Unsupported) ===")
    result2 = run_veriscore(question2, response2)
    print("API Response:", json.dumps(result2, indent=2))
    if "verification_result" in result2:
        print("Is Supported:", verify_response(result2["verification_result"]))
    
    # Test case 3: Multiple simple claims
    question3 = "What are some facts about Paris?"
    response3 = "Paris is the capital of France. The Eiffel Tower is located in Paris. The Louvre Museum is in Paris."
    
    print("\n=== Test Case 3 (Multiple Simple Claims - Last Sentence Only) ===")
    result3 = run_veriscore(question3, response3)
    print("API Response:", json.dumps(result3, indent=2))
    if "verification_result" in result3:
        print("Is Supported:", verify_response(result3["verification_result"]))
    
    print("\n=== Test Case 3 (Multiple Simple Claims - Full Response) ===")
    result3_full = run_veriscore(question3, response3, last_sentence_only=False)
    print("API Response:", json.dumps(result3_full, indent=2))
    if "verification_result" in result3_full:
        print("Is Supported:", verify_response(result3_full["verification_result"]))
