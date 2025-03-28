#!/usr/bin/env python3
"""
Test script to verify Ollama connectivity and response format.
Run this script to check if Ollama is working correctly.
"""

import json
import sys

import requests

# Configuration
OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
MODEL_NAME = "llama3.3:70b-instruct-q4_K_M"  # Use the same model as in your config


def test_ollama_connection():
    """Test basic connection to Ollama"""
    print(f"Testing connection to Ollama at {OLLAMA_URL}...")

    try:
        # Create a very simple request
        request_data = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "Reply with 'Test successful'"}],
        }

        # Make the request
        print("Sending request to Ollama...")
        print(f"Request data: {json.dumps(request_data, indent=2)}")

        response = requests.post(
            OLLAMA_URL,
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=15,
        )

        # Check status code
        print(f"Response status code: {response.status_code}")

        if response.status_code != 200:
            print(f"Error: Received non-200 status code: {response.status_code}")
            print(f"Response text: {response.text}")
            return False

        # Print raw response for debugging
        print(f"Raw response text (first 500 chars):\n{response.text[:500]}...")

        # Try to parse JSON response
        try:
            result = response.json()
            print(f"Response structure: {list(result.keys())}")

            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                print(f"SUCCESS! Extracted content: {content}")
                return True
            else:
                print("Error: No 'choices' field found in response.")
                print(f"Full response: {json.dumps(result, indent=2)}")
                return False
        except Exception as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Full response text: {response.text}")
            return False

    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        return False


if __name__ == "__main__":
    print("Ollama Connection Test")
    print("=====================")

    success = test_ollama_connection()

    if success:
        print(
            "\nTest PASSED: Successfully connected to Ollama and received valid response"
        )
        sys.exit(0)
    else:
        print("\nTest FAILED: Could not connect to Ollama or received invalid response")
        print("Please check that Ollama is running and the model is available")
        sys.exit(1)
