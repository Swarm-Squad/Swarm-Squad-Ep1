#!/usr/bin/env python3
"""
Test script to verify Ollama connectivity and response format.
Run this script to check if Ollama is working correctly.

Usage:
    python test_ollama.py            # Run all test_*.py files in tests folder
    python test_ollama.py -m MODEL   # Test specific Ollama model
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import requests

# Configuration
OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
DEFAULT_MODEL = "llama3.3:70b-instruct-q4_K_M"


def test_ollama_connection(model_name=DEFAULT_MODEL):
    """Test basic connection to Ollama with a specific model"""
    print(f"Testing connection to Ollama at {OLLAMA_URL} with model {model_name}...")

    try:
        # Create a very simple request
        request_data = {
            "model": model_name,
            "messages": [{"role": "user", "content": "Reply with 'Test successful'"}],
        }

        # Make the request
        print("Sending request to Ollama...")
        print(f"Request data: {json.dumps(request_data, indent=2)}")

        response = requests.post(
            OLLAMA_URL,
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=300,
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


def run_all_tests():
    """Run all test_*.py files in the tests directory"""
    print("Running all test files in tests directory...")

    # Get the directory containing this script
    tests_dir = os.path.dirname(os.path.abspath(__file__))

    # Find all test_*.py files (excluding this file)
    this_file = os.path.basename(__file__)
    test_files = [
        f
        for f in glob.glob(os.path.join(tests_dir, "test_*.py"))
        if os.path.basename(f) != this_file
    ]

    if not test_files:
        print("Running Ollama connection test...")
        if test_ollama_connection():
            print("✅ Ollama connection test passed")
            return True
        else:
            print("❌ Ollama connection test failed")
            return False

    print(f"Found {len(test_files)} test files:")
    for test_file in test_files:
        print(f"  - {os.path.basename(test_file)}")

    failed_tests = []

    # Run each test file
    for test_file in test_files:
        print(f"\nRunning {os.path.basename(test_file)}...")
        result = subprocess.run([sys.executable, test_file], capture_output=True)

        if result.returncode == 0:
            print(f"✅ {os.path.basename(test_file)} passed")
        else:
            print(f"❌ {os.path.basename(test_file)} failed")
            print(f"Output: {result.stdout.decode()}")
            print(f"Error: {result.stderr.decode()}")
            failed_tests.append(os.path.basename(test_file))

    # Also run this file's Ollama test
    print("\nRunning Ollama connection test...")
    if test_ollama_connection():
        print("✅ Ollama connection test passed")
    else:
        print("❌ Ollama connection test failed")
        failed_tests.append("ollama_connection")

    # Report results
    if failed_tests:
        print(f"\n{len(failed_tests)} test(s) failed: {', '.join(failed_tests)}")
        return False
    else:
        print("\nAll tests passed!")
        return True


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Test Ollama connectivity and run test files"
    )

    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help=f"Test specific Ollama model (default: {DEFAULT_MODEL})",
    )

    return parser.parse_args()


def main():
    """Main entry point for the test script"""
    args = parse_arguments()

    if args.model:
        # Test specified Ollama model
        print(f"Testing specific Ollama model: {args.model}")
        success = test_ollama_connection(args.model)

        if success:
            print(
                "\nTest PASSED: Successfully connected to Ollama and received valid response"
            )
            sys.exit(0)
        else:
            print(
                "\nTest FAILED: Could not connect to Ollama or received invalid response"
            )
            print("Please check that Ollama is running and the model is available")
            sys.exit(1)
    else:
        # Run all tests
        success = run_all_tests()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
