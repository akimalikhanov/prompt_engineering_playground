#!/usr/bin/env python3
"""
CLI script to test the universal router (via openai_adapter).

Examples:
  - OpenAI (string):
      python experiments/cli_router_test.py --provider openai --model gpt --message "Hello, world!"

  - Google Gemini (string):
      python experiments/cli_router_test.py --provider google --model gemini-flash --message "Ping"

  - vLLM local (string):
      python experiments/cli_router_test.py --provider vllm --model llama --message "Hi"

  - Chat history (list of dicts):
      python experiments/cli_router_test.py --provider openai --model gpt \
        --message "[{\"role\":\"system\",\"content\":\"You are helpful\"},{\"role\":\"user\",\"content\":\"Hi\"}]"
"""

import argparse
import os
from services.router import route_call, get_available_models, get_available_providers
import ast

def str2bool(v):
    """Convert string to boolean for argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected (True/False)")

def main():
    # Get available models and providers
    available_models = list(get_available_models().keys())
    available_providers = list(get_available_providers().keys())
    
    parser = argparse.ArgumentParser(description="Test router service with CLI")
    parser.add_argument("--provider", choices=available_providers, required=True,
                       help=f"Provider ID. Available: {available_providers}")
    parser.add_argument("--model", choices=available_models, required=True,
                       help=f"Model ID. Available: {available_models}")
    def parse_message(value: str):
        """Accept a plain string or a Python literal list/dict for chat history."""
        try:
            parsed = ast.literal_eval(value)
            return parsed
        except Exception:
            return value

    parser.add_argument("--message", type=parse_message, required=True,
                       help="Message to send. Can be a string or Python list of messages.")
    parser.add_argument("--stream", type=str2bool, nargs="?", const=True, default=True,
                       help="Enable streaming (default: True)")
    parser.add_argument("--temperature", type=float, default=0.9, help="Temperature (default: 0.9)")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (default: 1.0)")
    parser.add_argument("--max_tokens", type=int, default=100, help="Max tokens (default: 100)")
    parser.add_argument("--seed", type=int, default=None, help="Seed (default: None)")
    
    args = parser.parse_args()
    
    # Prepare parameters
    params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
    }
    
    print(f"Calling {args.provider}/{args.model} with message: {args.message}")
    print(f"Parameters: {params}")
    print(f"Streaming: {args.stream}")
    print("-" * 50)
    
    try:
        # Call the router
        response = route_call(
            provider_id=args.provider,
            model_id=args.model,
            messages=args.message,
            params=params,
            stream=args.stream
        )
        
        if args.stream:
            # Handle streaming response
            print("Streaming response:")
            for chunk in response:
                print(chunk, end="", flush=True)
            print()  # New line after streaming
        else:
            # Handle non-streaming response
            print("Response:")
            print(response)
            
    except Exception as e:
        # print(f"Error: {e}")
        print(e)
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
