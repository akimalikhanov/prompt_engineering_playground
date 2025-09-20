import os
import yaml
from dotenv import load_dotenv
from types import GeneratorType
from adapters.openai_conn import openai_call
from adapters.gemini_conn import gemini_call
from adapters.vllm_conn import vllm_call
import argparse

load_dotenv()
openai_key = os.getenv('OPENAI_API_KEY')
google_key = os.getenv('GOOGLE_API_KEY')

# Load model configurations from YAML file
try:
    with open('config/models.yaml', 'r') as f:
        models_config = yaml.safe_load(f)
except FileNotFoundError:
    print("Error: models.yaml not found in config/ directory")
    exit(1)
except yaml.YAMLError:
    print("Error: Invalid YAML format in models.yaml")
    exit(1)

# Create a lookup dictionary for easy access
models_lookup = {model['id']: model for model in models_config['models']}

# Get available model IDs for argument parser
available_models = list(models_lookup.keys())

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected (True/False)")

parser = argparse.ArgumentParser(description="Run adapter test")
parser.add_argument("--model_id", choices=available_models, default="gpt", 
                   help=f"Select from models: {available_models}")
parser.add_argument(
    "--stream",
    type=str2bool,
    nargs="?",
    const=True,
    default=models_config['defaults']['stream'],
    help=f"Enable streaming: True or False (default={models_config['defaults']['stream']})"
)
parser.add_argument("--message", type=str, required=True, help="Prompt message")
parser.add_argument("--temperature", type=float, default=models_config['defaults']['params']['temperature'])
parser.add_argument("--top_p", type=float, default=models_config['defaults']['params']['top_p'])
parser.add_argument("--max_tokens", type=int, default=models_config['defaults']['params']['max_tokens'])
parser.add_argument("--seed", type=int, default=models_config['defaults']['params']['seed'])
args = parser.parse_args()

params = {
    "temperature": args.temperature,
    "top_p": args.top_p,
    "max_tokens": args.max_tokens,
    "seed": args.seed,
}
message = args.message


def streamer(gen):
    try:
        for chunk in gen:
            print(chunk, end="", flush=True)
        print()
    except KeyboardInterrupt:
        print("\n[stream interrupted]")

def main():
    if args.model_id=='gpt':
        ress = openai_call(
            model=models_lookup[args.model_id]['model_name'],
            messages=message,
            params=params,
            stream=args.stream,
            api_key=openai_key
        )

        if isinstance(ress, GeneratorType):
            streamer(ress)
        else:
            # non-streaming mode
            print(ress)

    elif args.model_id=='gemini':
        ress = gemini_call(
            model=models_lookup[args.model_id]['model_name'],
            messages=message,
            params=params,
            stream=args.stream,
            api_key=google_key
        )

        if isinstance(ress, GeneratorType):
            streamer(ress)
        else:
            # non-streaming mode
            print(ress)

    elif args.model_id in ('llama', 'qwen'):
        model_config = models_lookup[args.model_id]
        ress = vllm_call(
            model=model_config['model_path'],
            messages=message,
            params=params,
            stream=args.stream,
            base_url=f"http://{model_config['server']['host']}:{model_config['server']['port']}{model_config['server']['base_path']}"
        )

        if isinstance(ress, GeneratorType):
            streamer(ress)
        else:
            # non-streaming mode
            print(ress)

if __name__ == "__main__":
    main()