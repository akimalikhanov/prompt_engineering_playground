#!/usr/bin/env python3
"""
Python example script to test the Prompts API endpoints
"""

import json

import requests

BASE_URL = "http://localhost:8001"


class PromptsAPIClient:
    """Simple client for the Prompts API"""

    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url.rstrip("/")

    def list_prompts(
        self, technique_key: str | None = None, q: str | None = None, enabled: bool | None = None
    ):
        """List prompts with optional filters"""
        params = {}
        if technique_key:
            params["technique_key"] = technique_key
        if q:
            params["q"] = q
        if enabled is not None:
            params["enabled"] = enabled

        response = requests.get(f"{self.base_url}/prompts", params=params)
        response.raise_for_status()
        return response.json()

    def create_prompt(
        self,
        technique_key: str,
        title: str,
        messages: list,
        variables: list | None = None,
        language: str = "en",
        model_hint: str | None = None,
    ):
        """Create a new prompt (v1)"""
        data = {
            "technique_key": technique_key,
            "title": title,
            "language": language,
            "messages": messages,
            "variables": variables or [],
            "model_hint": model_hint,
        }

        response = requests.post(f"{self.base_url}/prompts", json=data)
        response.raise_for_status()
        return response.json()

    def get_prompt(self, example_id: str):
        """Get a specific prompt by ID"""
        response = requests.get(f"{self.base_url}/prompts/{example_id}")
        response.raise_for_status()
        return response.json()

    def create_version(
        self,
        example_id: str,
        messages: list | None = None,
        variables: list | None = None,
        model_hint: str | None = None,
        language: str | None = None,
    ):
        """Create a new version of a prompt"""
        data = {}
        if messages:
            data["messages"] = messages
        if variables:
            data["variables"] = variables
        if model_hint:
            data["model_hint"] = model_hint
        if language:
            data["language"] = language

        response = requests.post(f"{self.base_url}/prompts/{example_id}/versions", json=data)
        response.raise_for_status()
        return response.json()

    def get_latest(self, technique_key: str, title: str):
        """Get latest version by technique and title"""
        response = requests.get(f"{self.base_url}/prompts/{technique_key}/{title}/latest")
        response.raise_for_status()
        return response.json()

    def render_prompt(self, example_id: str, variables: dict, version: int | None = None):
        """Render a prompt with Jinja2"""
        params = {}
        if version:
            params["version"] = version

        data = {"variables": variables}

        response = requests.post(
            f"{self.base_url}/prompts/{example_id}/render", json=data, params=params
        )
        response.raise_for_status()
        return response.json()

    def update_prompt(
        self, example_id: str, is_enabled: bool | None = None, status: str | None = None
    ):
        """Update prompt status or enabled flag"""
        data = {}
        if is_enabled is not None:
            data["is_enabled"] = is_enabled
        if status:
            data["status"] = status

        response = requests.patch(f"{self.base_url}/prompts/{example_id}", json=data)
        response.raise_for_status()
        return response.json()


def print_json(data):
    """Pretty print JSON data"""
    print(json.dumps(data, indent=2))


def main():
    """Run example tests"""
    client = PromptsAPIClient()

    print("=" * 60)
    print("Testing Prompts API with Python")
    print("=" * 60)
    print()

    # Test 1: List all prompts
    print("1. Listing all latest prompts...")
    prompts = client.list_prompts()
    print(f"   Found {prompts['count']} prompts")
    print()

    # Test 2: List CoT prompts
    print("2. Listing Chain-of-Thought prompts...")
    cot_prompts = client.list_prompts(technique_key="cot")
    print(f"   Found {cot_prompts['count']} CoT prompts")
    print()

    # Test 3: Create a new prompt
    print("3. Creating a new prompt...")
    new_prompt = client.create_prompt(
        technique_key="zero-shot",
        title="Python API Test Prompt",
        messages=[
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "Explain {{concept}} in {{language}}"},
        ],
        variables=[
            {
                "name": "concept",
                "type": "string",
                "required": True,
                "desc": "Programming concept to explain",
            },
            {
                "name": "language",
                "type": "string",
                "required": True,
                "desc": "Programming language context",
            },
        ],
        model_hint="GPT-4 or Claude recommended",
    )
    prompt_id = new_prompt["example_id"]
    print(f"   Created prompt with ID: {prompt_id}")
    print()

    # Test 4: Render the prompt
    print("4. Rendering the prompt with variables...")
    rendered = client.render_prompt(
        prompt_id, variables={"concept": "list comprehension", "language": "Python"}
    )
    print(f"   Used version: {rendered['used_version']}")
    print(f"   Missing vars: {rendered['missing_vars']}")
    print("   Rendered messages:")
    for msg in rendered["rendered_messages"]:
        print(f"     [{msg['role']}] {msg['content'][:60]}...")
    print()

    # Test 5: Test with missing variables
    print("5. Testing render with missing variables...")
    try:
        rendered = client.render_prompt(prompt_id, variables={"concept": "list comprehension"})
        print(f"   Missing vars: {rendered['missing_vars']}")
        print(f"   Warnings: {rendered['warnings']}")
    except Exception as e:
        print(f"   Error: {e}")
    print()

    # Test 6: Create a new version
    print("6. Creating a new version...")
    new_version = client.create_version(
        prompt_id,
        messages=[
            {"role": "system", "content": "You are an expert programming tutor."},
            {"role": "user", "content": "Explain {{concept}} in {{language}} with examples"},
        ],
        model_hint="GPT-4o recommended",
    )
    new_version_id = new_version["example_id"]
    print(f"   Created version {new_version['version']} with ID: {new_version_id}")
    print()

    # Test 7: Get latest by technique + title
    print("7. Getting latest version by technique + title...")
    latest = client.get_latest("zero-shot", "Python API Test Prompt")
    print(f"   Latest version: {latest['version']}")
    print(f"   Version ID: {latest['example_id']}")
    print()

    # Test 8: Disable the prompt
    print("8. Disabling the prompt...")
    updated = client.update_prompt(new_version_id, is_enabled=False)
    print(f"   Enabled status: {updated['is_enabled']}")
    print()

    # Test 9: Archive the prompt
    print("9. Archiving the prompt...")
    updated = client.update_prompt(new_version_id, status="archived")
    print(f"   Status: {updated['status']}")
    print()

    # Test 10: Render an existing CoT prompt
    print("10. Testing with an existing CoT prompt...")
    if cot_prompts["count"] > 0:
        cot_prompt = cot_prompts["prompts"][0]
        print(f"    Using prompt: {cot_prompt['title']}")

        # Check what variables it needs
        variables_needed = cot_prompt["variables"]
        print(f"    Variables needed: {[v['name'] for v in variables_needed]}")

        # Render with appropriate variables
        if variables_needed:
            var_name = variables_needed[0]["name"]
            rendered = client.render_prompt(
                cot_prompt["example_id"], variables={var_name: "42 * 13 + 7"}
            )
            print("    Rendered successfully!")
            print(f"    Missing vars: {rendered['missing_vars']}")
    else:
        print("    No CoT prompts found")
    print()

    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to API")
        print(f"Make sure the API is running at {BASE_URL}")
        print("Run: uvicorn api.prompts:app --reload --port 8001")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
