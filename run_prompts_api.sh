#!/bin/bash
# Script to run the Prompts API server

set -e

echo "Starting Prompts API server..."
echo "API will be available at: http://localhost:8001"
echo "Documentation: http://localhost:8001/docs"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "ERROR: .env file not found"
    echo "Please create a .env file with database configuration"
    exit 1
fi

# Load environment variables
source .env

# Run the API server (only watch specific directories)
uvicorn api.prompts:app --reload --port 8001 --host 0.0.0.0 \
  --reload-dir api \
  --reload-dir schemas \
  --reload-dir models \
  --reload-dir services \
  --reload-dir utils \
  --reload-dir config

