# api/main.py
from fastapi import FastAPI

app = FastAPI(title="Prompt Engineering Playground")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Hello from Stage 1 ✅"}

# Optional: separate hello
@app.get("/hello")
def hello():
    return {"message": "Hello"}
