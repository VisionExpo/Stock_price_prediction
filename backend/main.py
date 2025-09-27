from fastapi import FastAPI

app = FastAPI(title="Stock Market AI API")

@app.get("/")
def read_root():
    return {"message":"Welcome to the Stock Marcket Scam"}