# fragment main.py dla modelu
import uvicorn
# ... reszta kodu FastAPI ...
if __name__ == "__main__":
    # Uruchom na 0.0.0.0, na porcie 8000
    uvicorn.run(app, host="0.0.0.0", port=7)