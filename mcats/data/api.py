from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

#placeholder for predict function
@app.get("/predict")
def predict():
    return {'wait': "wait_prediction"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
