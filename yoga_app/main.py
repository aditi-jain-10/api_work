from fastapi import FastAPI
from yoga_app.api.api import router
app = FastAPI()



@app.get("/")
async def root():
    return {"message": "Hello World"}


app.include_router(router, )

