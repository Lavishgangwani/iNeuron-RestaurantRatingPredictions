#app.py

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


#create inputs using pydantic
class Item(BaseModel):
    name : str
    age : int


@app.get('/')
async def read_root():
    return {"Hello" : "World"}

@app.post('/items/')
async def create_item(item : Item):
    return item

