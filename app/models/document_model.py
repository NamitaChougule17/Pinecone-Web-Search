from pydantic import BaseModel

class Document(BaseModel):
    name: str
    content: str
