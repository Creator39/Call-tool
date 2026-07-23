from pydantic import BaseModel

class FunctionCallTest(BaseModel):
    prompt : str

class FunctionCallTestList(BaseModel):
    test : list[FunctionCallTest]
