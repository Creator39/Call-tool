from pydantic import BaseModel

class FunctionCallResult(BaseModel):
    prompt : str
    name : str
    parameters : dict[str, int | str | float | bool]

class FunctionCallResultList(BaseModel):
    result : list[FunctionCallResult]