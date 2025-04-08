from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import json

app = FastAPI()

# Define request schema
class CodeConvertRequest(BaseModel):
    source_language: str
    target_language: str
    code: str

# AWS Bedrock client
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

@app.post("/convert")
def convert_code(request: CodeConvertRequest):
    # Prompt construction for Titan 
    prompt = f"Convert the following {request.source_language} code into {request.target_language}:\n\n```{request.source_language}\n{request.code}\n```"

    body = json.dumps({
        "inputText": prompt,
        "textGenerationConfig": {
            "temperature": 0.7,
            "topP": 1.0,
            "maxTokenCount": 500,
            "stopSequences": []
        }
    })

    try:
        response = bedrock.invoke_model(
            modelId="amazon.titan-text-lite-v1",
            body=body,
            contentType="application/json",
            accept="application/json"
        )
        response_body = json.loads(response["body"].read())
        converted_code = response_body["results"][0]["outputText"]
        return {"converted_code": converted_code.strip()}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
