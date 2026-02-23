import runpod
from vllm import LLM, SamplingParams
import os

# Load model early
model_name = os.getenv("LLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0") 
llm = LLM(model=model_name)

def handler(event):
    """
    Expects input:
    {
        "input": {
            "prompt": "...",
            "max_tokens": 100
        }
    }
    """
    input_data = event["input"]
    prompt = input_data.get("prompt")
    max_tokens = input_data.get("max_tokens", 100)
    
    if not prompt:
        return {"error": "Missing prompt"}
    
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=max_tokens)
    
    outputs = llm.generate([prompt], sampling_params)
    
    # Print the outputs.
    generated_text = ""
    for output in outputs:
        generated_text += output.outputs[0].text
        
    return {
        "response": generated_text.strip()
    }

runpod.serverless.start({"handler": handler})
