import runpod
import subprocess
import base64
import os
import shutil

def handler(event):
    """
    Expects input:
    {
        "input": {
            "text": "..."
        }
    }
    """
    input_data = event["input"]
    text = input_data.get("text")
    
    if not text:
        return {"error": "Missing text"}
    
    output_path = "output.wav"
    
    # Securely call piper using subprocess
    try:
        # Check if piper is in path
        piper_path = shutil.which("piper")
        if not piper_path:
            return {"error": "Piper executable not found in PATH"}

        process = subprocess.Popen(
            [piper_path, "--model", "model.onnx", "--output_file", output_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(input=text)
        
        if process.returncode != 0:
            return {"error": f"Piper error: {stderr}"}
        
        if not os.path.exists(output_path):
            return {"error": "Output file not generated"}

        with open(output_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")
            
        if os.path.exists(output_path):
            os.remove(output_path)
            
        return {
            "audio_base64": audio_b64
        }
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
