import os
import subprocess
import sys

# Config - User Edit Required
DOCKER_USER = "amit14916" 

MODELS = {
    "whisper": "Dockerfile.whisper",
    "vllm": "Dockerfile.vllm",
    "tts": "Dockerfile.tts"
}

def run_cmd(cmd):
    print(f"Executing: {cmd}")
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
    process.wait()
    if process.returncode != 0:
        print(f"Error executing command: {cmd}")
        sys.exit(1)

def main():
    if DOCKER_USER == "your-docker-username":
        print("\n[!] ERROR: Please edit build_and_deploy.py and set DOCKER_USER to your Docker Hub username.")
        return

    print(f"🚀 Starting Build & Deploy Process for {DOCKER_USER}")
    
    # 1. Login Check (optional, but good to have)
    print("\nStep 1: Verifying Docker Login...")
    # run_cmd("docker login") # Uncomment if you haven't logged in

    # 2. Build and Push
    for name, dockerfile in MODELS.items():
        image_tag = f"{DOCKER_USER}/mcs-{name}:latest"
        
        print(f"\n--- Processing {name.upper()} ---")
        
        # Build
        print(f"Building image: {image_tag}...")
        run_cmd(f"docker build -t {image_tag} -f ai_engine/{dockerfile} ai_engine")
        
        # Push
        print(f"Pushing image: {image_tag} to Docker Hub...")
        run_cmd(f"docker push {image_tag}")
        
    print("\n✅ All images pushed successfully!")
    print("\nNext Steps:")
    print("1. Go to Runpod.io Console -> Serverless -> My Endpoints")
    print("2. Create New Endpoint for each image:")
    for name in MODELS.keys():
        print(f"   - {name}: {DOCKER_USER}/mcs-{name}:latest")
    print("\n3. Copy the Endpoint IDs and update your Railway/Local environment variables.")

if __name__ == "__main__":
    main()
