import base64
import json
import os
import sys
from pathlib import Path
import requests

def main():
    print("=== ASR Container Diagnostic Tool ===")
    
    # 1. Determine the team track from environment or search common paths
    track = os.getenv("TEAM_TRACK", "")
    data_dir = None
    
    possible_tracks = [track, "novice", "advanced"]
    for t in possible_tracks:
        if not t:
            continue
        p = Path(f"/home/jupyter/{t}/asr")
        if p.exists() and (p / "asr.jsonl").exists():
            data_dir = p
            print(f"Found ASR data directory: {data_dir}")
            break
            
    if data_dir is None:
        # Check standard folders in home
        for p in Path("/home/jupyter").glob("**/asr/asr.jsonl"):
            data_dir = p.parent
            print(f"Found ASR data directory by search: {data_dir}")
            break
            
    if data_dir is None:
        print("ERROR: Could not find the ASR data directory with 'asr.jsonl'.")
        print("Please set the TEAM_TRACK environment variable, e.g., export TEAM_TRACK=novice")
        sys.exit(1)
        
    # 2. Read the first few instances
    jsonl_path = data_dir / "asr.jsonl"
    print(f"Reading instances from: {jsonl_path}")
    instances = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                instances.append(json.loads(line.strip()))
                if len(instances) >= 3:
                    break
                    
    print(f"Loaded {len(instances)} instances for testing.")
    
    # 3. Prepare payload
    payload_instances = []
    for instance in instances:
        audio_path = data_dir / instance["audio"]
        if not audio_path.exists():
            print(f"ERROR: Audio file not found at {audio_path}")
            continue
            
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
            
        payload_instances.append({
            "key": instance["key"],
            "b64": base64.b64encode(audio_bytes).decode("ascii")
        })
        
    if not payload_instances:
        print("ERROR: No valid audio payloads prepared.")
        sys.exit(1)
        
    # 4. Query the local server
    url = "http://localhost:5001/asr"
    print(f"Querying local ASR server at {url}...")
    
    try:
        response = requests.post(
            url,
            json={"instances": payload_instances},
            timeout=180
        )
        print(f"Response Status Code: {response.status_code}")
        
        if response.status_code == 200:
            res_data = response.json()
            predictions = res_data.get("predictions", [])
            print("\n=== Test Results ===")
            for i, pred in enumerate(predictions):
                gt = instances[i]
                print(f"\nInstance #{i+1}:")
                print(f"  Audio File:  {gt['audio']}")
                print(f"  Language:    {gt['language']}")
                print(f"  Ground Truth: {repr(gt['transcript'])}")
                print(f"  Prediction:   {repr(pred)}")
        else:
            print("ERROR: Server returned a non-200 status code.")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("\nERROR: Could not connect to the local server at http://localhost:5001.")
        print("Please make sure you have started your ASR container locally.")
        print("You can start it manually or run 'til test asr' to let the CLI deploy it.")
    except Exception as e:
        print(f"\nERROR: Unexpected exception occurred: {e}")

if __name__ == "__main__":
    main()
