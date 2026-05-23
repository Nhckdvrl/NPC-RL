import os
import subprocess
import time
import atexit
import signal
import json
from pathlib import Path
try:
    from agents.openai_agent import OpenAIAgent
except ImportError:
    try:
        from .openai_agent import OpenAIAgent
    except ImportError:
        from openai_agent import OpenAIAgent
def get_model_name():
    # First check environment variable
    model_from_env = os.getenv("OPENAI_MODEL")
    if model_from_env:
        return model_from_env
        
    # If not in env, try to read from aicrowd.json
    try:
        aicrowd_path = Path(__file__).parent.parent / 'aicrowd.json'
        if aicrowd_path.exists():
            with open(aicrowd_path, 'r') as f:
                config = json.load(f)
                hf_models = config.get('hf_models', [])
                if hf_models and isinstance(hf_models, list) and len(hf_models) > 0:
                    return hf_models[0].get('repo_id', 'yinita/cpdc-qwen14-base-task1-v1-full-norag-3epoch')
    except Exception as e:
        print(f"Warning: Failed to read model from aicrowd.json: {e}")
    
    # Fallback to default if all else fails
    return "yinita/cpdc-qwen14-base-task1-v1-full-norag-3epoch"
def should_start_vllm():
    """Check if vLLM server should be started based on aicrowd.json"""
    try:
        # Get the parent directory of this file's directory
        parent_dir = Path(__file__).parent.parent
        aicrowd_path = parent_dir / 'aicrowd.json'
        
        if not aicrowd_path.exists():
            print(f"Warning: {aicrowd_path} not found, defaulting to not starting vLLM")
            return False
            
        with open(aicrowd_path, 'r') as f:
            config = json.load(f)
        use_gpu = config.get('gpu', False)
        print("use_gpu", use_gpu)
        return use_gpu
        
    except Exception as e:
        print(f"Error reading aicrowd.json: {e}")
        return False

def start_vllm_server(LLM_MODEL= "meta-llama/Llama-3.1-8B-Instruct"):
    """Start vLLM server in the background"""
    print(f"Starting vLLM server with {LLM_MODEL}...")
    extra_args = []
    if "llama-3.3" in LLM_MODEL.lower():
        extra_args = [
            "--tool-call-parser", "llama3_json",
            "--chat-template", Path(__file__).parent / "llama32.jinja",
            "--enable-auto-tool-choice"
        ]
    elif "llama-3.1" in LLM_MODEL.lower():
        extra_args = [
            "--tool-call-parser", "llama3_json",
            "--chat-template", Path(__file__).parent / "llama31.jinja",
            "--enable-auto-tool-choice"
        ]
    elif LLM_MODEL in ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct", 
    "Qwen/Qwen2.5-70B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen3-8B", "Qwen/Qwen3-14B"
    ]:
        extra_args = [
            "--tool-call-parser", "hermes",
            "--enable-auto-tool-choice"
        ]
    elif "qwen" in LLM_MODEL.lower():
        extra_args = [
            "--tool-call-parser", "hermes",
            "--enable-auto-tool-choice"
        ]
    elif "xlam" in LLM_MODEL.lower():
        xlam_PATH = Path(__file__).parent / "xlam_tool_call_parser.py"
        extra_args = [
            "--tool-call-parser", "xlam",
            "--tool-parser-plugin", xlam_PATH,
            "--enable-auto-tool-choice"
        ]

    else:
        extra_args = [
            "--tool-call-parser", "hermes",
            "--enable-auto-tool-choice"
        ]
    # if "gptq" in LLM_MODEL.lower():
    #     extra_args += [
    #         "--quantization", "gptq"
    #     ]
    # elif "awq" in LLM_MODEL.lower():
    #     extra_args += [
    #         "--quantization", "awq"
    #     ]
    import torch
    available_devices = torch.cuda.device_count()
    # Command to start the vLLM server
    cmd = [
        "vllm", "serve", LLM_MODEL,
        "--api-key", "123",
        "--gpu-memory-utilization", "0.92",
        "--port", "8112",
        "--max-model-len", "8000",
        "--tensor-parallel-size", str(available_devices),
        "--trust-remote-code",
        "--max-num-seqs", "32"
    ]
    cmd.extend(extra_args)
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create a safe filename from the model name
    model_log_name = LLM_MODEL.replace('/', '_').replace('.', '_')
    log_file = os.path.join(logs_dir, f'vllm_server_{model_log_name}.log')
    
    print(f"Redirecting vLLM server output to: {os.path.abspath(log_file)}")
    
    # Start the process in the background with output to log file
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid
        )
    
    # Function to clean up the server process
    def cleanup():
        print("\nStopping vLLM server...")
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
    
    # Register cleanup function to run at exit
    atexit.register(cleanup)
    WAIT_SEC = int(os.getenv("WAIT_SEC", 500))
    # Wait for the server to start by polling the log file
    print(f"Waiting for vLLM server to initialize (timeout: {WAIT_SEC} seconds)...")
    start_time = time.time()
    server_ready = False
    while time.time() - start_time < WAIT_SEC:
        if process.poll() is not None:
            print("vLLM server process terminated unexpectedly. Check logs for details.")
            break
        
        time.sleep(5)
        try:
            with open(log_file, 'r') as f:
                log_content = f.read()
                # Check for multiple possible success messages from vLLM or Uvicorn
                success_messages = [
                    "Started server process", 
                    # "Starting vLLM API server"
                ]
                if any(msg in log_content for msg in success_messages):
                    print("vLLM server started successfully.")
                    server_ready = True
                    break
        except FileNotFoundError:
            # Log file might not be created immediately, continue waiting
            pass
        except Exception as e:
            print(f"Error reading log file: {e}")
    
    if not server_ready:
        print(f"Warning: Timed out waiting for vLLM server to start. Check the log file for errors: {os.path.abspath(log_file)}")
    else:
        print("vLLM server should be ready now.")
    
    return process

# Set environment variables for OpenAI API
def setup_environment(LLM_MODEL):
    os.environ["OPENAI_API_KEY"] = "123"
    if os.getenv("IS_LOCAL", "false") == "true":
        os.environ["OPENAI_BASE_URL"] = "http://0.0.0.0:8112/v1"  # 本地用这个
    else:
        os.environ["OPENAI_BASE_URL"] = "http://localhost:8112/v1"  # aicrowd用这个
    os.environ["OPENAI_MODEL"] = LLM_MODEL
    # os.environ["DEBUG"] = "1"
    # os.environ["USE_RAG"] = "0"

# Only start vLLM server if GPU is enabled in aicrowd.json
if should_start_vllm():
    print("GPU is enabled in aicrowd.json, starting vLLM server...")
    LLM_MODEL= get_model_name()
    vllm_process = start_vllm_server(LLM_MODEL)
    setup_environment(LLM_MODEL)
else:
    print("GPU is not enabled in aicrowd.json, using api configuration")

# Set the UserAgent to use OpenAI
UserAgent = OpenAIAgent
