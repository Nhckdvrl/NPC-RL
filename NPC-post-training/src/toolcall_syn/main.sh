export OPENAI_API_MODEL="gpt-4o"
export OPENAI_API_TYPE="azure_msra"
pip install azure.identity
# --input "/path/to/npc-rl/data/sft/task1/stage_0.json" --output "/path/to/npc-rl/data/gpt-4o-toolcall-sft/v1.json" 
python /path/to/npc-rl/src/toolcall-syn/data_augmentation.py --model "gpt-4o"  
