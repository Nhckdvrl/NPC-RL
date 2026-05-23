from typing import List, Dict

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import copy

class VanillaLlamaAgent(object):
    """
        A simple agent implementation for the Sony CPDC challenge. 
        It calls a LLaMA-3.1-8B-Instruct model twice per turn. 
        The first call aims to find appropriate functions to call, 
        and the second generates responses. 
    """
    def __init__(self):
        """Load necesasry models and configurations here"""
        # Please specify all HF model paths in `aicrowd.json`. 
        model_path = 'meta-llama/Llama-3.1-8B-Instruct'
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='auto')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        self.max_seq_len = 2048
        self.max_gen_len = 50
    
    def _create_messages_for_function(self, tool_functions, action_functions, dialogue):
        """
            Formats function information into prompt. 
            'tool_functions' and 'action_functions' are both function registries
            that supports indexing a function with its name
        """
        base_prompt = """# Instruction
You are an assistant in estimating function names and arguments.
You need information to answer the text entered by the user.
Use the following steps to perform function's estimation.
1. From the function information below, select a function that can obtain the information you need.
2. Make the arguments needed by the function as appropriate.
The format of the output is:
function name: xxx
argument name: xxx, value: xxx

## Function Information
{}

## Additional Information
{}"""
        function_information = []
        for tool_name in tool_functions['function_registry'].keys():
            tool_ = tool_functions['function_registry'][tool_name]
            tool_prompt = """
# Function Name: {}
# Function Docstring: 
{}
            """.format(tool_['name'], tool_['description'])
            function_information.append(tool_prompt)
        for action_name in action_functions['function_registry'].keys():
            action_ = action_functions['function_registry'][action_name]
            action_prompt = """
# Function Name: {}
# Function Docstring: 
{}
            """.format(action_['name'], action_['description'])
            function_information.append(action_prompt)
        function_information_agg = '\n'.join(function_information)
        additional_info = ""
        for info in dialogue[-1]["target_item"]:
            if additional_info == "":
                additional_info = "parameter name: name, value: " + info["name"]
            else:
                additional_info = additional_info + "\nargument name: name, value: " + info["name"]

        input_text = dialogue[-1]["text"]
        prompt = base_prompt.format(function_information_agg, additional_info)
        messages = []
        messages.append({"role":"system", "content":prompt})
        messages.append({"role":"user", "content":input_text})

        return messages

    def _create_messages_for_dialogue(self, worldview, persona, role, knowledge, state, dialogue, function_results):
        base_prompt = """\
You are an assistant who becomes a character.
Use the following character settings and knowledge to create your response.

# Character Setting
[Here is Character Setting]

# Knowledge
[Here is Knowledge]"""

        worldview = worldview + "\n" + knowledge["general_info"]
        
        character_setting = ""
        for key in persona:
            if character_setting == "":
                character_setting = key + ": " + persona[key]
            else:
                character_setting = character_setting + "\n" + key + ": " + persona[key]

        knowledge_setting = ""
        for f_result in function_results:
            if knowledge_setting != "":
                knowledge_setting = knowledge_setting + "\n"
            for arg in f_result["parameters"]:
                if knowledge_setting == "":
                    knowledge_setting = arg + ": " + str(f_result["parameters"][arg])
                else:
                    knowledge_setting = knowledge_setting + ", " + arg + ": " + str(f_result["parameters"][arg])
            for item in f_result["return"]:
                for key in item:
                    if knowledge_setting == "":
                        knowledge_setting = key + ": " + item[key]
                    else:
                        knowledge_setting = knowledge_setting + ", " + key + ": " + item[key]
                        
        for item in knowledge["knowledge_info"]:
            if knowledge_setting != "":
                knowledge_setting = knowledge_setting + "\n"
            for key in item:
                if knowledge_setting == "":
                    knowledge_setting = key + ": " + item[key]
                else:
                    knowledge_setting = knowledge_setting + ", " + key + ": " + item[key]

                    
        history_list = []
        for item in dialogue:
            role = "user"
            if item["speaker"] == "npc":
                role = "assistant"
            history_list.append({"role":role, "content":item["text"]})

        prompt = base_prompt.replace("[Here is Worldview]", worldview)
        prompt = prompt.replace("[Here is Role]", role)
        prompt = prompt.replace("[Here is Character Setting]", character_setting)
        prompt = prompt.replace("[Here is Knowledge]", knowledge_setting)
        
        messages = []
        messages.append({"role":"system", "content":prompt})
        messages.extend(history_list)

        return messages
    
    def generate_functions_and_responses(self, tool_registry, action_registry, worldview, persona, role, knowledge, state, dialogue, executor):
        """
        Given the background information, perform adequate function calls, and based on the function call results, generate coherent and reasonable responses.
        This agent does the following four steps. 
        1. Ask the LLM to generate the necessary functions to call. 
        2. Parse the generation to obtain function names and arguments to call. 
        3. Call the `executor` to obtain results for the function calls. 
        4. Call the LLM to generate responses based on the background information and the function call results. 

        Parameters
        ----------
            tool_registry, action_registry: Dict[str, Dict[str, str]] are function registries for tool and action functions, 
                from which we can index functions with function names. 
                For example, you can index a tool function named 'tool1' via `tool_registry['function_registry']['tool1'][k]`, 
                where k can be 'args', 'description', or 'name'.  
            worldview: str, the worldview of the current dialogue. 
            persona: Dict[str, str], describes the persona of the NPC, e.g. persona['name'], persona['age'], persona['gender'], persona['occupation']. 
                See the sample and training datasets for details. 
            role: str, the role of the NPC. 
            knowledge: Dict[str, Any], contains basic knowledge about the items (e.g. quests, weapons). See the sample and training datasets for details. 
            state: Dict[str, str], the time, location, etc. of the current conversation. 
            dialogue: List[Dict[str, str]]. It records the previous turns of the dialogue. 
                Each dict in the list is of the following format: 
                {
                    "speaker": ..., 
                    "text": ..., 
                    "target_item": ...
                }
            executor: It is a module that can execute function calls you need and record the history of all function calls you make. 
                Call the executor with `executor.execute(function_items)`, where `function_items` 
                is a list of dictionaries containing all function calls to make. 
                Each dictionary in `function_items` should have the following format: 
                    {
                        'name': <function_name>, 
                        'parameters': {
                            <param_name>: <param_val>, 
                            ...
                        } 
                    }


        Returns
        ----------
            Dict[str, str] with the following structure. 
                {
                    "prompts": Optional. The prompt of the current turn. 
                    "final_responses": Your response of the current turn. 
                }
        
        NOTE: You do not need to return the generated function calls. The `executor` will automatically record that. 
        """

        
        # Step 1: In our first call to the LLM, we ask it to generate all necessary functions to call.  
        messages_func = self._create_messages_for_function(tool_registry, action_registry, dialogue)
        input_ids = self.tokenizer.apply_chat_template(
            messages_func, 
            add_generation_prompt=True, 
            return_tensors='pt'
        ).to(self.model.device)
        outputs = self.model.generate(
            input_ids,
            num_beams=1,
            do_sample=False,
            temperature=None,
            top_p=None, 
            max_new_tokens=self.max_gen_len,
            eos_token_id=self.terminators,
            pad_token_id=self.tokenizer.eos_token_id
        )
        res = outputs[0][input_ids.shape[-1]:]        
        items = self.tokenizer.decode(res, skip_special_tokens=True).split("\n")

        # Step 2: Parse the generated function call results.  
        final_functions = []
        res_item = {}
        for item in items:
            if "function name: " in item:
                if "name" in res_item:
                    final_functions.append(copy.deepcopy(res_item))
                    res_item = {}
                res_item["name"] = item.replace("function name: ", "").split(",")[0]
                res_item["parameters"] = {}
            elif "argument name: " in item:
                arg_name = ""
                arg_val = ""
                if "value: " in item:
                    arg_name = item.split("value: ")[0].replace("argument name: ", "").split(",")[0]
                    arg_val = item.split("value: ")[1]

                if arg_name != "":
                    if not "parameters" in res_item:
                        res_item["parameters"] = {}
                    res_item["parameters"][arg_name] = arg_val
                    
        if "name" in res_item:
            final_functions.append(res_item)

        # Step 3: Obtain the return values for all function call results via the executor. 
        function_results = executor.execute(final_functions)

        # Step 4: Based on the function call results, generate response. 
        messages_resp = self._create_messages_for_dialogue(worldview, persona, role, knowledge, state, dialogue, function_results)
        input_ids = self.tokenizer.apply_chat_template(
            messages_resp,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            num_beams=1,
            do_sample=False,
            temperature=None,
            top_p=None,
            max_new_tokens=self.max_gen_len,
            eos_token_id=self.terminators,
            pad_token_id=self.tokenizer.eos_token_id
        )
        res = outputs[0][input_ids.shape[-1]:]
        res_str = self.tokenizer.decode(res, skip_special_tokens=True).replace("\n", " ")
        
        # However, participants can do more than that, like back and forth calling of functions. 

        # Only return response. We don't return function calls as it is recorded in the 'executor'. 
        return {
            'prompts': messages_resp, 
            'final_responses': res_str
        }
        
        
