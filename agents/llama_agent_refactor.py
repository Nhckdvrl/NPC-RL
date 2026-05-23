from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import copy
import json

class LlamaAgentRefactor(object):
    """
    A simple agent implementation for the Sony CPDC challenge.
    It calls a LLaMA-3.1-8B-Instruct model twice per turn:
      1) to determine appropriate functions to call,
      2) to generate a user-facing response based on context.

    Public Interface:
      - __init__()
      - generate_functions_and_responses(...)
    """

    def __init__(self):
        """
        Loads the model and tokenizer.
        (Signature must remain unchanged.)
        """
        model_path = 'meta-llama/Llama-3.1-8B-Instruct'
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16, 
            device_map='auto'
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        self.max_seq_len = 4096
        self.max_func_len = 1024
        self.max_resp_len = 400

        # for testing structured output successes and failures
        self.success_json = 0
        self.failure_json = 0

    def generate_functions_and_responses(
        self, 
        tool_registry: Dict[str, Dict[str, Dict[str, str]]],
        action_registry: Dict[str, Dict[str, Dict[str, str]]],
        worldview: str,
        persona: Dict[str, str],
        role: str,
        knowledge: Dict[str, any],
        state: Dict[str, str],
        dialogue: List[Dict[str, any]],
        executor
    ) -> Dict[str, str]:
        """
        1) Ask the LLM which functions to call and parse them.
        2) Execute those functions via 'executor'.
        3) Ask the LLM for a final user-facing response, including function results.

        (Signature must remain unchanged.)
        """

        # STEP 1: Construct a prompt to estimate function calls
        func_messages = self._create_function_estimation_messages(
            tool_registry, action_registry, dialogue
        )
        # STEP 2: Generate text suggesting function calls
        generated_funcs_text = self._invoke_llm(func_messages, max_gen_len=self.max_func_len)
        # STEP 3: Parse those function calls
        functions_to_call = self._parse_generated_function_calls(generated_funcs_text)
        # STEP 4: Execute them
        function_results = executor.execute(functions_to_call)

        # STEP 5: Construct messages for the final response
        response_messages = self._create_final_response_messages(
            worldview, persona, role, knowledge, state, dialogue, function_results
        )
        # STEP 6: Generate the final user-facing response
        final_response = self._invoke_llm(response_messages, max_gen_len=self.max_resp_len)

        return {
            'prompts': response_messages, 
            'final_responses': final_response
        }

    # --------------------------------------------------------------------------
    #                           PRIVATE HELPERS
    # --------------------------------------------------------------------------

    def _create_function_estimation_messages(
        self,
        tool_registry: Dict[str, Dict[str, Dict[str, str]]],
        action_registry: Dict[str, Dict[str, Dict[str, str]]],
        dialogue: List[Dict[str, any]]
    ) -> List[Dict[str, str]]:
        """
        Create a prompt describing the available functions and the user's last query.
        """
        # Combine function info from both registries
        function_info = self._format_function_information(tool_registry, action_registry)
        additional_info = self._format_additional_information(dialogue)
        last_user_text = dialogue[-1]["text"] if dialogue else ""

        base_prompt = (
            "# Instruction\n"
            "You are an assistant in estimating function names and arguments.\n"
            "You need information to answer the question entered by the user.\n"
            "Use the following steps to estimate the needed function calls. \n"
            "1. From the function info, choose a function if needed.\n"
            "2. Create arguments for it.\n\n\n"
            f"## Function Information\n\n{function_info}\n\n"
            f"## Additional Information\n{additional_info}\n\n"
            "Please generate a list of json-formatted dictionaries.\n"
            "Each dictionary corresponds to a function call you want to make. \n"
            "Wrap your generated list of json-formatted dictionaries with <json> and </json>.\n"
            "Here are some examples: \n"
            "Generation example 1: <json>[]</json> # no need for function calls\n"
            'Generation example 2: <json>[{"name": "equip", "parameters": {"item": "sword"}}]</json> # one function call to the function equip, with the "item" parameter set to "sword". \n'
        )

        return [
            {"role": "system", "content": base_prompt},
            {"role": "user", "content": last_user_text}
        ]

    def _create_final_response_messages(
        self,
        worldview: str,
        persona: Dict[str, str],
        role: str,
        knowledge: Dict[str, any],
        state: Dict[str, str],
        dialogue: List[Dict[str, any]],
        function_results: List[Dict[str, any]]
    ) -> List[Dict[str, str]]:
        """
        Builds a prompt including persona, knowledge, and conversation history
        for generating the final response.
        """
        worldview_combined = f"{worldview}\n{knowledge.get('general_info', '')}"
        persona_text = self._format_persona(persona)
        knowledge_text = self._format_knowledge(function_results, knowledge)

        # Create a minimal system prompt.
        system_prompt = (
            "You are an assistant that plays the role of a character in a game.\n"
            "Use the character settings, the knowledge, and the worldview below to respond to the conversation.\n\n"
            "# Character Setting\n"
            f"{persona_text}\n\n"
            "# Knowledge\n"
            f"{knowledge_text}\n\n"
            "# Worldview\n"
            f"{worldview_combined}\n\n"
        )

        history_messages = self._build_conversational_history(dialogue)
        return [{"role": "system", "content": system_prompt}] + history_messages

    def _build_conversational_history(self, dialogue: List[Dict[str, any]]) -> List[Dict[str, str]]:
        """Convert dialogue dicts into message-role pairs."""
        return [
            {
                "role": "assistant" if turn.get("speaker") == "npc" else "user",
                "content": turn.get("text", "")
            }
            for turn in dialogue
        ]

    def _format_function_information(
        self,
        tool_registry: Dict[str, Dict[str, Dict[str, str]]],
        action_registry: Dict[str, Dict[str, Dict[str, str]]]
    ) -> str:
        """
        Combines tool and action function details into a single text block.
        """
        registry_blocks = []
        
        # A small helper to format each registry
        def registry_to_blocks(registry):
            for fn_info in registry["function_registry"].values():
                name = fn_info["name"]
                desc = fn_info["description"]
                yield f"# Function Name: {name}\n# Function Docstring:\n{desc}\n"

        # Gather from both tool and action
        registry_blocks.extend(registry_to_blocks(tool_registry))
        registry_blocks.extend(registry_to_blocks(action_registry))

        return "\n".join(registry_blocks)

    def _format_additional_information(self, dialogue: List[Dict[str, any]]) -> str:
        """Formats 'target_item' from the last dialogue turn, if present."""
        if not dialogue or "target_item" not in dialogue[-1]:
            return ""
        items = dialogue[-1]["target_item"] or []
        return "\n".join(
            f"parameter name: name, value: {item.get('name','')}" 
            for item in items
        )

    def _format_persona(self, persona: Dict[str, str]) -> str:
        """Returns persona as a newline-separated string of key-value pairs."""
        return "\n".join(f"{k}: {v}" for k, v in persona.items())

    def _format_knowledge(
        self,
        function_results: List[Dict[str, any]],
        knowledge: Dict[str, any]
    ) -> str:
        """Converts function call results + knowledge items into a single text block."""
        lines = []
        # Append function result details
        for f_res in function_results:
            # parameters
            for k, v in f_res.get("parameters", {}).items():
                lines.append(f"{k}: {v}")
            # returned data
            for ret_item in f_res.get("return", []):
                for rk, rv in ret_item.items():
                    lines.append(f"{rk}: {rv}")

        # Append domain knowledge
        for item in knowledge.get("knowledge_info", []):
            # item is a dict, e.g. {"some_key": "some_val", ...}
            lines.append(", ".join(f"{k}: {v}" for k, v in item.items()))

        return "\n".join(lines)
    
    

    def _parse_generated_function_calls(self, generated_text: str) -> List[Dict[str, any]]:
        """
        Parses json-formatted strings to extract function names and params. 
        """
        json_text = generated_text.split("<json>")[-1].split("</json>")[0]
        try: 
            parsed_json = json.loads(json_text)
            self.success_json += 1
            if isinstance(parsed_json, list):
                return parsed_json
            else:
                return [parsed_json]
        except: 
            self.failure_json += 1
            return []

    def _invoke_llm(self, messages: List[Dict[str, str]], max_gen_len: int) -> str:
        """
        Calls the underlying LLM to get a generated text response given the conversation history.
        """
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors='pt'
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            num_beams=1,
            do_sample=False,
            temperature=None,
            top_p=None,
            max_new_tokens=max_gen_len,
            eos_token_id=self.terminators,
            pad_token_id=self.tokenizer.eos_token_id
        )
        # Extract the newly generated tokens after the prompt
        generated = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).replace("\n", " ")