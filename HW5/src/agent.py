import os
import pdb
import yaml
import openai
import logging
import torch
import aiohttp  # for making API calls concurrently
import asyncio  # for running API calls concurrently
import json  # for saving results to a jsonl file
import logging  # for logging rate limit warnings and other messages
import tiktoken  # for counting tokens
import time  # for sleeping after rate limit is hit
from dataclasses import dataclass, field  # for storing API inputs, outputs, and metadata
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from typing import Union, List, Dict
    
class HuggingFaceAgent:
    def __init__(self, model_path: str, complete_config_path: str, quant_config: Dict = None, peft_path: str = None):
        if quant_config is None:
            quant_config = {
                'load_in_4bit': True,
                'bnb_4bit_quant_type': 'nf4',
                'bnb_4bit_compute_dtype': torch.bfloat16
            }
        quant_config = BitsAndBytesConfig(**quant_config)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=quant_config, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if peft_path:
            self.model = PeftModel.from_pretrained(self.model, peft_path)
        self.pipeline = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer)

        with open(complete_config_path, 'r', encoding='utf8') as fi:
            complete_config = yaml.safe_load(fi)
            self.prompt_template = complete_config.pop('prompt_template')
            self.text_generation_config = complete_config
            self.text_generation_config['eos_token_id'] = self.tokenizer.eos_token_id

    def complete(self, messages: Union[Dict[str, str], List[Dict[str, str]]], **kwargs):
        if isinstance(messages, dict):
            return self._complete_single(messages, **kwargs)
        elif isinstance(messages, list):
            return self._complete_multiple(messages, **kwargs)
        else:
            return

    def _complete_single(self, message: Dict, **kwargs):
        target_generation_config = self.text_generation_config.copy()
        target_generation_config.update(kwargs)
        inp  = self._get_prompt(message)
        oup = self.pipeline(inp, **target_generation_config)[0]['generated_text']
        
        return oup, oup[len(inp):]


    def _complete_multiple(self, messages: List[Dict[str, str]], batch_size: int = 4, **kwargs):
        target_generation_config = self.text_generation_config.copy()
        target_generation_config.update(kwargs)
        cue = '{\n  '
        inps = [self._get_prompt(message, cue) for message in messages]
        oups = []
        for i in range(0, len(inps), batch_size):
            local_inps = inps[i: i + batch_size]
            oups.extend([res[0]['generated_text'] for res in self.pipeline(local_inps, **target_generation_config)])

        return oups, [oup[len(inps[i]) - len(cue): ] for i, oup in enumerate(oups)]

    def _get_prompt(self, message: Dict, cue: str = ''):
        example_count = 0
        keys = ['instruction', 'input', 'output']
        for key in keys:
            if key in message:
                example_count = max(example_count, len(message[key]))

        prompt_parts = []
        for i in range(example_count):
            for key in keys:
                if key in message and i < len(message[key]):
                    prompt_parts.append(
                        self.prompt_template[key].format(message[key][i])
                    )
        prompt_parts.append(self.prompt_template['output'].format(cue))
        
        generated_prompt = '\n\n'.join(prompt_parts)

        logging.debug(f"Generated prompt:\n{generated_prompt}")

        return generated_prompt


class OpenAIAgent:
    def __init__(self, model: str, complete_config_path: str, 
                 api_key: str = None, max_attempts: int = 5, 
                 max_tokens_per_minute: float = 250_000 * 0.5, max_requests_per_minute: float = 3_000 * 0.5):
        self.model = model
        with open(complete_config_path, 'r', encoding='utf8') as fi:
            self.complete_config = yaml.safe_load(fi)
            self.prompt_template = self.complete_config.pop('prompt_template')
        
        # constants
        self.seconds_to_pause_after_rate_limit_error = 15
        self.seconds_to_sleep_each_loop = 0.001

        # infer API endpoint and construct request header
        self.api_key = os.getenv("OPENAI_API_KEY") if api_key is None else api_key
        self.request_url = 'https://api.openai.com/v1/chat/completions'
        self.api_endpoint = 'chat/completions'
        self.token_encoding_name = 'cl100k_base'
        self.request_header = {"Authorization": f"Bearer {self.api_key}"}
        self.max_attempts = max_attempts

        # initialize available capacity counts
        self.max_requests_per_minute = max_requests_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute
        self.available_request_capacity = max_requests_per_minute
        self.available_token_capacity = max_tokens_per_minute
        self.last_update_time = time.time()
        
    async def complete(self, messages: Union[Dict[str, str], List[Dict[str, str]]], **kwargs):
        if isinstance(messages, dict):
            return await self._complete_single(messages, **kwargs)
        elif isinstance(messages, list):
            return await self._complete_multiple(messages, **kwargs)
        else:
            return

    async def _complete_single(self, message: Dict, **kwargs):
        return await self._complete_multiple([message], **kwargs)
    
    async def _complete_multiple(self, messages: List[Dict[str, str]], **kwargs):
        # initialize trackers
        queue_of_requests_to_retry = asyncio.Queue()
        task_id_generator = task_id_generator_function()  # generates integer IDs of 1, 2, 3, ...
        status_tracker = StatusTracker()  # single instance to track a collection of variables
        next_request = None  # variable to hold the next request to call

        # initialize others
        sample_index = 0
        results = []
        logging.debug(f"Initialization complete and entering main loop.")

        while True:
            if next_request is None:
                if not queue_of_requests_to_retry.empty():
                    next_request = queue_of_requests_to_retry.get_nowait()
                    logging.debug(f"Retrying request {next_request.task_id}: {next_request}")
                elif sample_index < len(messages):
                    request_json = self._get_request_json(messages[sample_index], kwargs)
                    sample_index += 1
                    next_request = APIRequest(
                        task_id=next(task_id_generator),
                        request_json=request_json,
                        token_consumption=num_tokens_consumed_from_request(request_json, self.api_endpoint, self.token_encoding_name),
                        attempts_left=self.max_attempts,
                        metadata=request_json.pop("metadata", None)
                    )
                    status_tracker.num_tasks_started += 1
                    status_tracker.num_tasks_in_progress += 1
                    logging.debug(f"Reading request {next_request.task_id}: {next_request}")
            
            # update available capacity
            current_time = time.time()
            seconds_since_update = current_time - self.last_update_time
            self.available_request_capacity = min(
                self.available_request_capacity + self.max_requests_per_minute * seconds_since_update / 60.0,
                self.max_requests_per_minute,
            )
            self.available_token_capacity = min(
                self.available_token_capacity + self.max_tokens_per_minute * seconds_since_update / 60.0,
                self.max_tokens_per_minute,
            )      
            self.last_update_time = current_time

            # if enough capacity available, call API
            if next_request:
                next_request_tokens = next_request.token_consumption
                if (
                    self.available_request_capacity >= 1
                    and self.available_token_capacity >= next_request_tokens
                ):
                    # update counters
                    self.available_request_capacity -= 1
                    self.available_token_capacity -= next_request_tokens
                    next_request.attempts_left -= 1

                    # call API
                    asyncio.create_task(
                        next_request.call_api(
                            request_url=self.request_url,
                            request_header=self.request_header,
                            retry_queue=queue_of_requests_to_retry,
                            results=results,
                            status_tracker=status_tracker,
                        )
                    )
                    next_request = None  # reset next_request to empty
            
            # if all tasks are finished, break
            if status_tracker.num_tasks_in_progress == 0:
                break
            
            # main loop sleeps briefly so concurrent tasks can run
            await asyncio.sleep(self.seconds_to_sleep_each_loop)
            
            # if a rate limit error was hit recently, pause to cool down
            seconds_since_rate_limit_error = (time.time() - status_tracker.time_of_last_rate_limit_error)
            if seconds_since_rate_limit_error < self.seconds_to_pause_after_rate_limit_error:
                remaining_seconds_to_pause = (self.seconds_to_pause_after_rate_limit_error - seconds_since_rate_limit_error)
                await asyncio.sleep(remaining_seconds_to_pause)
                # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
                logging.warn(f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + self.seconds_to_pause_after_rate_limit_error)}")


        # after finishing, log final status
        logging.info(f"""Parallel processing complete. Results were saved.""")
        if status_tracker.num_tasks_failed > 0:
            logging.warning(f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors were logged.")
        if status_tracker.num_rate_limit_errors > 0:
            logging.warning(f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate.")
        
        with open('dump/call_openai_results.json', 'w', encoding='utf8') as fo:
            json.dump(results, fo, indent=4, ensure_ascii=False)

        results.sort(key=lambda x: x[2])

        return results, [(result[1]['choices'][0]['message']['content'] if isinstance(result[1], dict) else '<ERROR>') for result in results]

    def _get_request_json(self, message: Dict, complete_config: Dict):
        request_json = {
            'model': self.model,
        }
        target_complete_config = self.complete_config.copy()
        target_complete_config.update(complete_config)

        request_json.update(target_complete_config)

        if 'input' not in message:
            message['input'] = message.pop('instruction')
        
        messages = [{"role": "system", "content": self.prompt_template['instruction'].format(message.get('instruction', [''])[0])}]

        example_count = 0
        keys = ['input', 'output']
        key2role = {'input': 'user', 'output': 'assistant'}
        for key in keys:
            if key in message:
                example_count = max(example_count, len(message[key]))
        
        for i in range(example_count):
            for key in keys:
                if key in message and i < len(message[key]):
                    messages.append(
                        {
                            "role": key2role[key],
                            "content": self.prompt_template[key].format(message[key][i])
                        }
                    )
        
        request_json["messages"] = messages
        return request_json


@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits

@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""

    task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    metadata: dict
    result: list = field(default_factory=list)

    async def call_api(
        self,
        request_url: str,
        request_header: dict,
        retry_queue: asyncio.Queue,
        results: list,
        status_tracker: StatusTracker,
    ):
        """Calls the OpenAI API and saves results."""
        logging.info(f"Starting request #{self.task_id}")
        error = None
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url=request_url, headers=request_header, json=self.request_json,
                    proxy='http://127.0.0.1:7890'
                ) as response:
                    response = await response.json()
            if "error" in response:
                logging.warning(
                    f"Request {self.task_id} failed with error {response['error']}"
                )
                status_tracker.num_api_errors += 1
                error = response
                if "Rate limit" in response["error"].get("message", ""):
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= 1  # rate limit errors are counted separately

        except Exception as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            logging.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e
        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logging.error(f"Request {self.request_json} failed after all attempts. Saving errors: {self.result}")
                data = (
                    [self.request_json, [str(e) for e in self.result], self.task_id, self.metadata]
                    if self.metadata
                    else [self.request_json, [str(e) for e in self.result], self.task_id]
                )
                results.append(data)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            data = (
                [self.request_json, response, self.task_id, self.metadata]
                if self.metadata
                else [self.request_json, response, self.task_id]
            )
            results.append(data)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            logging.debug(f"Request {self.task_id} was saved.")


def num_tokens_consumed_from_request(
    request_json: dict,
    api_endpoint: str,
    token_encoding_name: str,
):
    """Count the number of tokens in the request. Only supports completion and embedding requests."""
    encoding = tiktoken.get_encoding(token_encoding_name)
    # if completions request, tokens = prompt + n * max_tokens
    if api_endpoint.endswith("completions"):
        max_tokens = request_json.get("max_tokens", 15)
        n = request_json.get("n", 1)
        completion_tokens = n * max_tokens

        # chat completions
        if api_endpoint.startswith("chat/"):
            num_tokens = 0
            for message in request_json["messages"]:
                num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":  # if there's a name, the role is omitted
                        num_tokens -= 1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
            return num_tokens + completion_tokens
        # normal completions
        else:
            prompt = request_json["prompt"]
            if isinstance(prompt, str):  # single prompt
                prompt_tokens = len(encoding.encode(prompt))
                num_tokens = prompt_tokens + completion_tokens
                return num_tokens
            elif isinstance(prompt, list):  # multiple prompts
                prompt_tokens = sum([len(encoding.encode(p)) for p in prompt])
                num_tokens = prompt_tokens + completion_tokens * len(prompt)
                return num_tokens
            else:
                raise TypeError('Expecting either string or list of strings for "prompt" field in completion request')
    # if embeddings request, tokens = input tokens
    elif api_endpoint == "embeddings":
        input = request_json["input"]
        if isinstance(input, str):  # single input
            num_tokens = len(encoding.encode(input))
            return num_tokens
        elif isinstance(input, list):  # multiple inputs
            num_tokens = sum([len(encoding.encode(i)) for i in input])
            return num_tokens
        else:
            raise TypeError('Expecting either string or list of strings for "inputs" field in embedding request')
    # more logic needed to support other API calls (e.g., edits, inserts, DALL-E)
    else:
        raise NotImplementedError(f'API endpoint "{api_endpoint}" not implemented in this script')

def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1

async def test():

    # initialize logging
    logging.basicConfig(level=10)
    logging.debug(f"Logging initialized at level {10}")

    # tasks
    from task import STS, TruthfulQA, SQuAD
    # sts = STS('mteb/sts12-sts')
    # messages = sts.format_input()[:2]
    # tqa = TruthfulQA('truthful_qa')
    # messages = tqa.format_input()[:2]
    squad = SQuAD('squad')
    messages = squad.format_input()[:4]

    ''' OpenAI
    complete_config_path = 'config/openai.yml'
    agent = OpenAIAgent('gpt-3.5-turbo', complete_config_path)
    res = await agent.complete(messages)
    '''

    ''' LLaMa
    model_path = r'/home/incoming/llm/llama/llama-30b'
    peft_path = r'/home/incoming/llm/llama/SuperCOT-LoRA/30b/gpu/cutoff-1024'
    complete_config_path = r'config/llama.yml'
    agent = HuggingFaceAgent(model_path, complete_config_path, peft_path=peft_path)
    res = agent.complete(messages)
    '''

    # ''' Falcon
    model_path = r'/home/incoming/llm/falcon/falcon-40b-instruct'
    complete_config_path = r'config/falcon.yml'
    agent = HuggingFaceAgent(model_path, complete_config_path)
    res = agent.complete(messages)
    # '''
    
    import code
    code.interact(local=locals())


if __name__ == '__main__':
    asyncio.run(test())