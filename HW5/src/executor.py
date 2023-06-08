import re
import pdb
import sys
import json
import string
import asyncio
import logging
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
from agent import HuggingFaceAgent, OpenAIAgent
from task import STS, TruthfulQA, SQuAD
from collections import Counter

class Executor:
    def __init__(self, agent_name: str, task_name: str):
        self.agent_name = agent_name.lower()
        self.task_name = task_name.lower()
        if agent_name.lower() == 'openai':
            complete_config_path = 'config/openai.yml'
            self.agent = OpenAIAgent('gpt-3.5-turbo', complete_config_path)
            self.async_mode = True
        elif agent_name.lower() == 'llama':
            model_path = r'/home/incoming/llm/llama/llama-30b'
            peft_path = r'/home/incoming/llm/llama/SuperCOT-LoRA/30b/gpu/cutoff-1024'
            complete_config_path = r'config/llama.yml'
            self.agent = HuggingFaceAgent(model_path, complete_config_path, peft_path=peft_path)
            self.async_mode = False
        elif agent_name.lower() == 'falcon':
            model_path = r'/home/incoming/llm/falcon/falcon-40b-instruct'
            complete_config_path = r'config/falcon.yml'
            self.agent = HuggingFaceAgent(model_path, complete_config_path)
            self.async_mode = False
        else:
            raise NotImplementedError

        if task_name.lower() == 'sts':
            self.task = STS('mteb/sts12-sts')
        elif task_name.lower() == 'truthful_qa':
            self.task = TruthfulQA('truthful_qa')
        elif task_name.lower() == 'squad':
            self.task = SQuAD('squad')
        else:
            raise NotImplementedError
    
    async def run(self, with_example: bool = False, with_cot: bool = False):
        messages = self.task.format_input(with_example, with_cot)
        
        if self.async_mode:
            res = await self.agent.complete(messages)
        else:
            res = self.agent.complete(messages)
        
        self.task.save_results(res[1], f'dump/{self.agent_name}-{self.task_name}-{with_example}-{with_cot}-results')

class Metrics:
    def __init__(self, task_name: str, results_path: str):
        self.task_name = task_name.lower()

        with open(results_path, 'r', encoding='utf8') as fo:
            self.results = json.load(fo)

        self._extract_result()

    def _extract_result(self):
        if self.task_name == 'sts':
            key = 'score'
            data_type = float
        elif self.task_name == 'truthful_qa':
            key = 'sequence'
            data_type = int
        elif self.task_name == 'squad':
            key = 'answer'
            data_type = str
        else:
            raise NotImplementedError
        res_p = re.compile(rf'({{(?:.|\n)*?"(?:{key.capitalize()}|{key})":(?:.|\n)*?}})')

        total_count = 0
        success_count = 0
        for result in tqdm(self.results):
            total_count += 1
            final_result = None
            found_result = res_p.findall(result['result'])

            if found_result:
                try:
                    json_result = json.loads(found_result[0])
                    if key in json_result:
                        final_result = data_type(json_result[key])
                        success_count += 1
                    elif key.capitalize() in json_result:
                        final_result = data_type(json_result[key.capitalize()])
                        success_count += 1
                except:
                    pass
            
            if final_result is None:
                logging.error(f"Parser Error: {result['result']}")
            result['result'] = final_result
        
        logging.info(f'Total count: {total_count}, success count: {success_count}')

    def cal(self):
        if self.task_name == 'sts':
            self._cal_sts()
        elif self.task_name == 'truthful_qa':
            self._cal_tqa()
        elif self.task_name == 'squad':
            self._cal_squad()
        else:
            raise NotImplementedError

    def _cal_sts(self):
        labels = []
        preds = []
        for result in self.results:
            if result['result'] is not None:
                labels.append(result['score'])
                preds.append(result['result'])
        spearman, _ = spearmanr(labels, preds)
        pearson, _ = pearsonr(labels, preds)
        print(f'Spearman: {spearman:.4f}, Pearson: {pearson:.4f}')

    def _cal_tqa(self):
        correct_count = 0
        for result in self.results:
            if result['result'] is not None:
                if result['result'] == result['mc1_targets']['labels'].index(1):
                    correct_count += 1
        print(f'Precision: {correct_count / len(self.results):.4f}')

    def _cal_squad(self):
        def normalize_answer(s):
            """Lower text and remove punctuation, articles and extra whitespace."""
            def remove_articles(text):
                return re.sub(r'\b(a|an|the)\b', ' ', text)

            def white_space_fix(text):
                return ' '.join(text.split())

            def remove_punc(text):
                exclude = set(string.punctuation)
                return ''.join(ch for ch in text if ch not in exclude)

            def lower(text):
                return text.lower()

            return white_space_fix(remove_articles(remove_punc(lower(s))))
        def f1_score(prediction, ground_truth):
            prediction_tokens = normalize_answer(prediction).split()
            ground_truth_tokens = normalize_answer(ground_truth).split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                return 0
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            return f1
        def exact_match_score(prediction, ground_truth):
            return (normalize_answer(prediction) == normalize_answer(ground_truth))
        def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
            scores_for_ground_truths = []
            for ground_truth in ground_truths:
                score = metric_fn(prediction, ground_truth)
                scores_for_ground_truths.append(score)
            return max(scores_for_ground_truths)

        exact_matchs = []
        f1s = []
        for result in self.results:
            if result['result'] is None:
                exact_matchs.append(0.)
                f1s.append(0.)
            else:
                pred = result['result']
                gts = result['answers']['text']
                exact_matchs.append(metric_max_over_ground_truths(exact_match_score, pred, gts))
                f1s.append(metric_max_over_ground_truths(f1_score, pred, gts))
        
        a_em = sum(exact_matchs) / len(exact_matchs)
        a_f1 = sum(f1s) / len(f1s)
        print(f'Exact Match: {a_em:.4f}, F1: {a_f1: .4f}')

    def _normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    


if __name__ == '__main__':

    logging.basicConfig(level=20)
    if len(sys.argv) > 1 and sys.argv[1].lower() in ['sts', 'truthful_qa', 'squad']:
        metrics = Metrics(sys.argv[1].lower(), sys.argv[2])
        metrics.cal()
    else:
        executor = Executor(
            sys.argv[1], sys.argv[2]
        )
        if len(sys.argv) > 3:
            with_example = eval(sys.argv[3].lower().capitalize())
        else:
            with_example = False
        if len(sys.argv) > 4:
            with_cot = eval(sys.argv[4].lower().capitalize())
        else:
            with_cot = False
        
        asyncio.run(executor.run(
            with_example, with_cot
        ))
