import openai
import time
import random
import numpy as np
import logging
import codecs
import os
import json

logger = logging.getLogger(__name__)


class OpenAIClient():
    def __init__(self, keys_file):
        if os.path.exists(keys_file):
            with open(keys_file) as f:
                self.keys = [i.strip() for i in f.readlines()]
        else:
            self.keys = [os.environ['OPENAI_TOKEN']]
        self.n_processes = len(self.keys)

    def call_api(self, prompt: str, engine: str, max_tokens=200, temperature=1,
                 stop=None, n=None, echo=False):
        result = None
        if temperature == 0:
            n = 1

        stop = stop.copy()
        for i, s in enumerate(stop):
            if '\\' in s:
                # hydra reads \n to \\n, here we decode it back to \n
                stop[i] = codecs.decode(s, 'unicode_escape')
        count = 0
        while result is None and count < 10:
            try:
                key = random.choice(self.keys)
                if engine == "gpt-3.5-turbo" or engine == "gpt-4":
                    result = openai.ChatCompletion.create(
                        model=engine,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        api_key=key,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        n=n,
                        stop=stop,
                        logprobs=True,
                        echo=echo
                    )
                else:
                    result = openai.Completion.create(
                        engine=engine,
                        prompt=prompt,
                        api_key=key,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        n=n,
                        stop=stop,
                        logprobs=1,
                        echo=echo
                    )
                time.sleep(5)
                return result
            except Exception as e:
                logger.info(f"{str(e)}, 'Retry.")
                count = count + 1
                time.sleep(5)

    def extract_response(self, response, engine: str):
        if response is None:
            return [{'text': 'error'}]
        if engine == "gpt-3.5-turbo" or engine == "gpt-4":
            texts = []
            if type(response) == str:
                response = json.loads(response)
            for r in response['choices']:
                if 'content' in r['message']:
                    texts.append(r['message']['content'])
                else:
                    texts.append('error')
            return [{'text': texts}]
        else:
            texts = [r['text'] for r in response['choices']]
            logprobs = [np.mean(r['logprobs']['token_logprobs']) for r in response['choices']]
        return [{"text": text, "logprob": logprob} for text, logprob in zip(texts, logprobs)]

    def extract_loss(self, response):
        lens = len(response['choices'][0]['logprobs']['tokens'])
        ce_loss = -sum(response['choices'][0]['logprobs']['token_logprobs'][1:])
        return ce_loss / (lens-1)  


def run_api(args, **kwargs):
    if isinstance(args, tuple):
        prompt, choices = args
    else:
        prompt, choices = args, None
    client = kwargs.pop('client')
    if choices is None:
        response = client.call_api(prompt=prompt, **kwargs)
        response = client.extract_response(response=response, engine=kwargs.pop('engine'))
    else:
        kwargs.update({"echo": True, "max_tokens": 0})
        losses = np.array([client.extract_loss(client.call_api(prompt=prompt+choice, **kwargs))
                          for choice in choices])
        pred = int(losses.argmin())  
        response = [{'text': pred}]
    return response