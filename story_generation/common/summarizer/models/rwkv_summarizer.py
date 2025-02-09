import time
import logging

from rwkvstic.load import RWKV #NOTE FOR REQUIREMENTS DOC use pip3 install rwkvstic==1.2.4 rwkv==0.5.0

import torch

#from rwkvutils import PIPELINE #not sure on this one, may not be necessary
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("20B_tokenizer.json") #download from BlinkDL rwkv repos

from story_generation.common.summarizer.models.abstract_summarizer import AbstractSummarizer
from story_generation.common.data.split_paragraphs import cut_last_sentence
from story_generation.common.util import *

class RWKVSummarizer(AbstractSummarizer):
    def __init__(self, args):
        assert args.rwkv_model is not None
        self.tokenizer = Tokenizer.from_file("20B_tokenizer.json") #download from BlinkDL rwkv repos
        self.model = args.rwkv_model
        self.tokenizer.add_bos_token = False #rwkv package has tokenizer built in
        self.args = args
        self.controller = None

    @torch.no_grad()
    def __call__(self, texts, suffixes=None, max_tokens=None, top_p=None, temperature=None, retry_until_success=True, stop=None, logit_bias=None, num_completions=1, cut_sentence=False, model_string=None):
        assert type(texts) == list
        if logit_bias is None:
            logit_bias = {}
        if suffixes is not None:
            assert len(texts) == len(suffixes)
        if model_string is None:
            logging.warning('model string not provided, using default model')
        if self.controller is None:
            return self._call_helper(texts, suffixes=suffixes, max_tokens=max_tokens, top_p=top_p, temperature=temperature, retry_until_success=retry_until_success, stop=stop, logit_bias=logit_bias, num_completions=num_completions, cut_sentence=cut_sentence, model_string=model_string)
        else:
            raise NotImplementedError

    @torch.no_grad()
    def _call_helper(self, texts, suffixes=None, max_tokens=None, top_p=None, temperature=None, retry_until_success=True, stop=None, logit_bias=None, num_completions=1, cut_sentence=False, model_string=None):
        assert model_string in PRETRAINED_MODELS

        if logit_bias is None:
            logit_bias = {}

        outputs = []
        rwkv_model = RWKV("/media/ubuntu/Download/RWKV-4-Pile-14B-20230313-ctx8192-test1050.pth",strategy="cuda fp16i8") #change model here
        for i in range(len(texts)):
            text = texts[i]
            prompt = text

            retry = True
            num_fails = 0
            while retry:
                try:
                    context_length = len(self.tokenizer.encode(prompt))
                    if context_length > self.args.max_context_length:
                        logging.warning('context length' + ' ' + str(context_length) + ' ' + 'exceeded artificial context length limit' + ' ' + str(self.args.max_context_length))
                        time.sleep(5) # similar interface to GPT rwkv query failing and retrying
                        assert False
                    if max_tokens is None:
                        max_tokens = min(self.args.max_tokens, self.args.max_context_length - context_length)
                    engine = self.model if model_string is None else model_string
                    if engine == 'text-davinci-001':
                        engine = 'text-davinci-002' # update to latest version
                    logging.log(21, 'PROMPT')
                    logging.log(21, prompt)
                    logging.log(21, 'MODEL STRING:' + ' ' + self.model if model_string is None else model_string)
                    TEMPERATURE=1.0 #completion = model.forward(prompt = prompt #openai.Completion.create(
                    top_p = 0.8
                    prompt=prompt
                    suffix=suffixes[i] if suffixes is not None else None,
                    rwkv_model.loadContext(ctx=input + instruction,newctx=context)
                    completion = rwkv_model.forward(prompt + suffix)["output"]
                    retry = False
                except Exception as e:
                    logging.warning(str(e))
                    retry = retry_until_success
                    num_fails += 1
                    if num_fails > 20:
                        raise e
                    if retry:
                        logging.warning('retrying...')
                        time.sleep(num_fails)
            outputs += [completion['choices'][j]['text'] for j in range(num_completions)]
        if cut_sentence:
            for i in range(len(outputs)):
                if len(outputs[i].strip()) > 0:
                    outputs[i] = cut_last_sentence(outputs[i])
        engine = self.rwkv_model if model_string is None else model_string
        logging.log(21, 'OUTPUTS')
        logging.log(21, str(outputs))
        logging.log(21, 'RWKV CALL' + ' ' + engine + ' ' + str(len(self.tokenizer.encode(texts[0])) + sum([len(self.tokenizer.encode(o)) for o in outputs])))
        return outputs
