#!/usr/bin/python3
# -*- coding: utf-8 -*-
from transformers import AutoTokenizer
import logging
logger = logging.getLogger(__name__)


def model_to_tokenizer(model_name):
    if "code-" in model_name:
        #logger.info(f"code")
        return "SaulLu/codex-like-tokenizer"
    if "gpt3" in model_name:
        #logger.info(f"gpt3")
        return "gpt2"
    if "gpt-" in model_name:
        #logger.info(f"gpt-")
        return "gpt-neo-2-7B-hf"
    logger.info(model_name)
    return model_name


def get_tokenizer(model_name):
    if model_name == 'bm25':
        return model_name
    return AutoTokenizer.from_pretrained(model_to_tokenizer(model_name))