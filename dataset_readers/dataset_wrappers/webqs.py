from utils.misc import App
from dataset_readers.dataset_wrappers.base_dsw import *
import logging
logger = logging.getLogger(__name__)
field_getter = App()


@field_getter.add("q")
def get_q(entry):
    return entry['question']


@field_getter.add("qa")
def get_qa(entry):
    return f"{get_q(entry)} {get_a(entry)}"


@field_getter.add("a")
def get_a(entry):
    return entry['answers'][0]


@field_getter.add("gen_a")
def get_gen_a(entry):
    return "{ice_prompt}{question} ".format(
        question=get_q(entry),
        ice_prompt="{ice_prompt}")


class DatasetWrapper(ABC):
    name = "webqs"
    ice_separator = "\n"
    question_field = "question"
    answer_field = "answers"
    hf_dataset = "webqs/webqs.py"
    field_getter = field_getter