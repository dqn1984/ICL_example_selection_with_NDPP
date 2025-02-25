from utils.misc import App
from dataset_readers.dataset_wrappers.base_dsw import ABC

field_getter = App()


@field_getter.add("q")
def get_q(entry):
    # in-context example for few-shot generating question
    return entry['question']


@field_getter.add("a")
def get_a(entry):
    return entry['target']


@field_getter.add("qa")
def get_qa(entry):
    return f"{get_q(entry)}\t{get_a(entry)}"


@field_getter.add("gen_a")
def get_gen_a_instruction(entry):
    #根据给出的四句话，生成合理的结局。结局只包含一句话，格式与样本保持一致。
    instruction = 'Generate plausible endings based on the four sentences given. The ending contains only one sentence and the format is consistent with the example.'
    prompt = "{instruction}\n{ice_prompt}{question}\t"
    prompt = prompt.format(
        question=get_q(entry),
        ice_prompt="{ice_prompt}",
        instruction=instruction)
    #return prompt
    return "{ice_prompt}{question}\t".format(question=get_q(entry), ice_prompt='{ice_prompt}')


class DatasetWrapper(ABC):
    name = "rocending"
    ice_separator = "\n"
    question_field = "question"
    answer_field = "target"
    hf_dataset = "KaiLv/UDR_RocEnding"
    hf_dataset_name = None
    field_getter = field_getter
