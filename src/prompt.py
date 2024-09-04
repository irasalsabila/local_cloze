from langchain_core.example_selectors.base import BaseExampleSelector
import pickle as pkl

sundanese_examples = pkl.load(open('icl_sundanese.pkl', 'rb'))

import random
class CustomExampleSelector(BaseExampleSelector):
    def __init__(self, examples):
        self.examples = examples

    def add_example(self, example):
        self.examples.append(example)

    def select_examples(self, n):
        return random.sample(self.examples, 5)
    
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate

example_prompt = PromptTemplate.from_template("""Story Premise: {story_premise}
Correct Ending: {correct_ending}
Incorrect Ending: {wrong_ending}""")

example_selector = CustomExampleSelector(sundanese_examples)
sundanese_overgeneration_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Your task is to write severals triplets of story premises consists of four sentences, wrong ending, and correct ending in Sundanese. Here are some examples:",
    suffix="Please return {n} generated triplets, following the format in the examples, without any additional response.",
    input_variables=["n"]
)    
