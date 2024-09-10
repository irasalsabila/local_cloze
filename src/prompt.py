from langchain_core.example_selectors.base import BaseExampleSelector
import pickle as pkl
import random
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate

sundanese_examples = pkl.load(open('icl_sundanese.pkl', 'rb'))
javanese_examples = pkl.load(open('icl_javanese.pkl', 'rb'))

class CustomExampleSelector(BaseExampleSelector):
    def __init__(self, examples):
        self.examples = examples

    def add_example(self, example):
        self.examples.append(example)

    def select_examples(self, n):
        return random.sample(self.examples, 5)

example_prompt = PromptTemplate.from_template("""Story Premise: {story_premise}
Correct Ending: {correct_ending}
Incorrect Ending: {wrong_ending}""")

topics = ["Food (e.g.: food souvenir, traditional foods and beverages, eating habit, traditional cutlery or cooking ware, local fruit)",
          "Wedding (e.g.: traditions before, during, and after marriage, bride & groom wedding clothes, invited guests, wedding location, food and gifts)",
          "Family relationships (e.g.: relationship with main and extended family, relation with society/neighbours, clan/descendant system)",
          "Pregnancy and Kids (e.g.: traditions during pregnancy, traditions after birth, how to care for a newborn baby, how to care for toddlers, how to care for children, teenagers, parents and childrens interactions as adults)",
          "Death (e.g.: tradition when death occurs, taking care of corpse, tradition after the body is buried, clothes of the mourners, inheritance matters)",
          "Religious holidays (e.g.: traditions before religious holidays, traditions leading up to religious holidays, traditions during religious holidays, traditions after religious holidays)",
          "Agriculture (e.g.:what to plant, traditions when planting, harvest)",
          "fisheries and trade (e.g.: traditions of taking care of livestock/fish, buying and selling traditions)",
          "Art (e.g.: musical instruments, folks songs, traditional dances, use of art at certain events, poetry or similar literature)",
          "Traditional games (e.g.: game types, location played)",
          "Daily activities (e.g.: morning activities, afternoon activities, evening activities, leisure activities, house, household and transportation)",
          "Socio-religious aspect of life (e.g.: regular religious activities, mystical things, traditional ceremonies, lifestyle, self care, traditional medicine, traditional saying)"]

# Function to create a FewShotPromptTemplate for any language
def create_overgeneration_prompt(language, examples):
    example_selector = CustomExampleSelector(examples)
    
    return FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=f"Your task is to write several triplets of story premises consisting of four sentences, wrong ending, and correct ending in {language}. Include {language} cultural values in the story with the topic \"{{topic}}\". Here are some examples of the story format:",
        suffix="Please generate several triplets, strictly following the format in the examples, do not add bullets or any additional response.",
        input_variables=["topic"]  # Add 'topic' so you can provide it dynamically
    )

# Create Sundanese and Javanese prompt templates using the function
sundanese_overgeneration_prompt = create_overgeneration_prompt("Sundanese", sundanese_examples)
javanese_overgeneration_prompt = create_overgeneration_prompt("Javanese", javanese_examples)

# Function to create a FewShotPromptTemplate for any language
def create_overgeneration_prompt_cohere(language, examples):
    example_selector = CustomExampleSelector(examples)
    
    return FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=f"Your task is to write three different examples of story premises consisting of four sentences, wrong ending, and correct ending in {language}. Include {language} cultural values in the story with the topic \"{{topic}}\". Here are some examples of the story format:",
        suffix="Please generate three different examples, strictly following the format in the examples, do not add bullets or any additional response.",
        input_variables=["topic"]  # Add 'topic' so you can provide it dynamically
    )

sundanese_overgeneration_prompt_cohere = create_overgeneration_prompt_cohere("Sundanese", sundanese_examples)
javanese_overgeneration_prompt_cohere = create_overgeneration_prompt_cohere("Javanese", javanese_examples)

def create_overgeneration_prompt_llama(language, examples):
    example_selector = CustomExampleSelector(examples)
    
    return FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>Your task is to write three different examples of story premises consisting of four sentences, wrong ending, and correct ending in {language}. Include {language} cultural values in the story with the topic \"{{topic}}\". Here are some examples of the story format:",
        suffix="Please generate several triplets, strictly following the format in the examples, do not add bullets or any additional response.<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
        input_variables=["topic"]  # Add 'topic' so you can provide it dynamically
    )

# Create Sundanese and Javanese prompt templates using the function
sundanese_overgeneration_prompt_llama = create_overgeneration_prompt_llama("Sundanese", sundanese_examples)
javanese_overgeneration_prompt_llama = create_overgeneration_prompt_llama("Javanese", javanese_examples)

def create_overgeneration_prompt_mixtral(language, examples):
    example_selector = CustomExampleSelector(examples)
    
    return FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=f"<s>[INST]Your task is to write three different examples of story premises consisting of four sentences, wrong ending, and correct ending in {language}. Include {language} cultural values in the story with the topic \"{{topic}}\". Here are some examples of the story format:",
        suffix="Please generate several triplets, strictly following the format in the examples, do not add bullets or any additional response.[/INST]",
        input_variables=["topic"]  # Add 'topic' so you can provide it dynamically
    )

# Create Sundanese and Javanese prompt templates using the function
sundanese_overgeneration_prompt_mixtral = create_overgeneration_prompt_mixtral("Sundanese", sundanese_examples)
javanese_overgeneration_prompt_mixtral = create_overgeneration_prompt_mixtral("Javanese", javanese_examples)

def create_overgeneration_prompt_gemma(language, examples):
    example_selector = CustomExampleSelector(examples)
    
    return FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=f"<start_of_turn>user\nYour task is to write three different examples of story premises consisting of four sentences, wrong ending, and correct ending in {language}. Include {language} cultural values in the story with the topic \"{{topic}}\". Here are some examples of the story format:",
        suffix="Please generate several triplets, strictly following the format in the examples, do not add bullets or any additional response.<end_of_turn>\n<start_of_turn>model\n",
        input_variables=["topic"]  # Add 'topic' so you can provide it dynamically
    )

# Create Sundanese and Javanese prompt templates using the function
sundanese_overgeneration_prompt_gemma = create_overgeneration_prompt_gemma("Sundanese", sundanese_examples)
javanese_overgeneration_prompt_gemma = create_overgeneration_prompt_gemma("Javanese", javanese_examples)