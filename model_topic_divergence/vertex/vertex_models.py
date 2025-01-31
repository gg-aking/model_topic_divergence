import re, json

from typing import Optional

import vertexai
from vertexai.language_models import TextGenerationModel
from vertexai.generative_models import GenerativeModel, Part, SafetySetting

class GeneralVertexModel:

    def __init__(self, 
                vertex_project : str,
                instruction : str,
                vertex_parameters : dict = {
                        "candidate_count": 1,
                        "max_output_tokens": 256,
                        "temperature": 0,
                        "top_p": 1,
                        "top_k": 1
                },
                vertex_model_name : str = "gemini-1.5-flash-002",
                vertex_location : str = 'us-central1',
    ):
        self.vertex_project = vertex_project
        self.instruction = instruction
        self.vertex_parameters = vertex_parameters

        self.vertex_model_name = vertex_model_name
        self.vertex_location = vertex_location
        
        self.safety_settings = [
            SafetySetting(
                    category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=SafetySetting.HarmBlockThreshold.OFF
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=SafetySetting.HarmBlockThreshold.OFF
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=SafetySetting.HarmBlockThreshold.OFF
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=SafetySetting.HarmBlockThreshold.OFF
            ),
        ]

        vertexai.init(project=self.vertex_project, location=self.vertex_location)
        self.vertex_model = GenerativeModel(self.vertex_model_name)

        
    
    def send_prompt(self, text : str, 
                            prepend_instruction : bool = True,
                            custom_instruction : Optional[str] = None):
        if prepend_instruction:
            if isinstance(custom_instruction, str):
                prompt = custom_instruction + '\n' + text
            else:
                prompt = self.instruction + '\n' + text
        else:
            prompt = text
        
        response = self.vertex_model.generate_content(
                [prompt],
                generation_config=self.vertex_parameters,
                safety_settings=self.safety_settings,
                stream=False,
        )
        return response

class VertexTopicNamer(GeneralVertexModel):

    def __init__(self, vertex_project : str, **kwargs):
        
        kwargs['vertex_project'] = vertex_project
        if 'instruction' not in kwargs:
            kwargs['instruction'] = """The following are top keywords from a topic in a topic modeling task. Please give a natural-sounding representative English name for this topic, given these keywords. Do not respond with anything beside the label name and the label should only include English words.
    Keywords:
    """
        if 'vertex_parameters' not in kwargs:
            kwargs['vertex_parameters'] =  {
                        "candidate_count": 1,
                        "max_output_tokens": 64,
                        "temperature": 0,
                        "top_p": 1,
                        "top_k": 1
                }
        super().__init__(**kwargs)

    def gen_topic_label_name(self, top_kws : list[str]) -> str:
        """
    Generate a descriptive label name for a topic based on its top keywords.

    This function takes a list of top keywords (`top_kws`) for a topic and generates 
    a human-readable label using a language model. The keywords are formatted as a 
    list and appended to a predefined instruction. The language model then predicts 
    an appropriate label based on the keywords.

    Args:
        top_kws (list[str]): A list of top keywords representing the topic.

    Returns:
        str: A descriptive label generated for the topic, refined by `clean_response_str`.
        """
        kws_as_str = "\n".join([f' - {kw}' for kw in top_kws])
        
        response = self.send_prompt(kws_as_str, prepend_instruction = True)

        response_str = response.text
        response_str = self.clean_response_str(response_str)
        return response_str

    def clean_response_str(self, response_str : str) -> str:
        """
            Clean the label name response from the LLM. 
            Remove non-alphanumeric characters from the edges, like dashes or astericks.
        """
        response_str = response_str.strip()
        response_str = re.sub(r'^\W+', '', response_str)
        response_str = re.sub(r'\W+$', '', response_str)
        return response_str

class VertexTopicDescriber(GeneralVertexModel):

    def __init__(self, vertex_project : str, **kwargs):

        kwargs['vertex_project'] = vertex_project
        if 'instruction' not in kwargs:
            kwargs['instruction'] = """You are a data annotation expert, designing a taxonomy of topics for webcontent. 
            You will be given the name of a topic from an NLP topic modeling task, though the name may be too specific to generally describe content.
Your task is to generate a topic name that is more general and broadly applicable to the content and to provide a description of content that falls in that topic.
Return the output in a JSON formatted like this: {"topic_name" : ...., "description" : ....}
Examples:
 - Input: "Yahoo Browser Compatibility"
 - Output: {
            "topic_name" : "Browser Issues",
            "description" : "Problems users encounter while using web browsers, including slow performance, compatibility errors, or security vulnerabilities, including content that categorizes text that discusses any difficulties related to browser functionality or user experience while browsing the internet.",
            }

 - Input: "Local Government Finance"
 - Output: {
            "topic_name" : "Local Issues",
            "description" : "Encompasses content related to the governance and administrative functions of local authorities, as well as challenges or concerns specific to a local community, including content that addresses municipal policies, public services, community problems, or local governance matters.",
            }
"""
        if 'vertex_parameters' not in kwargs:
            kwargs['vertex_parameters'] =  {
                        "candidate_count": 1,
                        "max_output_tokens": 256,
                        "temperature": 0,
                        "top_p": 1,
                        "top_k": 1
                }
        super().__init__(**kwargs)

    def rename_and_describe_topic(self, raw_topic_name : str) -> dict[str, str]:
        """
            Take a list of rough topic names and product a new list of more general topics,
            with short descriptions for each of them.
        Args:
            raw_labels (list[str]): A list of label names

        Returns:
            tuple[str, str]: A list of pairs of new topic names and label descriptions
        """

        prompt = self.instruction
        prompt += f""" - Input: "{raw_topic_name}"
- Output: """

        response = self.send_prompt(text = prompt, prepend_instruction = False)
        response_text = response.text
        j = self.parse_and_deduplicate_labels(response_text)
        return j
        
    def parse_and_deduplicate_label(self, response_text : str) -> dict[str, str]:
        response_text = response_text.replace('\n', ' ')
        response_text = re.sub(r'^.*\{', '{', response_text)
        response_text = re.sub(r'\}.*$', '}', response_text)
        j = json.loads(response_text)
        return j
