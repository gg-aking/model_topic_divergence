import re

import vertexai
from vertexai.language_models import TextGenerationModel
from vertexai.generative_models import GenerativeModel, Part, SafetySetting

class VertexTopicNamer:

    def __init__(self, 
                 vertex_project : str,
                 vertex_model_name : str = "gemini-1.5-flash-002",
                 vertex_location : str = 'us-central1',
                 vertex_parameters : dict = {
                        "candidate_count": 1,
                        "max_output_tokens": 64,
                        "temperature": 0,
                        "top_p": 1,
                        "top_k": 1
                },
    ):
        self.vertex_project = vertex_project
        self.vertex_model_name = vertex_model_name
        self.vertex_location = vertex_location
        self.vertex_parameters = vertex_parameters

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

        self.instruction = """The following are top keywords from a topic in a topic modeling task. Please give a natural-sounding representative English name for this topic, given these keywords. Do not respond with anything beside the label name and the label should only include English words.
    Keywords:
    """
    
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
        
        prompt = self.instruction + '\n' + kws_as_str
        
        response = self.vertex_model.generate_content(
                [prompt],
                generation_config=self.vertex_parameters,
                safety_settings=self.safety_settings,
                stream=False,
        )

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
