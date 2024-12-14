import json
import google.generativeai as genai
from PIL import Image
import re


class GoogleGeminiExtractor:
    def __init__(self, google_api_key: str, model='gemini-1.5-flash'):
        """
        Initialize GoogleGeminiExtractor with Google API key.

        Parameters:
            google_api_key (str): API key for Google Generative AI.
        """
        genai.configure(api_key=google_api_key)
        # self.model = genai.GenerativeModel('gemini-pro-vision')
        self.model = genai.GenerativeModel(model)

    def get_prompt(self, fields: dict) -> str:
        """
        Create a prompt for extracting information from an image based on the given fields.

        Parameters:
            fields (dict): The desired fields to extract.

        Returns:
            str: Prompt formatted for the AI model.
        """
        prompt = (
                "Extract the following information from the image: "
                + json.dumps(fields)
                + "\n\nOutput format:\n"
                + json.dumps(fields, indent=2)
        )
        return prompt

    def get_gemini_response(self, input_text: str, image: Image, prompt: str) -> str:
        """
        Get responses from Gemini models based on text and image input.

        Parameters:
            input_text (str): Input text for the model.
            image (Image): Input image for the model.
            prompt (str): Prompt for information extraction.

        Returns:
            str: Response from the model in text form.
        """
        try:
            response = self.model.generate_content([input_text, image, prompt])
            return response.text
        except Exception as e:
            return f"Error generating content from Google Gemini: {str(e)}"

    def extract_information(self, image: str | Image.Image, fields: dict) -> dict:
        """
        Extracting information from the image based on the given fields.

        Parameters:
            image (str | Image): Path or PIL Image of the image.
            fields (dict): The desired fields to extract.

        Returns:
            dict: Extracted information in dictionary form.
        """
        input_prompt = """
                       You are an expert in understanding image documents.
                       You will receive input images of documents such as invoices,
                       receipts, identity cards, forms, etc.
                       You have to extract information based on the input image.
                       """

        if isinstance(image, str):
            img = Image.open(image)
        else:
            img = image

        prompt = self.get_prompt(fields)
        response_text = self.get_gemini_response(input_prompt, img, prompt)

        # Convert the response_text into a dictionary
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            extracted_info_str = response_text[json_start:json_end]
            try:
                extracted_info_dict = json.loads(extracted_info_str)
            except json.JSONDecodeError:
                extracted_info_dict = {"error": "Failed to parse JSON", "extracted_info": response_text}
        else:
            extracted_info_dict = {"error": "No JSON found in the response", "extracted_info": response_text}

        return extracted_info_dict
