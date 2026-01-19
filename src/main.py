"""
Entry script for your ML project.
"""

import openai
import pandas as pd
import argparse

client = openai.OpenAI()

def create_prompt(text: str) -> str:
    '''
    Docstring for create_prompt
    
    :param text: Description
    :type text: str
    :return: Prompt for text classification
    :rtype: str
    '''
    
    instructions = "Classify the sentiment of the following text as Positive, Negative, or Neutral.\n\n"
    formatting = "Positive, Negative, or Neutral\n\n"
    return f'Text: "{text}"\n\n{instructions}\n\nAnswer({formatting})'

def call_llm(prompt: str) -> str:
    '''
    Docstring for call_llm
    
    :param prompt: Description
    :type prompt: str
    :return: LLM response
    :rtype: str
    '''

    messages=[
            {"role": "user", "content": prompt}
        ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    print(response)
    return response.choices[0].message.content

def classify_text(text: str) -> str:
    '''
    Docstring for classify_text
    
    :param text: Description
    :type text: str
    :return: Classification result
    :rtype: str
    '''
    
    prompt = create_prompt(text)
    answer = call_llm(prompt)
    return answer



def __main__():
    print("🚀 Project is ready. Start building in src/ !")
    print("Tip: open ML_Workflow.txt for a step-by-step checklist.")
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='Path to the input text file')
    args = parser.parse_args()

    df = pd.read_csv(args.path)   
    df['Sentiment'] = df['Text'].apply(classify_text)
    df.to_csv('classified_output.csv', index=False)
    # prompt = create_prompt(text)
    # answer = call_llm(prompt)
    # print(f"Input Text: {text}")
    # print(f"LLM Response: {answer}")

if __name__ == "__main__":
    __main__()
