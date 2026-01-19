import argparse

import openai

client = openai.OpenAI()

def analyze_image_question(image_url_1: str, image_url_2: str, question: str) -> str:
    '''
    Docstring for analyze_image_question
    
    :param image_url_1: URL of the first image
    :type image_url_1: str
    :param image_url_2: URL of the second image
    :type image_url_2: str
    :param question: Question about the images
    :type question: str
    :return: LLM response to the image question
    :rtype: str
    '''
    
    
    messages=[
            {"role": "user", "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": image_url_1}},
                {"type": "image_url", "image_url": {"url": image_url_2}}
            ]}
        ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    print(response)
    return response.choices[0].message.content

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('image_url_1', type=str, help='URL of the first image to be analyzed')
    parser.add_argument('image_url_2', type=str, help='URL of the second image to be analyzed')
    parser.add_argument('question', type=str, help='Question about the images')
    args = parser.parse_args()

    answer = analyze_image_question(args.image_url_1, args.image_url_2, args.question)
    print(f"Answer: {answer}")