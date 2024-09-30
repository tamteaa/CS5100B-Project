from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv("../.env")

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

if __name__ == '__main__':
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Explain the importance of fast language models",
            }
        ],
        model="llama3-8b-8192",
    )

    print(chat_completion.choices[0].message.content)

