from groq import Groq
import os
from dotenv import load_dotenv

# 1. go to groq.com and generate an api key
# 2. create a .env file in the root directory
# 3. put the api key in .env file (GROQ_API_KEY = xxxx)

# load the GROQ API KEY from a .env file
load_dotenv("../.env")

# initialize a groq client (get an api key from their website)
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

