from dotenv import load_dotenv
import os
load_dotenv()
print(str(os.getenv("GROQ_API_KEY")))