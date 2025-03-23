from insert import create_index
import os
from dotenv import load_dotenv

load_dotenv()

INDEX_NAME = os.getenv("INDEX_NAME")

if __name__ == "__main__":
    print(f"Creating index: {INDEX_NAME}")
    create_index(INDEX_NAME)
    print("Index creation completed") 