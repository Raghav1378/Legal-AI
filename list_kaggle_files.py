from kaggle.api.kaggle_api_extended import KaggleApi
import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), "src", "ai", ".env")
load_dotenv(dotenv_path=env_path)

api = KaggleApi()
api.authenticate()

dataset_slug = "fanaticauthorship/sc-judgments-india-1950-2024"
files = api.dataset_list_files(dataset_slug).files
print(f"Total files: {len(files)}")
for f in files[:20]:
    print(f.name)
