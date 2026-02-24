import os
import json
import fitz # PyMuPDF
from tqdm import tqdm
from kaggle.api.kaggle_api_extended import KaggleApi
from .vector_service import VectorService

class KaggleLegalIngestor:
    def __init__(self):
        self.api = KaggleApi()
        self.api.authenticate()
        self.vector_service = VectorService()
        self.dataset_slug = "fanaticauthorship/sc-judgments-india-1950-2024"

    def ingest_year(self, year: int, max_files: int = 10):
        """Downloads and processes judgments for a specific year."""
        download_path = f"./temp_judgments/{year}"
        os.makedirs(download_path, exist_ok=True)
        
        print(f"[KAGGLE] Fetching judgments for year {year}...")
        
        # In this dataset, files are organized in folders by year
        # We search for files specifically in that year's folder
        files = self.api.dataset_list_files(self.dataset_slug).files
        year_files = [f for f in files if f.name.startswith(f"supreme_court_judgments/{year}/")]
        
        # Limit processing for testing
        to_process = year_files[:max_files]
        
        for file in tqdm(to_process, desc=f"Processing {year}"):
            file_path = os.path.join(download_path, os.path.basename(file.name))
            
            # Download PDF
            self.api.dataset_download_file(self.dataset_slug, file.name, path=download_path)
            
            # Extract Text
            text = self._extract_text(file_path + ".zip") # Kaggle downloads often wrap in zip
            if not text: 
                continue

            # Case Name Extraction from filename
            case_name = os.path.basename(file.name).replace("_", " ").replace(".PDF", "")
            
            # Upload to Pinecone (in chunks of 1000 characters for better search)
            self._chunk_and_upload(text, case_name, year)

    def _extract_text(self, pdf_path: str) -> str:
        try:
            # Note: Logic to unzip may be needed depending on Kaggle's specific zip structure
            # For brevity, assuming direct PDF access or simple unzip
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            print(f"[ERR] Failed to extract {pdf_path}: {e}")
            return ""

    def _chunk_and_upload(self, text: str, case_name: str, year: int):
        # Professional RAG chunking: 1000 chars with 200 char overlap
        chunk_size = 1000
        overlap = 200
        
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            chunk_id = f"{case_name}_{year}_{i}"
            
            metadata = {
                "case_name": case_name,
                "case_year": year,
                "source": "Supreme Court of India",
                "ingested_on": str(os.environ.get("CURRENT_TIME", "2026-02-24"))
            }
            
            self.vector_service.upsert_judgment_chunk(chunk_id, chunk, metadata)

if __name__ == "__main__":
    # Example usage: ingest 5 judgments from 2023 to test the pipeline
    ingestor = KaggleLegalIngestor()
    ingestor.ingest_year(2023, max_files=5)
