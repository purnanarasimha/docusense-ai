"""api configuration central config using pydantic settings"""

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from pathlib import Path

class Settings(BaseSettings):
    """application setting loading from enb=v"""

    #--- App ---------
    app_name : str = "DocuSense AI"
    app_version : str = "1.0.0"
    app_env : str = "development"
    log_level : str = "INFO"

    # --- api keys -----
    google_api_key : str = ""
    hf_token : str = ""

    # --qdrant ----
    qdrant_url : str = ""
    qdrant_api_key : str = ""

    # --- langsmith ----
    langchain_api_key : str = ""
    langchain_tracing_v2 : str = "true"
    langchain_project : str = "docusense-ai"

    # ---- processing ------------
    chunk_size : int = 512
    chunk_overlap : int = 50
    max_chunks_per_doc : int = 500

    # --- LLM ---------
    llm_provider : str = "google"
    llm_model : str = "gemini-3.5-flash-lite"

    # -- paths ------
    base_dir : Path = Path(__file__).parent.parent
    data_dir : Path = Path(__file__).parent.parent / "data"
    raw_dir : Path = Path(__file__).parent.parent / "data" / "raw"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

@lru_cache
def get_settings() -> Settings:
    """Cached settings instance"""

    return Settings()