# src/config.py
"""
Configuración centralizada usando Pydantic Settings
Proporciona validación automática de variables de entorno
"""
from pydantic_settings import BaseSettings
from pydantic import Field
import pathlib

class Settings(BaseSettings):
    """Configuración de la aplicación validada con Pydantic"""
    
    # Supabase
    supabase_url: str = Field(..., env="SUPABASE_URL", description="URL de Supabase")
    supabase_key: str = Field(..., env="SUPABASE_KEY", description="API Key de Supabase")
    
    # Rutas del proyecto (calculadas automáticamente)
    base_dir: pathlib.Path = Field(default_factory=lambda: pathlib.Path(__file__).resolve().parent.parent)
    
    @property
    def model_path(self) -> pathlib.Path:
        return self.base_dir / 'models' / 'production_model.json'
    
    @property
    def data_path(self) -> pathlib.Path:
        return self.base_dir / 'data' / 'processed' / 'train_features.parquet'
    
    @property
    def metrics_path(self) -> pathlib.Path:
        return self.base_dir / 'reports' / 'metrics.json'
    
    @property
    def processed_dir(self) -> pathlib.Path:
        return self.base_dir / 'data' / 'processed'
    
    # API Configuration
    api_url: str = Field(default="http://127.0.0.1:8000", env="API_URL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# Instancia global de configuración
settings = Settings()
