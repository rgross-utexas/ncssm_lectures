from datetime import datetime
import logging
from pathlib import Path

def get_logger(name: str)-> logging.Logger:
    
    Path("log").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=f'log/{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
                        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        level=logging.INFO)
    return logging.getLogger(__name__)
