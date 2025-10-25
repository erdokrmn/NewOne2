import json
import os
from datetime import datetime
from typing import Dict, List

class DataSaver:
    def __init__(self, save_path: str = "data/training_data.jsonl"):
        self.save_path = save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self._file = open(self.save_path, "a", encoding="utf-8")
    
    def save_single(self, record: Dict):
        record["timestamp"] = datetime.now().isoformat(timespec="milliseconds")
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()
    
    def save_all(self, records: List[Dict]):
        for record in records:
            record["timestamp"] = datetime.now().isoformat(timespec="milliseconds")
            self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self):
        self._file.close()
