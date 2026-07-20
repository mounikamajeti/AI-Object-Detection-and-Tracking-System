import sqlite3
from datetime import datetime
from pathlib import Path
import threading

class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.connection = None
        self.db_lock = threading.Lock()
        self.create_database()

    def create_database(self) -> None:
        try:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = self.connection.cursor()
            cursor.execute(
                '''
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    object_name TEXT NOT NULL,
                    tracking_id INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    frame_number INTEGER NOT NULL
                )
                '''
            )
            self.connection.commit()
        except sqlite3.Error as exc:
            raise RuntimeError(f'Database creation failed: {exc}')

    def save_detection(self, object_name: str, tracking_id: int, confidence: float, frame_number: int) -> None:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        try:
            with self.db_lock:
                cursor = self.connection.cursor()
                cursor.execute(
                    '''
                    INSERT INTO detections (object_name, tracking_id, confidence, timestamp, frame_number)
                    VALUES (?, ?, ?, ?, ?)
                    ''',
                    (object_name, tracking_id, confidence, timestamp, frame_number)
                )
                self.connection.commit()
        except sqlite3.Error as exc:
            raise RuntimeError(f'Failed to save detection: {exc}')

    def get_history(self, object_name: str = '', tracking_id: str = '', date: str = '') -> list[tuple]:
        try:
            with self.db_lock:
                query = 'SELECT id, object_name, tracking_id, confidence, timestamp, frame_number FROM detections WHERE 1=1'
                params: list = []
                if object_name:
                    query += ' AND object_name LIKE ?'
                    params.append(f'%{object_name}%')
                if tracking_id:
                    query += ' AND tracking_id = ?'
                    params.append(int(tracking_id))
                if date:
                    query += ' AND timestamp LIKE ?'
                    params.append(f'%{date}%')
                cursor = self.connection.cursor()
                cursor.execute(query, params)
                return cursor.fetchall()
        except sqlite3.Error as exc:
            raise RuntimeError(f'Failed to query history: {exc}')

    def close(self) -> None:
        if self.connection:
            self.connection.close()
