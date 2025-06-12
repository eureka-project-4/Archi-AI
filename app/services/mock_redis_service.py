# app/services/mock_redis_service.py
import json
from typing import Dict, Any
import asyncio
from collections import defaultdict

class MockRedisService:
    def __init__(self):
        self.streams = defaultdict(list)
        self.groups = defaultdict(dict)
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
        
    async def init_stream_group(self, stream_name: str, group_name: str):
        self.groups[stream_name][group_name] = []
        print(f"[MOCK] Consumer group '{group_name}' created for stream '{stream_name}'")
        
    async def send_message(self, stream_name: str, data: Dict[str, Any]):
        message_id = f"mock-{len(self.streams[stream_name])}"
        self.streams[stream_name].append({
            "id": message_id,
            "data": data
        })
        print(f"[MOCK] Message sent to '{stream_name}': {message_id}")
        return message_id
        
    async def consume_messages(self, stream_name: str, group_name: str, consumer_name: str, count: int = 1):
        # 간단한 시뮬레이션
        if self.streams[stream_name]:
            message = self.streams[stream_name].pop(0)
            return [(stream_name, [(message["id"], message["data"])])]
        return []
        
    async def acknowledge_message(self, stream_name: str, group_name: str, message_id: str):
        print(f"[MOCK] Message {message_id} acknowledged")
        
    @staticmethod
    def parse_message(message_data: Dict[str, str]) -> Dict[str, Any]:
        parsed = {}
        for key, value in message_data.items():
            try:
                parsed[key] = json.loads(value)
            except:
                parsed[key] = value
        return parsed
        
    async def close(self):
        pass