import redis.asyncio as redis
import json
from typing import Dict, Any
from app.config import settings


class RedisService:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=True
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def init_stream_group(self, stream_name: str, group_name: str):
        """초기 1회만 호출하여 Consumer Group을 생성"""
        try:
            await self.redis_client.xgroup_create(stream_name, group_name, id='0', mkstream=True)
        except redis.exceptions.ResponseError:
            pass  # 이미 존재

    async def consume_messages(self, stream_name: str, group_name: str, consumer_name: str, count: int = 1):
        """Redis Stream에서 메시지 소비"""
        try:
            messages = await self.redis_client.xreadgroup(
                group_name,
                consumer_name,
                {stream_name: '>'},
                count=count,
                block=1000  # ms 단위
            )
            return messages
        except Exception as e:
            print(f"메시지 소비 오류: {e}")
            return []

    async def acknowledge_message(self, stream_name: str, group_name: str, message_id: str):
        """메시지 ACK 처리"""
        try:
            await self.redis_client.xack(stream_name, group_name, message_id)
        except Exception as e:
            print(f"ACK 처리 오류: {e}")

    async def send_message(self, stream_name: str, data: Dict[str, Any]):
        """Redis Stream에 메시지 전송"""
        try:
            formatted_data = {
                key: json.dumps(value) if isinstance(value, (dict, list)) else str(value)
                for key, value in data.items()
            }
            message_id = await self.redis_client.xadd(stream_name, formatted_data)
            return message_id
        except Exception as e:
            print(f"메시지 전송 오류: {e}")
            return None

    @staticmethod
    def parse_message(message_data: Dict[str, str]) -> Dict[str, Any]:
        """Redis 메시지 파싱"""
        parsed = {}
        for key, value in message_data.items():
            try:
                parsed[key] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                parsed[key] = value
        return parsed

    async def close(self):
        """Redis 연결 종료"""
        await self.redis_client.close()


# 예시 사용:
# async with RedisService() as redis_service:
#     await redis_service.send_message("ai-request-stream", {"data": {...}})