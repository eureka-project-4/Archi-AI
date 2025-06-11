
import asyncio
import json
from typing import Dict, Any
from datetime import datetime, timezone
from app.services.redis_service import RedisService
from app.services.processor import message_processor
from app.services.ai_classifier import ai_classifier
from app.services.content_filter import content_filter
from app.models.message import AiPromptMessage, MessageType
from app.models.chat import ChatMessage
from app.database import get_db  # 수정된 import
from app.config import settings

class StreamConsumer:
    def __init__(self):
        self.running = False
        
    async def start_consuming(self):
        """메시지 소비 시작"""
        self.running = True
        
        async with RedisService() as redis_service:
            # Consumer Group 초기화
            await redis_service.init_stream_group(
                settings.REQUEST_STREAM, 
                settings.CONSUMER_GROUP
            )
            
            print(f"[INFO] 컨슈머 시작: {settings.REQUEST_STREAM}")
            
            while self.running:
                try:
                    # 메시지 읽기
                    messages = await redis_service.consume_messages(
                        stream_name=settings.REQUEST_STREAM,
                        group_name=settings.CONSUMER_GROUP,
                        consumer_name=settings.CONSUMER_NAME,
                        count=1
                    )
                    
                    # 메시지 처리
                    for stream_name, stream_messages in messages:
                        for message_id, message_data in stream_messages:
                            await self._process_single_message(
                                redis_service, message_id, message_data
                            )
                            
                except Exception as e:
                    print(f"[ERROR] 컨슈머 오류: {e}")
                    await asyncio.sleep(5)  # 오류 시 5초 대기
    
    async def _process_single_message(self, redis_service: RedisService, message_id: str, message_data: Dict[str, str]):
        """개별 메시지 처리"""
        try:
            # 메시지 파싱
            parsed_data = redis_service.parse_message(message_data)
            print(f"[INFO] 메시지 수신 - ID: {message_id}")
            print(f"[DEBUG] 메시지 데이터: {parsed_data}")
            
            # Pydantic 모델로 변환
            ai_message = AiPromptMessage(**parsed_data)
            
            # 1. 유저 메시지 DB 저장
            chat_id = await self._save_user_message(ai_message)
            print(f"[DB] 유저 메시지 저장 완료 - Chat ID: {chat_id}")
            
            # 0단계: AI 기반 금칙어 필터링
            if await content_filter.contains_forbidden_content(ai_message.payload.content):
                print(f"[WARN] 금칙어 감지됨: {ai_message.payload.content[:20]}...")
                response = await self._create_filtered_response(ai_message)
            else:
                # 1단계: AI로 메시지 타입 분류
                classified_type = await ai_classifier.classify_message_type(
                    ai_message.payload.content, 
                    ai_message.metadata
                )
                ai_message.payload.type = classified_type
                print(f"[INFO] 분류 결과: {classified_type}")
                
                # 2단계: 분류된 타입으로 메시지 처리 (AI 응답 생성)
                response = await message_processor.process_message(ai_message)
            
            # 3. 봇 응답 DB 저장 (content만)
            bot_chat_id = await self._save_bot_response(response, chat_id)
            print(f"[DB] 봇 응답 저장 완료 - Chat ID: {bot_chat_id}")
            
            # 4. 응답에 실제 DB ID들 반영
            response["messageId"] = str(bot_chat_id)
            response["chatId"] = str(chat_id)  # 유저 메시지의 chat_id
            
            print(f"[INFO] 메시지 처리 완료 - 타입: {response['type']}")
            
            # 응답 전송
            await self._send_response(redis_service, response)
            
            # ACK 처리
            await redis_service.acknowledge_message(
                settings.REQUEST_STREAM,
                settings.CONSUMER_GROUP,
                message_id
            )
            
        except Exception as e:
            print(f"[ERROR] 메시지 처리 실패 - ID: {message_id}, 에러: {e}")
            # 실패한 메시지는 ACK 하지 않음 (재처리 가능)
    
    async def _save_user_message(self, ai_message: AiPromptMessage) -> int:
        """유저 메시지 DB 저장 - chat_id 반환"""
        async for db in get_db():
            try:
                message = ChatMessage(
                    user_id=int(ai_message.payload.user_id),
                    message=ai_message.payload.content,
                    sender="USER",
                    created_at=datetime.now(),
                    message_type=ai_message.payload.type
                )
                db.add(message)
                await db.commit()
                await db.refresh(message)
                return message.chat_id  # Primary Key 반환
            except Exception as e:
                await db.rollback()
                print(f"[ERROR] 유저 메시지 저장 실패: {e}")
                raise e

    async def _save_bot_response(self, response: Dict[str, Any], chat_id: int) -> int:
        """봇 응답 DB 저장 - 기본 content만 저장"""
        async for db in get_db():
            try:
                message = ChatMessage(
                    user_id=int(response["userId"]),
                    message=response["content"],  # 기본 메시지만 저장
                    sender="BOT",
                    created_at=datetime.now(),
                    message_type=response["type"]
                )
                db.add(message)
                await db.commit()
                await db.refresh(message)
                return message.chat_id
            except Exception as e:
                await db.rollback()
                print(f"[ERROR] 봇 응답 저장 실패: {e}")
                raise e
    
    async def _create_filtered_response(self, message: AiPromptMessage) -> Dict[str, Any]:
        """금칙어 감지 시 응답 생성"""
        return {
            "messageId": "temp",  # DB 저장 후 덮어씀
            "userId": message.payload.user_id,
            "content": "부적절한 표현이 감지되었습니다. 정중한 언어를 사용해 주세요.",
            "type": MessageType.FILTERED_MESSAGE,
            "sender": "BOT",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "bundleRecommendations": "",
            "extractedKeywords": "",
            "updatedPreference": ""
        }
    
    async def _send_response(self, redis_service: RedisService, response: Dict[str, Any]):
        """응답 전송"""
        await redis_service.send_message(settings.RESPONSE_STREAM, response)
        print(f"[INFO] 응답 전송 완료 - 타입: {response['type']}")
    
    def stop(self):
        """컨슈머 중지"""
        self.running = False
        print("[INFO] 컨슈머 중지 요청")

stream_consumer = StreamConsumer()