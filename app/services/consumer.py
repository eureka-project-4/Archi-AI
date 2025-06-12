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
from app.database import get_db
from app.config import settings
from app.database import SessionLocal


class StreamConsumer:
    def __init__(self):
        self.running = False
        
    async def start_consuming(self):
        """메시지 소비 시작"""
        self.running = True
        print("[DEBUG] Consumer 시작됨")
        
        async with RedisService() as redis_service:
            print("[DEBUG] Redis 연결 완료")
            # Consumer Group 초기화
            await redis_service.init_stream_group(
                settings.REQUEST_STREAM, 
                settings.CONSUMER_GROUP
            )
            print("[DEBUG] Stream Group 초기화 완료")
            print(f"[INFO] 컨슈머 시작: {settings.REQUEST_STREAM}")
            
            while self.running:
                try:
                    # 메시지 읽기
                    print(f"[DEBUG] 메시지 읽기 시도 - Stream: {settings.REQUEST_STREAM}")
                    messages = await redis_service.consume_messages(
                        stream_name=settings.REQUEST_STREAM,
                        group_name=settings.CONSUMER_GROUP,
                        consumer_name=settings.CONSUMER_NAME,
                        count=1
                    )
                    print(f"[DEBUG] 읽은 메시지 수: {len(messages)}")
                    
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
            print(f"\n[DEBUG] 원본 message_data: {message_data}")

            # 메시지 파싱
            parsed_data = redis_service.parse_message(message_data)
            print(f"[DEBUG] parsed_data 타입: {type(parsed_data)}")
            print(f"[DEBUG] parsed_data 내용: {parsed_data}")
            print(f"[INFO] 메시지 수신 - ID: {message_id}")

            # Pydantic 모델 변환
            ai_message = AiPromptMessage(**parsed_data)
            print(f"[DEBUG] Pydantic 변환 성공 - userId: {ai_message.payload.user_id}, content: {ai_message.payload.content}")
            print(f"[DEBUG] 초기 type 값: {ai_message.payload.type}")

            # 0단계: 금칙어 필터링
            if await content_filter.contains_forbidden_content(ai_message.payload.content):
                print(f"[WARN] 금칙어 감지됨: {ai_message.payload.content[:30]}")
                response = await self._create_filtered_response(ai_message)
            else:
                # 1단계: 타입 분류
                print(f"[STEP] 메시지 타입 분류 시도")
                classified_type = await ai_classifier.classify_message_type(
                    ai_message.payload.content, 
                    ai_message.metadata
                )
                print(f"[INFO] 분류된 타입: {classified_type}")
                ai_message.payload.type = classified_type

                # 2단계: 메시지 처리
                print(f"[STEP] 메시지 처리 시작")
                response = await message_processor.process_message(ai_message)

            # 3. 봇 응답 저장
            print(f"[STEP] 봇 응답 DB 저장 시도")
            bot_chat_id = await self._save_bot_response(response)
            print(f"[DB] 봇 응답 저장 완료 - Chat ID: {bot_chat_id}")

            # 응답 메시지 ID 반영
            response["messageId"] = str(bot_chat_id)

            # 응답 Redis 전송
            print(f"[STEP] Redis 응답 전송 시도")
            await self._send_response(redis_service, response)
            print(f"[INFO] 응답 전송 완료 - 타입: {response['type']}")

            # ACK
            print(f"[STEP] 메시지 ACK 처리")
            await redis_service.acknowledge_message(
                settings.REQUEST_STREAM,
                settings.CONSUMER_GROUP,
                message_id
            )
            print(f"[INFO] 메시지 ACK 완료")

        except Exception as e:
            print(f"[ERROR] 메시지 처리 실패 - ID: {message_id}, 에러: {e}")

    async def _save_bot_response(self, response: Dict[str, Any]) -> int:
        db = SessionLocal()
        try:
            message = ChatMessage(
                user_id=int(response["userId"]),
                message=response["content"],
                sender="BOT",
                created_at=datetime.now(),
                message_type=response["type"]
            )
            db.add(message)
            db.commit()
            db.refresh(message)
            return message.chat_id
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
    
    async def _create_filtered_response(self, message: AiPromptMessage) -> Dict[str, Any]:
        """금칙어 감지 시 응답 생성"""
        return {
            "messageId": "temp",
            "userId": message.payload.user_id,
            "content": "부적절한 표현이 감지되었습니다. 정중한 언어를 사용해 주세요.",
            "type": MessageType.FILTERED_MESSAGE.value,
            "sender": "BOT",
            "timestamp": datetime.now().isoformat(),
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