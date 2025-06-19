# app/services/consumer.py - 최종 수정 (메모리 저장 문제 해결)

import asyncio
import json
import redis.asyncio as redis
from typing import Dict, Any, Optional
import traceback
from datetime import datetime

from app.config import settings
from app.services.redis_service import RedisService

class StreamConsumer:
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.running: bool = False
        self.consumer_group: str = "ai-processors"
        self.consumer_name: str = "ai-server-1"
        self.stream_name: str = "ai-request-stream"
        
    async def connect_redis(self) -> bool:
        """Redis 연결 설정"""
        try:
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=getattr(settings, 'REDIS_DB', 0),
                password=getattr(settings, 'REDIS_PASSWORD', None),
                decode_responses=True,
                socket_connect_timeout=10,
                socket_timeout=10,
                retry_on_timeout=True
            )
            
            await self.redis_client.ping()
            print(f"✅ Redis 연결 성공: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
            return True
            
        except Exception as e:
            print(f"❌ Redis 연결 실패: {e}")
            return False
    
    async def create_consumer_group(self) -> bool:
        """컨슈머 그룹 생성"""
        try:
            await self.redis_client.xgroup_create(
                self.stream_name, 
                self.consumer_group, 
                id='0', 
                mkstream=True
            )
            print(f"✅ 컨슈머 그룹 '{self.consumer_group}' 생성됨")
            return True
        except redis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                print(f"ℹ️ 컨슈머 그룹 '{self.consumer_group}' 이미 존재")
                return True
            else:
                print(f"❌ 컨슈머 그룹 생성 실패: {e}")
                return False
    
    async def start_consuming(self):
        """메시지 컨슈밍 시작"""
        print(f"[INFO] 컨슈머 시작: {self.stream_name}")
        
        if not await self.connect_redis():
            print("❌ Redis 연결 실패로 컨슈머 시작 불가")
            return
        
        if not await self.create_consumer_group():
            print("❌ 컨슈머 그룹 생성 실패로 컨슈머 시작 불가")
            return
        
        self.running = True
        print(f"🚀 컨슈머 '{self.consumer_name}' 시작됨")
        
        while self.running:
            try:
                print(f"👀 새 메시지 대기 중... ({datetime.now().strftime('%H:%M:%S')})")
                
                messages = await self.redis_client.xreadgroup(
                    self.consumer_group,
                    self.consumer_name,
                    {self.stream_name: '>'},
                    count=1,
                    block=5000
                )
                
                if messages:
                    print(f"📨 새 메시지 수신!")
                    
                    for stream, msgs in messages:
                        for message_id, fields in msgs:
                            await self.process_message(message_id, fields)
                
            except asyncio.CancelledError:
                print("🛑 컨슈머 취소 신호 수신")
                break
            except Exception as e:
                print(f"❌ 컨슈머 오류: {e}")
                await asyncio.sleep(2)
        
        self.running = False
        print("[INFO] 컨슈머 중지됨")
        if self.redis_client:
            await self.redis_client.close()
    
    async def process_message(self, message_id: str, fields: Dict[str, Any]):
        """메시지 처리"""
        print(f"🔄 메시지 처리: {message_id}")
        
        try:
            # JSON 파싱
            message_data = None
            
            if 'data' in fields:
                try:
                    raw_data = fields['data']
                    
                    if isinstance(raw_data, str):
                        parsed_once = json.loads(raw_data)
                        
                        if isinstance(parsed_once, str):
                            message_data = json.loads(parsed_once)
                            print(f"✅ 이중 JSON 파싱 성공")
                        else:
                            message_data = parsed_once
                            print(f"✅ 단일 JSON 파싱 성공")
                    else:
                        message_data = raw_data
                        
                except json.JSONDecodeError as e:
                    print(f"❌ JSON 파싱 실패: {e}")
            
            if message_data is None:
                message_data = dict(fields)
            
            user_id = None
            message = None
            
            if isinstance(message_data, dict):
                if 'payload' in message_data:
                    payload = message_data['payload']
                    user_id = payload.get('userId') or payload.get('user_id')
                    message = payload.get('content') or payload.get('message')
                
                if not user_id or not message:
                    user_id = message_data.get('user_id') or message_data.get('userId')
                    message = message_data.get('message') or message_data.get('content')
            
            if user_id is not None:
                user_id = str(user_id)
            
            if not user_id or not message:
                print(f"❌ 필수 필드 누락 - user_id: {user_id}, message: {message}")
                return
            
            print(f"🎯 AI 처리 시작: user_id={user_id}, message='{message[:30]}...'")
            
            ai_response = await self.get_ai_response_with_memory(user_id, message)
            from app.services.ai_classifier import ai_classifier
            classification_result = await ai_classifier.classify_with_ai_response(
                user_input=message,
                ai_response=ai_response
            )
            print(f"🔍 메시지 분류 결과: {classification_result}")
            response_data = {
                'user_id': user_id,
                'original_message': message,
                'ai_response': ai_response,
                'message_type': classification_result.get('message_type', 'GENERAL_RESPONSE').value,
                'confidence_score': classification_result.get('confidence', 0.0),
                'mentioned_plans': ','.join(classification_result['mentioned_plans']) if classification_result['mentioned_plans'] else '',
                'processed_at': datetime.now().isoformat()
            }
            
            await self.redis_client.xadd('ai-response-stream', response_data)
            print(f"✅ 처리 완료: AI 응답 길이 {len(ai_response)}")
                
        except Exception as e:
            print(f"❌ 처리 오류: {e}")
            print(f"❌ 상세: {traceback.format_exc()}")
        finally:
            try:
                await self.redis_client.xack(
                    self.stream_name, 
                    self.consumer_group, 
                    message_id
                )
                print(f"✅ ACK: {message_id}")
            except Exception as ack_error:
                print(f"❌ ACK 실패: {ack_error}")
    
    async def get_ai_response_with_memory(self, user_id: str, message: str) -> str:
        """메모리 저장이 포함된 AI 응답 생성"""
        try:
            from app.main import rag_manager
            
            if not rag_manager:
                return "죄송합니다. AI 시스템이 초기화되지 않았습니다."
            
            from app.core.content_filter import content_filter
            print(f"🤖 RAG 매니저를 통한 AI 처리...")
            is_inappropriate = content_filter.contains_forbidden_content(message)
            if is_inappropriate:
                print(f"🚫 부적절한 메시지 감지됨: user_id={user_id}")
                return "죄송합니다. 부적절한 내용이 포함된 메시지는 처리할 수 없습니다. 정중하고 건전한 대화를 부탁드립니다."
            
           
            if hasattr(rag_manager, 'chat'):
                print(f"📞 chat 메서드 호출")
                result = rag_manager.chat(user_id, message)
            else:
                print(f"❌ chat 메서드 없음")
                return "죄송합니다. AI 시스템에 문제가 발생했습니다."
            
            # 결과 처리
            if isinstance(result, str):
                return result
            elif isinstance(result, dict):
                return result.get('response', str(result))
            else:
                return str(result)
                
        except Exception as e:
            print(f"❌ AI 처리 오류: {e}")
            # JSON 직렬화 오류인 경우 특별 처리
            if "JSON serializable" in str(e) or "MessageType" in str(e):
                print(f"🔧 JSON 직렬화 오류 감지 - 간단한 응답 반환")
                return f"안녕하세요 {user_id}님! 메시지를 잘 받았습니다. 무엇을 도와드릴까요?"
            return "죄송합니다. 처리 중 오류가 발생했습니다."
    
    def stop(self):
        """컨슈머 중지"""
        print("[INFO] 컨슈머 중지 요청")
        self.running = False

class ImageStreamConsumer:
    def __init__(self):
        self.running = False
        from app.core.img_analyze.image_analyze import analyzer
        self.analyzer = analyzer
    async def start_consuming(self):
        self.running = True
        async with RedisService() as redis_service:
            # Consumer Group 초기화
            await redis_service.init_stream_group(
                settings.IMAGE_REQUEST_STREAM,
                settings.IMAGE_CONSUMER_GROUP
            )
            print(f"[INFO] 이미지 컨슈머 시작: {settings.IMAGE_REQUEST_STREAM}")

            while self.running:
                try:
                    messages = await redis_service.consume_messages(
                        stream_name=settings.IMAGE_REQUEST_STREAM,
                        group_name=settings.IMAGE_CONSUMER_GROUP,
                        consumer_name=settings.IMAGE_CONSUMER_NAME,
                        count=1
                    )
                    for stream_name, stream_messages in messages:
                        for message_id, message_data in stream_messages:
                            await self._process_image_message(
                                redis_service, message_id, message_data
                            )
                except Exception as e:
                    print(f"[ERROR] 이미지 컨슈머 오류: {e}")
                    await asyncio.sleep(5)

    async def _process_image_message(self, redis_service, message_id, message_data):
        print(f"[INFO] 📸 Raw message_data: {message_data}")

        try:

            # ✅ keys 찍기 (이미 확인됨)
            print(f"[DEBUG] Message Data Keys: {message_data.keys()}")

            # ✅ robust: str or bytes 둘 다 대응
            payload_raw = (
                message_data.get("payload") or
                message_data.get(b"payload")
            )

            if payload_raw is None:
                raise ValueError("Payload not found!")

            # ✅ robust decode
            if isinstance(payload_raw, bytes):
                payload_raw = payload_raw.decode()

            payload = json.loads(payload_raw)

            print(f"[INFO] Decoded payload: {payload}")

            # ✅ 이후 동일
            user_id = payload["userId"]
            base64_image = payload["image"]

            print(f"[INFO] ✅ Parsed user_id: {user_id}")

            result = self.analyzer.analyze_image_and_tags(base64_image)
            print(f"[INFO] ✅ Analysis result: {result}")

            # 5️⃣ 응답 전송
            response = {
                "user_id": user_id,
                "summary": result["summary"],
                "tags": result["tags"],
                "message_type" : "IMAGE_ANALYSIS"
            }
            await redis_service.send_message(
                settings.IMAGE_RESPONSE_STREAM,
                response
            )
            print(f"[INFO] ✅ Response sent to {settings.IMAGE_RESPONSE_STREAM}")

        except Exception as e:
            print(f"[ERROR] 분석 중 오류: {e}")

        finally:
            await redis_service.acknowledge_message(
                settings.IMAGE_REQUEST_STREAM,
                settings.IMAGE_CONSUMER_GROUP,
                message_id
            )
            print(f"[INFO] ✅ ACK 완료: {message_id}")

    def stop(self):
        self.running = False



stream_consumer = StreamConsumer()
image_stream_consumer = ImageStreamConsumer()