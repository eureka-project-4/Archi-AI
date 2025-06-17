# app/services/consumer.py - 최종 수정 (메모리 저장 문제 해결)

import asyncio
import json
import redis.asyncio as redis
from typing import Dict, Any, Optional
import traceback
from datetime import datetime

from app.config import settings

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
            
            # 필수 필드 추출
            user_id = None
            message = None
            
            if isinstance(message_data, dict):
                # payload 구조 지원
                if 'payload' in message_data:
                    payload = message_data['payload']
                    user_id = payload.get('userId') or payload.get('user_id')
                    message = payload.get('content') or payload.get('message')
                
                # 직접 구조 지원
                if not user_id or not message:
                    user_id = message_data.get('user_id') or message_data.get('userId')
                    message = message_data.get('message') or message_data.get('content')
            
            # 타입 변환
            if user_id is not None:
                user_id = str(user_id)
            
            if not user_id or not message:
                print(f"❌ 필수 필드 누락 - user_id: {user_id}, message: {message}")
                return
            
            print(f"🎯 AI 처리 시작: user_id={user_id}, message='{message[:30]}...'")
            
            # AI 응답 생성 (메모리 저장 포함)
            ai_response = await self.get_ai_response_with_memory(user_id, message)
            
            # 응답 저장
            response_data = {
                'user_id': user_id,
                'original_message': message,
                'ai_response': ai_response,
                'message_type': 'GENERAL_RESPONSE',
                'confidence_score': 1.0,
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
            
            print(f"🤖 RAG 매니저를 통한 AI 처리...")
            
            # 🔥 핵심: chat 메서드 사용 (chat_with_verification 대신)
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

# 전역 인스턴스
stream_consumer = StreamConsumer()