from app.models.message import AiPromptMessage, MessageType
from typing import Dict, Any
from datetime import datetime, timezone

class MessageProcessor:
    async def process_message(self, message: AiPromptMessage) -> Dict[str, Any]:
        """메시지 타입별 분기 처리"""
        message_type = message.payload.type

        if message_type == MessageType.USER_MESSAGE:
            return await self._handle_user_message(message)
        elif message_type == MessageType.SUGGESTION:
            return await self._handle_suggestion(message)
        elif message_type == MessageType.KEYWORD_RECOMMENDATION:
            return await self._handle_keyword_recommendation(message)
        elif message_type == MessageType.PREFERENCE_UPDATE:
            return await self._handle_preference_update(message)
        elif message_type == MessageType.PROACTIVE_SUGGESTION:
            return await self._handle_proactive_suggestion(message)
        elif message_type == MessageType.GENERAL_RESPONSE:
            return await self._handle_general_response(message)
        else:
            return await self._handle_unknown_type(message)

    async def _build_base_response(self, message: AiPromptMessage) -> Dict[str, Any]:
        """공통 응답 필드 생성"""
        return {
            "messageId": message.payload.message_id,
            "userId": message.payload.user_id,
            "content": "",
            "type": "",
            "sender": "BOT",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "bundleRecommendations": "",
            "extractedKeywords": "",
            "updatedPreference": ""
        }

    async def _handle_user_message(self, message: AiPromptMessage) -> Dict[str, Any]:
        """유저 메시지 처리 - 일반 응답으로 처리"""
        from app.main import rag_manager
        
        res = await self._build_base_response(message)
        
        if rag_manager:
            ai_result = rag_manager.chat(
                user_id=message.payload.user_id,
                message=message.payload.content
            )
            res.update({
                "type": MessageType.GENERAL_RESPONSE,
                "content": ai_result.get("response", "응답을 생성할 수 없습니다.")
            })
        else:
            res.update({
                "type": MessageType.GENERAL_RESPONSE,
                "content": "AI 시스템이 초기화되지 않았습니다."
            })
        
        return res

    async def _handle_suggestion(self, message: AiPromptMessage) -> Dict[str, Any]:
        """성향 기반 맞춤 추천"""
        from app.main import rag_manager
        
        res = await self._build_base_response(message)
        
        if rag_manager:
            ai_result = rag_manager.chat_with_verification(
                user_id=message.payload.user_id,
                message=message.payload.content
            )
            
            res.update({
                "type": MessageType.SUGGESTION,
                "content": ai_result.get("response", "추천을 생성할 수 없습니다."),
                "bundleRecommendations": ", ".join(ai_result.get("mentioned_plans", []))
            })
        else:
            res.update({
                "type": MessageType.SUGGESTION,
                "content": "AI 시스템이 초기화되지 않았습니다."
            })
        
        return res

    async def _handle_keyword_recommendation(self, message: AiPromptMessage) -> Dict[str, Any]:
        """키워드 기반 조합 추천"""
        from app.main import rag_manager
        
        res = await self._build_base_response(message)
        
        if rag_manager:
            # TODO: 키워드 추출 로직 구현
            ai_result = rag_manager.chat_with_verification(
                user_id=message.payload.user_id,
                message=message.payload.content
            )
            
            res.update({
                "type": MessageType.KEYWORD_RECOMMENDATION,
                "content": ai_result.get("response", "키워드 추천을 생성할 수 없습니다."),
                "extractedKeywords": ", ".join(ai_result.get("mentioned_plans", [])),
                "bundleRecommendations": ", ".join(ai_result.get("mentioned_plans", []))
            })
        else:
            res.update({
                "type": MessageType.KEYWORD_RECOMMENDATION,
                "content": "AI 시스템이 초기화되지 않았습니다."
            })
        
        return res

    async def _handle_preference_update(self, message: AiPromptMessage) -> Dict[str, Any]:
        """사용자 성향 업데이트"""
        from app.main import rag_manager
        
        res = await self._build_base_response(message)
        
        if rag_manager:
            # TODO: 성향 분석 및 업데이트 로직 구현
            ai_result = rag_manager.chat(
                user_id=message.payload.user_id,
                message=message.payload.content
            )
            
            res.update({
                "type": MessageType.PREFERENCE_UPDATE,
                "content": ai_result.get("response", "성향 업데이트를 처리할 수 없습니다."),
                "updatedPreference": "성향 업데이트됨"  # TODO: 실제 성향 데이터
            })
        else:
            res.update({
                "type": MessageType.PREFERENCE_UPDATE,
                "content": "AI 시스템이 초기화되지 않았습니다."
            })
        
        return res

    async def _handle_proactive_suggestion(self, message: AiPromptMessage) -> Dict[str, Any]:
        """주기적 조합 제안"""
        from app.main import rag_manager
        
        res = await self._build_base_response(message)
        
        if rag_manager:
            # TODO: 주기적 추천 로직 구현
            ai_result = rag_manager.chat_with_verification(
                user_id=message.payload.user_id,
                message=message.payload.content
            )
            
            res.update({
                "type": MessageType.PROACTIVE_SUGGESTION,
                "content": ai_result.get("response", "정기 추천을 생성할 수 없습니다."),
                "bundleRecommendations": ", ".join(ai_result.get("mentioned_plans", []))
            })
        else:
            res.update({
                "type": MessageType.PROACTIVE_SUGGESTION,
                "content": "AI 시스템이 초기화되지 않았습니다."
            })
        
        return res

    async def _handle_general_response(self, message: AiPromptMessage) -> Dict[str, Any]:
        """일반 대화 처리"""
        from app.main import rag_manager
        
        res = await self._build_base_response(message)
        
        if rag_manager:
            ai_result = rag_manager.chat(
                user_id=message.payload.user_id,
                message=message.payload.content
            )
            
            res.update({
                "type": MessageType.GENERAL_RESPONSE,
                "content": ai_result.get("response", "응답을 생성할 수 없습니다.")
            })
        else:
            res.update({
                "type": MessageType.GENERAL_RESPONSE,
                "content": "AI 시스템이 초기화되지 않았습니다."
            })
        
        return res

    async def _handle_unknown_type(self, message: AiPromptMessage) -> Dict[str, Any]:
        """알 수 없는 타입 처리"""
        res = await self._build_base_response(message)
        res.update({
            "type": MessageType.GENERAL_RESPONSE,
            "content": "죄송합니다. 요청을 처리할 수 없습니다."
        })
        return res

message_processor = MessageProcessor()