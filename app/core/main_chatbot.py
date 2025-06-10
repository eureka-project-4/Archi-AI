import os
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from message_classifier import MessageClassifier
# from data_models import MessageType, ChatEntry, HallucinationCheck
from csv_verification_system import CSVVerificationSystem
from memory_manager import MemoryManager
from rag_system import RAGSystem

class PricingPlanChatbot:
    def __init__(self, openai_api_key: str, pricing_data_file: str, 
                 csv_file_path: str, memory_dir: str = "user_memories"):
        """
        Args:
            openai_api_key: OpenAI API 키
            pricing_data_file: RAG용 텍스트 파일 경로
            csv_file_path: 검증용 CSV 파일 경로 (필수)
            memory_dir: 사용자 메모리 저장 디렉토리
        """
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # LLM 설정
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1000
        )
        
        self.analysis_llm = ChatOpenAI(
            model="gpt-3.5-turbo", 
            temperature=0.2,
            max_tokens=1000
        )
        
        self.embeddings = OpenAIEmbeddings()
        
        # 상태 변수
        self.current_user: Optional[str] = None
        self.chat_history: List[Dict[str, Any]] = []
        self.conversation_summary: str = ""
        
        # 핵심 시스템들
        self.message_classifier = MessageClassifier(self.analysis_llm)
        self.rag_system = RAGSystem(self.embeddings)
        self.csv_verifier = CSVVerificationSystem(csv_file_path)  # 필수 컴포넌트
        self.memory_manager = MemoryManager(memory_dir, self.llm)
        
        # RAG 시스템 초기화
        self.rag_system.setup_rag_system(pricing_data_file)
        self.setup_chatbot_chain()
        
        print("✅ 챗봇 초기화 완료")
        print(f"   - RAG 시스템: {'사용' if self.rag_system.is_available() else '사용 불가'}")
        print(f"   - CSV 검증: {self.csv_verifier.get_plan_database_info()['total_plans']}개 요금제")
    
    def setup_chatbot_chain(self):
        """RAG 체인 설정"""
        system_prompt = """
        당신은 통신사 요금제 추천 전문가입니다. 
        사용자의 성향과 사용 패턴을 파악하여 가장 적합한 요금제를 추천해주세요.
        
        **현재 사용자: {current_user}**
        
        **지침:**
        1. 사용자와 친근하고 자연스럽게 대화하세요
        2. 사용자의 통화량, 데이터 사용량, 예산 등을 파악하세요  
        3. 제공된 요금제 정보를 바탕으로 정확한 추천을 해주세요
        4. 이전 대화 내용을 기억하고 연관성 있게 대화하세요
        5. 추천 이유를 명확하게 설명해주세요
        6. 기존 사용자라면 이전 대화를 참고하여 개인화된 서비스를 제공하세요
        
        **컨텍스트 정보:**
        {context}
        
        **이전 대화 내용:**
        {chat_history}
        """
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        
        if self.rag_system.is_available():
            question_answer_chain = create_stuff_documents_chain(self.llm, self.prompt)
            self.rag_chain = create_retrieval_chain(self.rag_system.get_retriever(), question_answer_chain)
        else:
            self.rag_chain = self.prompt | self.llm | StrOutputParser()
    
    def login_user(self, username: str) -> Dict[str, Any]:
        """사용자 로그인"""
        if not username or not username.strip():
            return {
                "success": False,
                "message": "올바른 사용자 이름을 입력해주세요.",
                "is_new_user": False
            }
        
        username = username.strip()
        self.current_user = username
        
        chat_history, conversation_summary, is_existing_user = self.memory_manager.load_user_memory(username)
        self.chat_history = chat_history
        self.conversation_summary = conversation_summary
        
        return {
            "success": True,
            "message": f"안녕하세요, {username}님!" + 
                      (" 이전 대화를 이어서 시작하겠습니다." if is_existing_user 
                       else " 새로운 대화를 시작하겠습니다."),
            "is_new_user": not is_existing_user,
            "username": username,
            "conversation_count": len(self.chat_history)
        }
    
    def logout_user(self) -> bool:
        """사용자 로그아웃"""
        if self.current_user:
            save_success = self.memory_manager.save_user_memory(
                self.current_user, self.chat_history, self.conversation_summary
            )
            
            if save_success:
                print(f"👋 {self.current_user}님, 안녕히 가세요! 대화 내용이 저장되었습니다.")
            else:
                print(f"⚠️ {self.current_user}님의 대화 저장 중 오류가 발생했습니다.")
            
            self.current_user = None
            self.chat_history = []
            self.conversation_summary = ""
            return save_success
        else:
            print("현재 로그인된 사용자가 없습니다.")
            return True
    
    def chat(self, user_input: str) -> str:
        """기본 채팅 (검증 없음)"""
        if not self.current_user:
            return "먼저 사용자 이름을 설정해주세요. login_user() 메소드를 사용하세요."
        
        try:
            chat_history_str = self.memory_manager.format_chat_history(
                self.chat_history, self.conversation_summary
            )
            
            # RAG 체인으로 응답 생성
            if self.rag_system.is_available():
                response = self.rag_chain.invoke({
                    "input": user_input,
                    "chat_history": chat_history_str,
                    "current_user": self.current_user
                })
                
                if isinstance(response, dict) and "answer" in response:
                    ai_response = response["answer"]
                else:
                    ai_response = str(response)
            else:
                response = self.rag_chain.invoke({
                    "input": user_input,
                    "chat_history": chat_history_str,
                    "context": "요금제 정보 파일을 로드할 수 없습니다.",
                    "current_user": self.current_user
                })
                ai_response = response
            
            # 메시지 분류
            classification = self.message_classifier.classify_message(user_input, ai_response)
            message_type = classification["message_type"]
            mentioned_plans = self.csv_verifier.find_mentioned_plans(ai_response)
            
            # 대화 기록 저장
            conversation_entry = {
                "timestamp": datetime.now().isoformat(),
                "human": user_input,
                "ai": ai_response,
                "message_type": message_type,
                "mentioned_plans": mentioned_plans
            }
            
            self.chat_history.append(conversation_entry)
            
            # 대화 요약 (필요시)
            if len(self.chat_history) > self.memory_manager.max_conversation_length:
                self.chat_history, self.conversation_summary = self.memory_manager.summarize_old_conversations(
                    self.current_user, self.chat_history, self.conversation_summary
                )
            
            # 메모리 저장
            self.memory_manager.save_user_memory(
                self.current_user, self.chat_history, self.conversation_summary
            )
            
            return ai_response
        
        except Exception as e:
            print(f"채팅 처리 중 오류: {e}")
            return "죄송합니다. 일시적인 오류가 발생했습니다. 다시 시도해주세요."
    
    def chat_with_verification(self, user_input: str) -> str:
        """검증이 포함된 채팅 (추천)"""
        if not self.current_user:
            return "먼저 사용자 이름을 설정해주세요."
        
        # 1. 기본 AI 응답 생성
        ai_response = self.chat(user_input)
        
        # 2. 요금제 언급 확인
        mentioned_plans = self.csv_verifier.find_mentioned_plans(ai_response)
        
        if not mentioned_plans:
            return ai_response
        
        # 3. CSV 직접 검증
        verification_issues = []
        verified_plans = []
        
        for plan_name in mentioned_plans:
            verification = self.csv_verifier.verify_plan_exists(plan_name)
            
            if verification['confidence'] >= 0.8:
                verified_plans.append(plan_name)
            else:
                verification_issues.append({
                    'plan_name': plan_name,
                    'issue': verification['match_type'],
                    'confidence': verification['confidence'],
                    'suggested': verification['matched_plan']['name'] if verification['matched_plan'] else None
                })
        
        # 4. 검증 결과 추가
        if verification_issues:
            warning_msg = "\n\n🔍 **정보 검증 결과:**\n"
            
            for issue in verification_issues:
                if issue['issue'] == 'no_match':
                    warning_msg += f"⚠️ '{issue['plan_name']}' - 존재하지 않는 요금제입니다.\n"
                elif issue['suggested']:
                    warning_msg += f"🔄 '{issue['plan_name']}' → '{issue['suggested']}' (유사한 요금제)\n"
                else:
                    warning_msg += f"❓ '{issue['plan_name']}' - 정확한 요금제명을 확인해주세요.\n"
            
            warning_msg += "\n💡 정확한 정보는 공식 홈페이지에서 확인해주세요."
            ai_response += warning_msg
        
        return ai_response
    
    def verify_plan(self, plan_name: str) -> Dict[str, Any]:
        """특정 요금제 검증"""
        verification = self.csv_verifier.verify_plan_exists(plan_name)
        
        if verification['exists']:
            plan_info = verification['matched_plan']
            return {
                'verified': True,
                'plan_name': plan_info['name'],
                'price': f"{plan_info['price']:,}원",
                'data': plan_info['data'],
                'calls': plan_info['calls'],
                'sms': plan_info['sms'],
                'benefit': plan_info['benefit'],
                'confidence': verification['confidence']
            }
        else:
            return {
                'verified': False,
                'confidence': verification['confidence'],
                'suggested_plan': verification['matched_plan']['name'] if verification['matched_plan'] else None,
                'message': f"'{plan_name}' 요금제를 찾을 수 없습니다."
            }
    
    def search_plans(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """조건별 요금제 검색"""
        price_range = criteria.get('price_range')  # (min, max)
        data_min = criteria.get('data_min')        # GB 단위
        age_code = criteria.get('age_code')        # 001: 유쓰, 002: 시니어, etc.
        
        results = self.csv_verifier.find_plans_by_criteria(
            price_range=price_range,
            data_min=data_min, 
            age_code=age_code
        )
        
        # 결과 포맷팅
        formatted_results = []
        for plan in results:
            formatted_results.append({
                'name': plan['name'],
                'price': f"{plan['price']:,}원",
                'data': plan['data'],
                'benefit': plan['benefit']
            })
        
        return formatted_results
    
    def get_verification_report(self, text: str) -> Dict[str, Any]:
        """텍스트의 검증 보고서 생성"""
        mentioned_plans = self.csv_verifier.find_mentioned_plans(text)
        
        report = {
            'total_mentioned': len(mentioned_plans),
            'verified_plans': [],
            'unverified_plans': [],
            'overall_accuracy': 0.0
        }
        
        if not mentioned_plans:
            report['message'] = "언급된 요금제가 없습니다."
            return report
        
        total_confidence = 0
        for plan_name in mentioned_plans:
            verification = self.csv_verifier.verify_plan_exists(plan_name)
            total_confidence += verification['confidence']
            
            if verification['exists']:
                report['verified_plans'].append({
                    'name': plan_name,
                    'confidence': verification['confidence'],
                    'actual_info': verification['matched_plan']
                })
            else:
                report['unverified_plans'].append({
                    'name': plan_name,
                    'confidence': verification['confidence'],
                    'suggested': verification['matched_plan']['name'] if verification['matched_plan'] else None
                })
        
        report['overall_accuracy'] = total_confidence / len(mentioned_plans)
        return report
    
    # 기존 메서드들 (메모리 관리 등)
    def get_chat_history(self) -> List[Dict[str, Any]]:
        return self.chat_history
    
    def clear_chat_history(self):
        if not self.current_user:
            print("현재 로그인된 사용자가 없습니다.")
            return
        
        self.chat_history = []
        self.conversation_summary = ""
        self.memory_manager.save_user_memory(
            self.current_user, self.chat_history, self.conversation_summary
        )
        print(f"{self.current_user}님의 대화 기록이 초기화되었습니다.")
    
    def get_statistics(self) -> Dict[str, Any]:
        """시스템 통계"""
        if not self.current_user:
            return {"error": "로그인이 필요합니다."}
        
        chat_count = sum(1 for entry in self.chat_history if entry.get("message_type") == "chat")
        suggestion_count = sum(1 for entry in self.chat_history if entry.get("message_type") == "suggestion")
        
        return {
            "user": self.current_user,
            "total_messages": len(self.chat_history),
            "chat_messages": chat_count,
            "suggestion_messages": suggestion_count,
            "suggestion_ratio": suggestion_count / len(self.chat_history) if self.chat_history else 0,
            "verification_system": "CSV 직접검증",
            "total_plans_in_db": self.csv_verifier.get_plan_database_info()['total_plans']
        }