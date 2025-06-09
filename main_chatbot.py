import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from message_classifier import MessageClassifier
from data_models import MessageType, ChatEntry
from data_models import HallucinationCheck, VerificationResult, ChatEntry
from verification_system import VerificationSystem
from memory_manager import MemoryManager
from rag_system import RAGSystem

class PricingPlanChatbot:
    def __init__(self, openai_api_key: str, pricing_data_file: str, memory_dir: str = "user_memories"):
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
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
        
        self.current_user: Optional[str] = None
        self.chat_history: List[Dict[str, Any]] = []
        self.conversation_summary: str = ""
        self.message_classifier = MessageClassifier(self.analysis_llm)
        self.rag_system = RAGSystem(self.embeddings)
        self.verification_system = VerificationSystem(self.analysis_llm)
        self.memory_manager = MemoryManager(memory_dir, self.llm)
        
        self.rag_system.setup_rag_system(pricing_data_file)
        
        if self.rag_system.is_available():
            self.verification_system.extract_plan_information(self.rag_system.get_vectorstore())
        
        self.setup_chatbot_chain()
    
    def setup_chatbot_chain(self):
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
        7. **중요**: 실제로 존재하는 요금제만 추천하세요. 요금제명과 정보를 정확히 확인해주세요.
        
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
        if not self.current_user:
            return "먼저 사용자 이름을 설정해주세요. login_user() 메소드를 사용하세요."
        
        try:
            chat_history_str = self.memory_manager.format_chat_history(
                self.chat_history, self.conversation_summary
            )
            
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
            
            if hasattr(self, 'message_classifier'):
                classification = self.message_classifier.classify_message(user_input, ai_response)
                message_type = classification["message_type"]
                mentioned_plans = classification["mentioned_plans"]
            else:
                # 키워드 추가 필요
                suggestion_keywords = ['요금제', '추천', '적합', '월', '원', 'GB', '혜택', '할인', '무제한', '데이터', '통화']
                
                if any(keyword in ai_response for keyword in suggestion_keywords):
                    message_type = "suggestion"
                    import re
                    valid_plans = []
                    #요금제명 기입 필요
                    exact_plans = re.findall(r'(라이트|스탠다드|프리미엄|학생|실버|데이터\s*전용|무제한)\s*요금제', ai_response)
                    family_plans = re.findall(r'가족\s*쉐어\s*\d+인', ai_response)
                    
                    valid_plans.extend(exact_plans)
                    valid_plans.extend(family_plans)
                    
                    mentioned_plans = [plan + " 요금제" if not plan.endswith("요금제") else plan for plan in valid_plans]
                    mentioned_plans = list(set(mentioned_plans))  
                else:
                    message_type = "chat"
                    mentioned_plans = []
            
            confidence_score = None
            if message_type == "suggestion" and mentioned_plans:
                confidence_scores = []
                
                for plan_name in mentioned_plans:
                    try:
                        check_result = self.verification_system.check_plan_hallucination(
                            plan_name, ai_response, self.rag_system.get_retriever()
                        )
                        confidence_scores.append(check_result.confidence_score)
                    except Exception as e:
                        confidence_scores.append(0.5) 
                
                if confidence_scores:
                    confidence_score = sum(confidence_scores) / len(confidence_scores)
            
            conversation_entry = {
                "timestamp": datetime.now().isoformat(),
                "human": user_input,
                "ai": ai_response,
                "message_type": message_type,
                "mentioned_plans": mentioned_plans,
                "confidence_score": confidence_score
            }
            
            self.chat_history.append(conversation_entry)
            
            if len(self.chat_history) > self.memory_manager.max_conversation_length:
                self.chat_history, self.conversation_summary = self.memory_manager.summarize_old_conversations(
                    self.current_user, self.chat_history, self.conversation_summary
                )
            
            self.memory_manager.save_user_memory(
                self.current_user, self.chat_history, self.conversation_summary
            )
            
            return ai_response
        
        except Exception as e:
            return "죄송합니다. 일시적인 오류가 발생했습니다. 다시 시도해주세요."
    
    def get_verified_chat_response(self, user_input: str) -> Dict[str, Any]:
        if not self.current_user:
            return {"error": "먼저 사용자 이름을 설정해주세요."}
        
        try:
            basic_response = self.chat(user_input)
            mentioned_plans = self.verification_system.find_mentioned_plans(basic_response)
            
            verification_results = {}
            overall_confidence = 1.0
            
            for plan_name in mentioned_plans:
                check_result = self.verification_system.check_plan_hallucination(
                    plan_name, basic_response, self.rag_system.get_retriever()
                )
                verification_results[plan_name] = {
                    "plan_exists": check_result.plan_exists,
                    "confidence_score": check_result.confidence_score,
                    "matched_plan": check_result.matched_plan,
                    "discrepancies": check_result.discrepancies,
                    "evidence": check_result.evidence
                }
                
                overall_confidence = min(overall_confidence, check_result.confidence_score)
            
            verification_status = self.verification_system.get_verification_status_message(overall_confidence)
            
            return {
                "response": basic_response,
                "mentioned_plans": mentioned_plans,
                "verification_results": verification_results,
                "overall_confidence": overall_confidence,
                "verification_status": verification_status,
                "has_verification_issues": overall_confidence < 0.7
            }
            
        except Exception as e:
            return {"error": f"검증된 응답 생성 실패: {e}"}
    
    def chat_with_verification(self, user_input: str) -> str:
        if not self.current_user:
            return "먼저 사용자 이름을 설정해주세요."
        
        result = self.get_verified_chat_response(user_input)
        
        if "error" in result:
            return f"오류: {result['error']}"
        
        response = result["response"]
        
        if result["mentioned_plans"] and result["has_verification_issues"]:
            verification_info = []
            verification_info.append(f"\n🔍 {result['verification_status']}")
            verification_info.append(f"신뢰도: {result['overall_confidence']:.1%}")
            
            for plan_name, verification in result["verification_results"].items():
                if verification["confidence_score"] < 0.7:
                    verification_info.append(f"'{plan_name}' 정보 확인 필요")
                    if verification["matched_plan"]:
                        verification_info.append(f"   → 유사 요금제: {verification['matched_plan']}")
            
            response += "\n" + "\n".join(verification_info)
        
        return response
    
    def generate_verification_report(self, user_input: str) -> str:
        if not self.current_user:
            return "먼저 로그인해주세요."
        
        result = self.get_verified_chat_response(user_input)
        
        if "error" in result:
            return f"검증 보고서 생성 실패: {result['error']}"
        
        report = []
        report.append("=" * 60)
        report.append("할루시네이션 검증 보고서")
        report.append("=" * 60)
        report.append(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"사용자: {self.current_user}")
        report.append(f"입력: {user_input}")
        report.append("")
        
        report.append("전체 검증 결과")
        report.append("-" * 30)
        report.append(f"전체 신뢰도: {result['overall_confidence']:.1%}")
        report.append(f"검증 상태: {result['verification_status']}")
        report.append(f"언급된 요금제 수: {len(result['mentioned_plans'])}개")
        report.append("")
        
        if result["mentioned_plans"]:
            report.append("요금제별 상세 검증")
            report.append("-" * 30)
            
            for plan_name, verification in result["verification_results"].items():
                report.append(f"요금제: {plan_name}")
                report.append(f"  존재 여부: {'예' if verification['plan_exists'] else '아니오'}")
                report.append(f"  신뢰도: {verification['confidence_score']:.1%}")
                
                if verification['matched_plan']:
                    report.append(f"  매칭된 요금제: {verification['matched_plan']}")
                
                if verification['evidence']:
                    report.append("  검증 근거:")
                    for evidence in verification['evidence']:
                        report.append(f"    • {evidence}")
                
                if verification['discrepancies']:
                    report.append("  발견된 문제:")
                    for discrepancy in verification['discrepancies']:
                        report.append(f"    • {discrepancy}")
                
                report.append("")
        else:
            report.append("언급된 요금제가 없습니다.")
            report.append("")
        
        report.append("챗봇 응답")
        report.append("-" * 30)
        report.append(result["response"])
        report.append("")
        
        if result["has_verification_issues"]:
            report.append("권장사항")
            report.append("-" * 30)
            report.append("• 언급된 요금제 정보를 공식 웹사이트에서 재확인하세요")
            report.append("• 'verify' 명령어로 데이터베이스 정보를 확인하세요")
            report.append("• 'check [요금제명]' 명령어로 특정 요금제를 검증하세요")
        else:
            report.append("모든 정보가 검증되었습니다.")
        
        return "\n".join(report)
    
    def check_plan_hallucination(self, plan_name: str, content: str = None) -> HallucinationCheck:
        if content is None:
            content = f"요금제명: {plan_name}"
        return self.verification_system.check_plan_hallucination(
            plan_name, content, self.rag_system.get_retriever()
        )
    
    def get_plan_database_info(self) -> Dict[str, Any]:
        return self.verification_system.get_plan_database_info()
    
    def list_all_users(self) -> List[Dict[str, Any]]:
        return self.memory_manager.list_all_users()
    
    def delete_user_memory(self, username: str) -> bool:
        return self.memory_manager.delete_user_memory(username)
    
    def get_user_statistics(self) -> Dict[str, Any]:
        if not self.current_user:
            return {"error": "현재 로그인된 사용자가 없습니다."}
        return self.memory_manager.get_user_statistics(
            self.current_user, self.chat_history, self.conversation_summary
        )
    
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
    
    def save_chat_history(self, filename: str = None):
        if not self.current_user:
            print("현재 로그인된 사용자가 없습니다.")
            return
        
        self.memory_manager.save_chat_history_export(
            self.current_user, self.chat_history, self.conversation_summary, filename
        )
    
    def get_export_file_info(self, username: str = None) -> Dict[str, Any]:
        if username is None:
            username = self.current_user
        if not username:
            return {"error": "사용자 이름이 없습니다."}
        return self.memory_manager.get_export_file_info(username)
    
    def get_chat_history(self) -> List[Dict[str, Any]]:
        return self.chat_history
    
    def get_message_statistics(self) -> Dict[str, Any]:
        if not self.current_user:
            return {"error": "로그인이 필요합니다."}
        
        chat_count = 0
        suggestion_count = 0
        total_confidence = 0
        suggestion_with_confidence = 0
        
        for entry in self.chat_history:
            message_type = entry.get("message_type", "chat")
            
            if message_type == "chat":
                chat_count += 1
            elif message_type == "suggestion":
                suggestion_count += 1
                
                confidence = entry.get("confidence_score")
                if confidence is not None:
                    total_confidence += confidence
                    suggestion_with_confidence += 1
        
        avg_confidence = (total_confidence / suggestion_with_confidence) if suggestion_with_confidence > 0 else 0
        
        return {
            "username": self.current_user,
            "total_messages": len(self.chat_history),
            "chat_messages": chat_count,
            "suggestion_messages": suggestion_count,
            "average_confidence": avg_confidence,
            "suggestion_ratio": suggestion_count / len(self.chat_history) if self.chat_history else 0
        }