import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from app.config import settings
from app.core.message_classifier import MessageClassifier
from app.core.csv_verification_system import CSVVerificationSystem
from app.core.memory_manager import MemoryManager

class RAGManager:
    def __init__(self):
        os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS
        )
        
        self.analysis_llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0.2,
            max_tokens=settings.MAX_TOKENS
        )
        
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        
        self.message_classifier = MessageClassifier(self.analysis_llm)
        
        pricing_dir = getattr(settings, 'PRICING_DATA_DIR', 'data/pricing')
        
        possible_paths = [
            pricing_dir,
            'data/pricing',
            'app/data/pricing',
            './app/data/pricing'
        ]
        
        self.csv_verifier = None
        for path in possible_paths:
            path_obj = Path(path)
            if path_obj.exists():
                try:
                    self.csv_verifier = CSVVerificationSystem(path)
                    print(f"CSV 검증 시스템 로드: {path}")
                    break
                except Exception as e:
                    print(f"CSV 로드 실패 ({path}): {e}")
                    continue
        
        if not self.csv_verifier:
            print(f"모든 경로에서 CSV 디렉토리를 찾을 수 없습니다")
            print(f"시도한 경로들: {possible_paths}")
        
        self.memory_manager = MemoryManager(
            memory_dir=settings.MEMORY_DIR,
            llm=self.llm
        )
        
        Path(settings.PRICING_DATA_DIR).mkdir(parents=True, exist_ok=True)
        Path(settings.VECTOR_STORE_DIR).mkdir(parents=True, exist_ok=True)
        Path(settings.MEMORY_DIR).mkdir(parents=True, exist_ok=True)
    
    def initialize(self):
        try:
            vector_path = Path(settings.VECTOR_STORE_DIR) / "faiss_index"
            if vector_path.exists():
                self.vectorstore = FAISS.load_local(
                    str(vector_path), 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print("기존 벡터스토어 로드됨")
            else:
                self._create_vectorstore_from_files()
            
            if self.vectorstore:
                self.retriever = self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": settings.RETRIEVAL_K}
                )
                self._setup_chain()
                print("RAG 체인 설정 완료")
                
        except Exception as e:
            print(f"RAG 시스템 초기화 오류: {e}")
            self.retriever = None
    
    def _create_vectorstore_from_files(self):
        pricing_dir = Path(settings.PRICING_DATA_DIR)
        files = list(pricing_dir.glob("*.txt")) + list(pricing_dir.glob("*.csv"))
        
        if not files:
            print("요금제 데이터 파일이 없습니다.")
            return
        
        all_documents = []
        for file_path in files:
            try:
                loader = TextLoader(str(file_path), encoding='utf-8')
                documents = loader.load()
                all_documents.extend(documents)
            except Exception as e:
                print(f"파일 로드 오류 {file_path}: {e}")
        
        if all_documents:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            splits = text_splitter.split_documents(all_documents)
            
            self.vectorstore = FAISS.from_documents(
                documents=splits, 
                embedding=self.embeddings
            )
            
            vector_path = Path(settings.VECTOR_STORE_DIR) / "faiss_index"
            self.vectorstore.save_local(str(vector_path))
            
            print(f"벡터스토어 생성 완료: {len(splits)}개 청크")
    
    def _setup_chain(self):
        system_prompt = """
        당신은 통신사 요금제 추천 전문가입니다. 
        사용자의 성향과 사용 패턴을 파악하여 가장 적합한 요금제를 추천해주세요.
        
        **현재 사용자: {user_id}**
        
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
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        
        if self.retriever:
            question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
            self.rag_chain = create_retrieval_chain(self.retriever, question_answer_chain)
        else:
            self.rag_chain = prompt | self.llm | StrOutputParser()
    
    def load_user_context(self, user_id: str) -> tuple:
        chat_history, conversation_summary, is_existing_user = self.memory_manager.load_user_memory(user_id)
        return chat_history, conversation_summary, is_existing_user
    
    def extract_plan_names_from_input(self, text: str) -> List[str]:
        """사용자 입력에서 요금제명 추출"""
        plan_names = []
        
        patterns = [
            r'(\S+)\s*요금제',
            r'요금제\s*(\S+)',
            r'(\S+\s+\S+)\s*요금제',
            r'(\S+)\s*플랜',
            r'(\S+)\s*plan'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            plan_names.extend(matches)
        
        cleaned_names = []
        for name in plan_names:
            cleaned = name.strip()
            if cleaned and len(cleaned) > 1:
                cleaned_names.append(cleaned)
        
        return list(set(cleaned_names))
    
    def chat(self, user_id: str, message: str) -> Dict[str, Any]:
        """기본 채팅 메서드 - 검증 포함"""
        return self.chat_with_verification(user_id, message)
    
    def chat_with_verification(self, user_id: str, message: str) -> Dict[str, Any]:
        try:
            asked_plans = self.extract_plan_names_from_input(message)
            
            if asked_plans and self.csv_verifier:
                for plan_name in asked_plans:
                    verification = self.csv_verifier.verify_plan_exists(plan_name)
                    
                    if verification['confidence'] < 0.5:
                        return {
                            "response": f"죄송합니다. '{plan_name}' 요금제는 현재 제공되지 않는 요금제입니다. 다른 요금제를 추천해드릴까요?",
                            "verification_status": "요금제 없음",
                            "mentioned_plans": [plan_name],
                            "confidence_score": verification['confidence'],
                            "message_type": "chat",
                            "verification_results": {
                                plan_name: {
                                    "plan_exists": False,
                                    "confidence_score": verification['confidence'],
                                    "match_type": verification['match_type'],
                                    "evidence": ["CSV 검증: 요금제 없음"]
                                }
                            },
                            "verification_method": "CSV 직접검증"
                        }
            
            chat_history, conversation_summary, _ = self.memory_manager.load_user_memory(user_id)
            chat_history_str = self.memory_manager.format_chat_history(chat_history, conversation_summary)
            
            if self.rag_chain is None:
                return {
                    "response": "죄송합니다. AI 시스템이 초기화되지 않았습니다.",
                    "verification_status": "시스템 오류",
                    "mentioned_plans": [],
                    "confidence_score": 0.0
                }
            
            if self.retriever:
                response = self.rag_chain.invoke({
                    "input": message,
                    "chat_history": chat_history_str,
                    "user_id": user_id
                })
                ai_response = response.get("answer", str(response))
                used_sources = response.get("context", [])
            else:
                response = self.rag_chain.invoke({
                    "input": message,
                    "chat_history": chat_history_str,
                    "context": "요금제 정보를 로드할 수 없습니다.",
                    "user_id": user_id
                })
                ai_response = str(response)
                used_sources = []
            
            classification = self.message_classifier.classify_message(message, ai_response)
            message_type = classification["message_type"]
            
            if self.csv_verifier:
                mentioned_plans = self.csv_verifier.find_mentioned_plans(ai_response)
            else:
                mentioned_plans = classification["mentioned_plans"]
            
            verification_results = {}
            overall_confidence = 1.0
            
            if message_type == "suggestion" and mentioned_plans and self.csv_verifier:
                for plan_name in mentioned_plans:
                    verification = self.csv_verifier.verify_plan_exists(plan_name)
                    verification_results[plan_name] = {
                        "plan_exists": verification['exists'],
                        "confidence_score": verification['confidence'],
                        "matched_plan": verification['matched_plan']['name'] if verification['matched_plan'] else None,
                        "match_type": verification['match_type'],
                        "discrepancies": [],
                        "evidence": [f"CSV 직접 검증: {verification['match_type']}"]
                    }
                    overall_confidence = min(overall_confidence, verification['confidence'])
            
            conversation_entry = {
                "timestamp": datetime.now().isoformat(),
                "human": message,
                "ai": ai_response,
                "message_type": message_type,
                "mentioned_plans": mentioned_plans,
                "confidence_score": overall_confidence if message_type == "suggestion" else None
            }
            
            chat_history.append(conversation_entry)
            
            if len(chat_history) > self.memory_manager.max_conversation_length:
                chat_history, conversation_summary = self.memory_manager.summarize_old_conversations(
                    user_id, chat_history, conversation_summary
                )
            
            self.memory_manager.save_user_memory(user_id, chat_history, conversation_summary)
            
            if self.csv_verifier:
                verification_status = self.csv_verifier.get_verification_status_message(overall_confidence)
            else:
                verification_status = "검증 시스템 없음"
            
            if overall_confidence < 0.7 and mentioned_plans:
                ai_response += f"\n\n검증 상태: {verification_status}\n신뢰도: {overall_confidence:.1%}"
                for plan_name, verification in verification_results.items():
                    if verification["confidence_score"] < 0.7:
                        if verification["match_type"] == "no_match":
                            ai_response += f"\n'{plan_name}' - 존재하지 않는 요금제일 수 있습니다"
                        elif verification["matched_plan"]:
                            ai_response += f"\n'{plan_name}' → '{verification['matched_plan']}' (유사한 요금제)"
            
            return {
                "response": ai_response,
                "verification_status": verification_status,
                "mentioned_plans": mentioned_plans,
                "confidence_score": overall_confidence,
                "message_type": message_type,
                "verification_results": verification_results,
                "used_knowledge": [doc.page_content[:100] + "..." for doc in used_sources],
                "verification_method": "CSV 직접검증" if self.csv_verifier else "분류기만"
            }
            
        except Exception as e:
            print(f"채팅 처리 오류: {e}")
            return {
                "response": f"죄송합니다. 오류가 발생했습니다: {e}",
                "verification_status": "오류",
                "mentioned_plans": [],
                "confidence_score": 0.0,
                "verification_method": "오류"
            }
    
    def verify_plan_directly(self, plan_name: str) -> Dict[str, Any]:
        if not self.csv_verifier:
            return {"error": "CSV 검증 시스템이 없습니다"}
        
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
                'confidence': verification['confidence'],
                'match_type': verification['match_type']
            }
        else:
            return {
                'verified': False,
                'confidence': verification['confidence'],
                'suggested_plan': verification['matched_plan']['name'] if verification['matched_plan'] else None,
                'message': f"'{plan_name}' 요금제를 찾을 수 없습니다.",
                'match_type': verification['match_type']
            }
    
    def search_plans_by_criteria(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.csv_verifier:
            return []
        
        results = self.csv_verifier.find_plans_by_criteria(
            price_range=criteria.get('price_range'),
            data_min=criteria.get('data_min'),
            age_code=criteria.get('age_code')
        )
        
        return [
            {
                'name': plan['name'],
                'price': f"{plan['price']:,}원",
                'data': plan['data'],
                'benefit': plan['benefit']
            }
            for plan in results
        ]
    
    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        chat_history, conversation_summary, _ = self.memory_manager.load_user_memory(user_id)
        stats = self.memory_manager.get_user_statistics(user_id, chat_history, conversation_summary)
        
        stats['verification_system'] = "CSV 직접검증" if self.csv_verifier else "없음"
        if self.csv_verifier:
            stats['total_plans_in_db'] = self.csv_verifier.get_plan_database_info()['total_plans']
        
        return stats
    
    def list_all_users(self) -> List[Dict[str, Any]]:
        return self.memory_manager.list_all_users()
    
    def delete_user_memory(self, username: str) -> bool:
        return self.memory_manager.delete_user_memory(username)
    
    def update_vectorstore(self, file_paths: List[str]) -> dict:
        try:
            result = self._update_vectorstore_internal(file_paths)
            return result
            
        except Exception as e:
            return {"success": False, "message": f"업데이트 오류: {e}"}
    
    def _update_vectorstore_internal(self, file_paths: List[str]) -> dict:
        all_documents = []
        for file_path in file_paths:
            path = Path(file_path)
            if path.exists():
                loader = TextLoader(str(path), encoding='utf-8')
                documents = loader.load()
                all_documents.extend(documents)
            else:
                print(f"파일을 찾을 수 없습니다: {file_path}")
        
        if not all_documents:
            return {"success": False, "message": "로드할 문서가 없습니다."}
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(all_documents)
        
        self.vectorstore = FAISS.from_documents(
            documents=splits, 
            embedding=self.embeddings
        )
        
        vector_path = Path(settings.VECTOR_STORE_DIR) / "faiss_index"
        self.vectorstore.save_local(str(vector_path))
        
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.RETRIEVAL_K}
        )
        self._setup_chain()
        
        return {
            "success": True, 
            "message": "벡터스토어 업데이트 완료",
            "chunks_created": len(splits)
        }
    
    def update_csv_verification(self, new_csv_path: str) -> dict:
        try:
            if not Path(new_csv_path).exists():
                return {"success": False, "message": f"CSV 파일을 찾을 수 없습니다: {new_csv_path}"}
            
            self.csv_verifier = CSVVerificationSystem(new_csv_path)
            plan_count = self.csv_verifier.get_plan_database_info()['total_plans']
            
            return {
                "success": True,
                "message": f"CSV 검증 시스템 업데이트 완료: {plan_count}개 요금제",
                "total_plans": plan_count
            }
            
        except Exception as e:
            return {"success": False, "message": f"CSV 업데이트 오류: {e}"}
    
    def get_plan_database_info(self) -> Dict[str, Any]:
        if self.csv_verifier:
            return self.csv_verifier.get_plan_database_info()
        else:
            return {
                "error": "CSV 검증 시스템이 없습니다",
                "total_plans": 0,
                "total_entries": 0
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        return {
            "rag_system": "사용 가능" if self.rag_chain else "사용 불가",
            "vectorstore": "로드됨" if self.vectorstore else "없음",
            "csv_verification": "사용 가능" if self.csv_verifier else "사용 불가",
            "total_plans": self.csv_verifier.get_plan_database_info()['total_plans'] if self.csv_verifier else 0,
            "memory_manager": "사용 가능"
        }
    
    def generate_verification_report(self, user_id: str, message: str) -> str:
        result = self.chat_with_verification(user_id, message)
        
        report = []
        report.append("=" * 60)
        report.append("할루시네이션 검증 보고서")
        report.append("=" * 60)
        report.append(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"사용자: {user_id}")
        report.append(f"입력: {message}")
        report.append(f"검증 방식: {result.get('verification_method', '알 수 없음')}")
        report.append("")
        
        report.append("전체 검증 결과")
        report.append("-" * 30)
        report.append(f"전체 신뢰도: {result['confidence_score']:.1%}")
        report.append(f"검증 상태: {result['verification_status']}")
        report.append(f"언급된 요금제 수: {len(result['mentioned_plans'])}개")
        report.append("")
        
        if result.get("verification_results"):
            report.append("요금제별 상세 검증")
            report.append("-" * 30)
            
            for plan_name, verification in result["verification_results"].items():
                report.append(f"요금제: {plan_name}")
                report.append(f"  존재 여부: {'예' if verification['plan_exists'] else '아니오'}")
                report.append(f"  신뢰도: {verification['confidence_score']:.1%}")
                report.append(f"  매칭 타입: {verification.get('match_type', '알 수 없음')}")
                
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
            report.append("검증 결과가 없습니다.")
            report.append("")
        
        report.append("챗봇 응답")
        report.append("-" * 30)
        report.append(result["response"])
        
        return "\n".join(report)