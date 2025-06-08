import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.summarize import load_summarize_chain

from dotenv import load_dotenv

# 환경변수 로드
load_dotenv(override=True)

class PricingPlanChatbot:
    def __init__(self, openai_api_key: str, pricing_data_file: str, memory_dir: str = "user_memories"):
        """
        요금제 추천 챗봇 초기화
        
        Args:
            openai_api_key: OpenAI API 키
            pricing_data_file: 요금제 정보가 담긴 텍스트 파일 경로
            memory_dir: 사용자 메모리 파일들이 저장될 디렉토리
        """
        # OpenAI API 키 설정
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # LLM 초기화
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1000
        )
        
        # 임베딩 모델 초기화
        self.embeddings = OpenAIEmbeddings()
        
        # 메모리 디렉토리 설정 및 생성
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        
        # export 디렉토리 설정 및 생성
        self.export_dir = Path("chat_history")
        self.export_dir.mkdir(exist_ok=True)
        
        # 현재 사용자 정보
        self.current_user: Optional[str] = None
        self.chat_history: List[Dict[str, Any]] = []
        self.conversation_summary: str = ""  # 대화 요약 저장
        
        # 대화 관리 설정
        self.max_conversation_length = 10  # 최대 저장할 대화 수
        self.summary_threshold = 8  # 이 개수를 넘으면 요약 시작
        
        # RAG 시스템 초기화
        self.setup_rag_system(pricing_data_file)
        
        # 챗봇 체인 생성
        self.setup_chatbot_chain()
    
    def setup_rag_system(self, pricing_data_file: str):
        """RAG 시스템 설정"""
        try:
            # 텍스트 파일 로드
            loader = TextLoader(pricing_data_file, encoding='utf-8')
            documents = loader.load()
            
            # 문서 분할
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            splits = text_splitter.split_documents(documents)
            
            # 벡터 스토어 생성
            self.vectorstore = FAISS.from_documents(
                documents=splits, 
                embedding=self.embeddings
            )
            
            # 리트리버 설정
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            
            print(f"RAG 시스템 초기화 완료: {len(splits)}개의 문서 청크 생성됨")
            
        except Exception as e:
            print(f"RAG 시스템 초기화 중 오류 발생: {e}")
            self.retriever = None
    
    def setup_chatbot_chain(self):
        """챗봇 체인 설정"""
        # 시스템 프롬프트 정의
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
        
        # 프롬프트 템플릿 생성
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        
        if self.retriever:
            # RAG 체인 생성
            question_answer_chain = create_stuff_documents_chain(self.llm, self.prompt)
            self.rag_chain = create_retrieval_chain(self.retriever, question_answer_chain)
        else:
            # RAG 없이 기본 체인 생성
            self.rag_chain = self.prompt | self.llm | StrOutputParser()
    
    def get_memory_file_path(self, username: str) -> Path:
        """사용자별 메모리 파일 경로 생성"""
        # 안전한 파일명 생성 (특수문자 제거)
        safe_username = "".join(c for c in username if c.isalnum() or c in ('-', '_')).strip()
        if not safe_username:
            safe_username = "unknown_user"
        return self.memory_dir / f"{safe_username}_memory.json"
    
    def load_user_memory(self, username: str) -> bool:
        """
        사용자 메모리 로드
        
        Args:
            username: 사용자 이름
            
        Returns:
            bool: 기존 메모리가 있었는지 여부
        """
        self.current_user = username
        memory_file = self.get_memory_file_path(username)
        
        if memory_file.exists():
            try:
                with open(memory_file, 'r', encoding='utf-8') as f:
                    user_data = json.load(f)
                    
                self.chat_history = user_data.get('chat_history', [])
                self.conversation_summary = user_data.get('conversation_summary', "")
                last_login = user_data.get('last_login', 'Unknown')
                
                print(f"✅ {username}님의 기존 대화 기록을 불러왔습니다.")
                print(f"   - 저장된 대화 수: {len(self.chat_history)}개")
                print(f"   - 대화 요약 존재: {'예' if self.conversation_summary else '아니오'}")
                print(f"   - 마지막 접속: {last_login}")
                return True
                
            except Exception as e:
                print(f"⚠️ 메모리 파일 로드 중 오류 발생: {e}")
                self.chat_history = []
                self.conversation_summary = ""
                return False
        else:
            self.chat_history = []
            self.conversation_summary = ""
            print(f"👋 {username}님, 처음 뵙겠습니다! 새로운 대화를 시작하겠습니다.")
            return False
    
    def save_user_memory(self) -> bool:
        """
        현재 사용자의 메모리 저장
        
        Returns:
            bool: 저장 성공 여부
        """
        if not self.current_user:
            print("⚠️ 현재 로그인된 사용자가 없습니다.")
            return False
        
        try:
            memory_file = self.get_memory_file_path(self.current_user)
            
            user_data = {
                'username': self.current_user,
                'chat_history': self.chat_history,
                'conversation_summary': self.conversation_summary,
                'last_login': datetime.now().isoformat(),
                'total_conversations': len(self.chat_history),
                'created_at': datetime.now().isoformat() if not memory_file.exists() else None
            }
            
            # 기존 파일이 있으면 created_at 유지
            if memory_file.exists():
                with open(memory_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    user_data['created_at'] = existing_data.get('created_at', datetime.now().isoformat())
            
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(user_data, f, ensure_ascii=False, indent=2)
            
            return True
            
        except Exception as e:
            print(f"⚠️ 메모리 저장 중 오류 발생: {e}")
            return False
    
    def list_all_users(self) -> List[Dict[str, Any]]:
        """저장된 모든 사용자 목록 반환"""
        users = []
        
        for memory_file in self.memory_dir.glob("*_memory.json"):
            try:
                with open(memory_file, 'r', encoding='utf-8') as f:
                    user_data = json.load(f)
                    
                users.append({
                    'username': user_data.get('username', 'Unknown'),
                    'total_conversations': user_data.get('total_conversations', 0),
                    'last_login': user_data.get('last_login', 'Unknown'),
                    'created_at': user_data.get('created_at', 'Unknown')
                })
                
            except Exception as e:
                print(f"사용자 파일 읽기 오류 ({memory_file}): {e}")
                
        return sorted(users, key=lambda x: x['last_login'], reverse=True)
    
    def delete_user_memory(self, username: str) -> bool:
        """사용자 메모리 삭제"""
        try:
            memory_file = self.get_memory_file_path(username)
            if memory_file.exists():
                memory_file.unlink()
                print(f"✅ {username}님의 대화 기록이 삭제되었습니다.")
                return True
            else:
                print(f"⚠️ {username}님의 대화 기록을 찾을 수 없습니다.")
                return False
        except Exception as e:
            print(f"⚠️ 메모리 삭제 중 오류 발생: {e}")
            return False
    
    def summarize_old_conversations(self) -> str:
        """오래된 대화들을 요약"""
        if len(self.chat_history) <= self.summary_threshold:
            return self.conversation_summary
        
        try:
            # 요약할 대화들 (오래된 순서로)
            conversations_to_summarize = self.chat_history[:-self.summary_threshold]
            
            # 대화를 텍스트로 변환
            conversation_text = ""
            for conv in conversations_to_summarize:
                conversation_text += f"사용자: {conv['human']}\n"
                conversation_text += f"챗봇: {conv['ai']}\n\n"
            
            # 요약 프롬프트 생성
            summary_prompt = ChatPromptTemplate.from_template("""
            다음은 {username}님과의 요금제 상담 대화 내용입니다. 
            중요한 정보들을 요약해주세요:
            
            1. 사용자의 요구사항 (예산, 데이터 사용량, 선호도 등)
            2. 추천했던 요금제들
            3. 사용자의 반응이나 결정사항
            4. 기타 중요한 맥락 정보
            
            대화 내용:
            {conversation_text}
            
            요약 (한국어로 간결하게):
            """)
            
            # 기존 요약이 있으면 포함
            if self.conversation_summary:
                conversation_text = f"[이전 대화 요약]\n{self.conversation_summary}\n\n[새로운 대화]\n{conversation_text}"
            
            # 요약 생성
            summary_chain = summary_prompt | self.llm | StrOutputParser()
            new_summary = summary_chain.invoke({
                "username": self.current_user,
                "conversation_text": conversation_text
            })
            
            # 대화 기록에서 요약된 부분 제거하고 최근 대화만 유지
            self.chat_history = self.chat_history[-self.summary_threshold:]
            self.conversation_summary = new_summary
            
            print(f"💾 대화 요약 완료: {len(conversations_to_summarize)}개 대화가 요약되었습니다.")
            
            return new_summary
            
        except Exception as e:
            print(f"⚠️ 대화 요약 중 오류 발생: {e}")
            return self.conversation_summary
    
    def format_chat_history(self) -> str:
        """대화 기록을 문자열로 포맷팅 (요약 포함)"""
        formatted_parts = []
        
        # 기존 대화 요약이 있으면 포함
        if self.conversation_summary:
            formatted_parts.append("[이전 대화 요약]")
            formatted_parts.append(self.conversation_summary)
            formatted_parts.append("\n[최근 대화]")
        
        # 최근 대화 기록
        if not self.chat_history:
            if not self.conversation_summary:
                formatted_parts.append("이전 대화 내용이 없습니다.")
        else:
            for entry in self.chat_history:
                formatted_parts.append(f"사용자: {entry['human']}")
                formatted_parts.append(f"챗봇: {entry['ai']}")
        
        return "\n".join(formatted_parts)
    
    def chat(self, user_input: str) -> str:
        """
        사용자 입력에 대한 챗봇 응답 생성
        
        Args:
            user_input: 사용자 입력 텍스트
            
        Returns:
            챗봇 응답 텍스트
        """
        if not self.current_user:
            return "⚠️ 먼저 사용자 이름을 설정해주세요. login_user() 메소드를 사용하세요."
        
        try:
            # 채팅 히스토리 포맷팅
            chat_history_str = self.format_chat_history()
            
            if self.retriever:
                # RAG 체인 실행
                response = self.rag_chain.invoke({
                    "input": user_input,
                    "chat_history": chat_history_str,
                    "current_user": self.current_user
                })
                
                # RAG 체인의 응답에서 answer 추출
                if isinstance(response, dict) and "answer" in response:
                    ai_response = response["answer"]
                else:
                    ai_response = str(response)
            else:
                # 기본 체인 실행
                response = self.rag_chain.invoke({
                    "input": user_input,
                    "chat_history": chat_history_str,
                    "context": "요금제 정보 파일을 로드할 수 없습니다.",
                    "current_user": self.current_user
                })
                ai_response = response
            
            # 대화 기록에 추가
            conversation_entry = {
                "timestamp": datetime.now().isoformat(),
                "human": user_input,
                "ai": ai_response
            }
            self.chat_history.append(conversation_entry)
            
            # 대화가 임계점을 넘으면 요약 수행
            if len(self.chat_history) > self.max_conversation_length:
                self.summarize_old_conversations()
            
            # 자동 저장
            self.save_user_memory()
            
            return ai_response
            
        except Exception as e:
            error_msg = f"챗봇 응답 생성 중 오류가 발생했습니다: {e}"
            print(error_msg)
            return "죄송합니다. 일시적인 오류가 발생했습니다. 다시 시도해주세요."
    
    def login_user(self, username: str) -> Dict[str, Any]:
        """
        사용자 로그인 (메모리 로드)
        
        Args:
            username: 사용자 이름
            
        Returns:
            Dict: 로그인 결과 정보
        """
        if not username or not username.strip():
            return {
                "success": False,
                "message": "올바른 사용자 이름을 입력해주세요.",
                "is_new_user": False
            }
        
        username = username.strip()
        is_existing_user = self.load_user_memory(username)
        
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
        """현재 사용자 로그아웃"""
        if self.current_user:
            # 최종 메모리 저장
            save_success = self.save_user_memory()
            
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
    
    def get_chat_history(self) -> List[Dict[str, Any]]:
        """대화 기록 반환"""
        return self.chat_history
    
    def clear_chat_history(self):
        """현재 사용자의 대화 기록 초기화"""
        if not self.current_user:
            print("현재 로그인된 사용자가 없습니다.")
            return
        
        self.chat_history = []
        self.conversation_summary = ""
        self.save_user_memory()
        print(f"{self.current_user}님의 대화 기록이 초기화되었습니다.")
    
    def save_chat_history(self, filename: str = None):
        """대화 기록을 별도 파일로 저장 (사용자별 고정 파일명)"""
        if not self.current_user:
            print("현재 로그인된 사용자가 없습니다.")
            return
        
        if filename is None:
            # 사용자별 고정된 파일명 사용 (chat_history 폴더에 저장)
            safe_username = "".join(c for c in self.current_user if c.isalnum() or c in ('-', '_')).strip()
            filename = self.export_dir / f"{safe_username}_chat_export.json"
        
        try:
            export_data = {
                'username': self.current_user,
                'export_date': datetime.now().isoformat(),
                'total_conversations': len(self.chat_history),
                'conversation_summary': self.conversation_summary,
                'chat_history': self.chat_history,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            print(f"대화 기록이 {filename}에 저장되었습니다.")
        except Exception as e:
            print(f"대화 기록 저장 중 오류 발생: {e}")
    
    def get_export_file_info(self, username: str = None) -> Dict[str, Any]:
        """export 파일 정보 조회"""
        if username is None:
            username = self.current_user
            
        if not username:
            return {"error": "사용자 이름이 없습니다."}
        
        safe_username = "".join(c for c in username if c.isalnum() or c in ('-', '_')).strip()
        export_filename = self.export_dir / f"{safe_username}_chat_export.json"
        
        if export_filename.exists():
            try:
                with open(export_filename, 'r', encoding='utf-8') as f:
                    export_data = json.load(f)
                
                return {
                    'exists': True,
                    'filename': str(export_filename),
                    'last_updated': export_data.get('last_updated', 'Unknown'),
                    'export_date': export_data.get('export_date', 'Unknown'),
                    'total_conversations': export_data.get('total_conversations', 0),
                    'file_size': export_filename.stat().st_size
                }
            except Exception as e:
                return {"error": f"파일 읽기 오류: {e}"}
        else:
            return {
                'exists': False,
                'filename': str(export_filename)
            }
    
    def delete_export_file(self, username: str = None) -> bool:
        """export 파일 삭제"""
        if username is None:
            username = self.current_user
            
        if not username:
            print("사용자 이름이 없습니다.")
            return False
        
        safe_username = "".join(c for c in username if c.isalnum() or c in ('-', '_')).strip()
        export_filename = self.export_dir / f"{safe_username}_chat_export.json"
        
        try:
            if export_filename.exists():
                export_filename.unlink()
                print(f"✅ {export_filename} 파일이 삭제되었습니다.")
                return True
            else:
                print(f"⚠️ {export_filename} 파일을 찾을 수 없습니다.")
                return False
        except Exception as e:
            print(f"⚠️ 파일 삭제 중 오류 발생: {e}")
            return False
    
    def get_user_statistics(self) -> Dict[str, Any]:
        """현재 사용자의 통계 정보 반환"""
        if not self.current_user:
            return {"error": "현재 로그인된 사용자가 없습니다."}
        
        memory_file = self.get_memory_file_path(self.current_user)
        
        try:
            if memory_file.exists():
                with open(memory_file, 'r', encoding='utf-8') as f:
                    user_data = json.load(f)
                
                return {
                    'username': self.current_user,
                    'total_conversations': len(self.chat_history),
                    'has_summary': bool(self.conversation_summary),
                    'summary_length': len(self.conversation_summary) if self.conversation_summary else 0,
                    'first_visit': user_data.get('created_at', 'Unknown'),
                    'last_login': user_data.get('last_login', 'Unknown'),
                    'current_session_messages': len(self.chat_history)
                }
            else:
                return {
                    'username': self.current_user,
                    'total_conversations': len(self.chat_history),
                    'has_summary': bool(self.conversation_summary),
                    'summary_length': len(self.conversation_summary) if self.conversation_summary else 0,
                    'first_visit': 'Current session',
                    'last_login': 'Current session',
                    'current_session_messages': len(self.chat_history)
                }
        except Exception as e:
            return {"error": f"통계 정보 조회 중 오류 발생: {e}"}


def main():
    """챗봇 사용 예시"""
    # 환경변수에서 API 키 읽기
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PRICING_DATA_FILE = os.getenv("PRICING_DATA_FILE")  # 요금제 정보 파일 경로
    # API 키 확인
    if not OPENAI_API_KEY:
        print("⚠️ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("   .env 파일에 OPENAI_API_KEY=your-api-key 를 추가해주세요.")
        return
    
    print(f"✅ OpenAI API Key 로드됨: {OPENAI_API_KEY[:8]}...{OPENAI_API_KEY[-4:]}")
    

    
    # 챗봇 초기화
    try:
        chatbot = PricingPlanChatbot(OPENAI_API_KEY, PRICING_DATA_FILE)
    except Exception as e:
        print(f"❌ 챗봇 초기화 실패: {e}")
        return
    
    print("=== 요금제 추천 챗봇 ===")
    print("명령어:")
    print("  - 'users': 저장된 사용자 목록 보기")
    print("  - 'login [이름]': 사용자 로그인")
    print("  - 'logout': 현재 사용자 로그아웃")
    print("  - 'stats': 현재 사용자 통계")
    print("  - 'summary': 대화 요약 보기")
    print("  - 'clear': 대화 기록 초기화")
    print("  - 'export': 대화 기록 내보내기")
    print("  - 'export-info': export 파일 정보 확인")
    print("  - 'delete [이름]': 사용자 데이터 삭제")
    print("  - 'quit' 또는 'exit': 종료")
    print("-" * 50)
    
    while True:
        try:
            if chatbot.current_user:
                user_input = input(f"\n[{chatbot.current_user}] 입력: ").strip()
            else:
                user_input = input("\n[로그인 필요] 입력: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '종료']:
                if chatbot.current_user:
                    chatbot.logout_user()
                print("챗봇을 종료합니다. 안녕히 가세요!")
                break
            
            if not user_input:
                continue
            
            # 명령어 처리
            if user_input.lower() == 'users':
                users = chatbot.list_all_users()
                if users:
                    print("\n📋 저장된 사용자 목록:")
                    for i, user in enumerate(users, 1):
                        print(f"  {i}. {user['username']} "
                              f"(대화 {user['total_conversations']}개, "
                              f"최근 접속: {user['last_login'][:19]})")
                else:
                    print("저장된 사용자가 없습니다.")
                continue
            
            elif user_input.lower().startswith('login '):
                username = user_input[6:].strip()
                if username:
                    result = chatbot.login_user(username)
                    print(f"\n{result['message']}")
                    if result['success'] and not result['is_new_user']:
                        print(f"저장된 대화 수: {result['conversation_count']}개")
                else:
                    print("사용자 이름을 입력해주세요. 예: login 홍길동")
                continue
            
            elif user_input.lower() == 'logout':
                chatbot.logout_user()
                continue
            
            elif user_input.lower() == 'stats':
                stats = chatbot.get_user_statistics()
                if 'error' in stats:
                    print(f"⚠️ {stats['error']}")
                else:
                    print(f"\n📊 {stats['username']}님의 통계:")
                    print(f"  - 총 대화 수: {stats['total_conversations']}개")
                    print(f"  - 대화 요약 존재: {'예' if stats['has_summary'] else '아니오'}")
                    if stats['has_summary']:
                        print(f"  - 요약 길이: {stats['summary_length']}자")
                    print(f"  - 첫 방문: {stats['first_visit'][:19]}")
                    print(f"  - 마지막 접속: {stats['last_login'][:19]}")
                continue
            
            elif user_input.lower() == 'summary':
                if not chatbot.current_user:
                    print("⚠️ 먼저 로그인해주세요.")
                elif chatbot.conversation_summary:
                    print(f"\n📝 {chatbot.current_user}님의 대화 요약:")
                    print("-" * 40)
                    print(chatbot.conversation_summary)
                    print("-" * 40)
                else:
                    print("저장된 대화 요약이 없습니다.")
                continue
            
            elif user_input.lower() == 'clear':
                chatbot.clear_chat_history()
                continue
            
            elif user_input.lower() == 'export':
                chatbot.save_chat_history()
                continue
            
            elif user_input.lower() == 'export-info':
                if not chatbot.current_user:
                    print("⚠️ 먼저 로그인해주세요.")
                else:
                    export_info = chatbot.get_export_file_info()
                    if 'error' in export_info:
                        print(f"⚠️ {export_info['error']}")
                    elif export_info['exists']:
                        print(f"\n📄 {chatbot.current_user}님의 export 파일 정보:")
                        print(f"  - 파일명: {export_info['filename']}")
                        print(f"  - 마지막 업데이트: {export_info['last_updated'][:19]}")
                        print(f"  - 대화 수: {export_info['total_conversations']}개")
                        print(f"  - 파일 크기: {export_info['file_size']:,} bytes")
                    else:
                        print(f"📄 {export_info['filename']} 파일이 존재하지 않습니다.")
                        print("   'export' 명령어로 파일을 생성할 수 있습니다.")
                continue
            
            elif user_input.lower().startswith('delete '):
                username_to_delete = user_input[7:].strip()
                if username_to_delete:
                    confirm = input(f"⚠️ {username_to_delete}님의 모든 데이터를 삭제하시겠습니까? (y/N): ")
                    if confirm.lower() == 'y':
                        chatbot.delete_user_memory(username_to_delete)
                    else:
                        print("삭제가 취소되었습니다.")
                else:
                    print("삭제할 사용자 이름을 입력해주세요. 예: delete 홍길동")
                continue
            
            # 일반 채팅 처리
            if not chatbot.current_user:
                # 로그인이 안된 경우, 자동으로 사용자 이름으로 간주하고 로그인 시도
                print("먼저 로그인을 진행하겠습니다.")
                result = chatbot.login_user(user_input)
                print(f"{result['message']}")
                if result['success'] and not result['is_new_user']:
                    print(f"저장된 대화 수: {result['conversation_count']}개")
                continue
            
            # 챗봇 응답 생성
            response = chatbot.chat(user_input)
            print(f"\n🤖 챗봇: {response}")
            
        except KeyboardInterrupt:
            print("\n\n프로그램을 종료합니다.")
            if chatbot.current_user:
                chatbot.logout_user()
            break
        except Exception as e:
            print(f"오류 발생: {e}")


if __name__ == "__main__":
    main()