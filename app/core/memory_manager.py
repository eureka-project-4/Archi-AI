import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# from .data_models import ChatEntry, UserMemory

class MemoryManager:
    def __init__(self, memory_dir: str, llm: ChatOpenAI):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir = Path("chat_history")
        self.export_dir.mkdir(exist_ok=True)
        self.llm = llm
        self.max_conversation_length = 10
        self.summary_threshold = 8
    
    def get_memory_file_path(self, username: str) -> Path:
        safe_username = "".join(c for c in username if c.isalnum() or c in ('-', '_')).strip()
        if not safe_username:
            safe_username = "unknown_user"
        return self.memory_dir / f"{safe_username}_memory.json"
    
    def load_user_memory(self, username: str) -> tuple[List[Dict[str, Any]], str, bool]:
        memory_file = self.get_memory_file_path(username)
        
        if memory_file.exists():
            try:
                with open(memory_file, 'r', encoding='utf-8') as f:
                    user_data = json.load(f)
                
                chat_history = user_data.get('chat_history', [])
                conversation_summary = user_data.get('conversation_summary', "")
                last_login = user_data.get('last_login', 'Unknown')
                
                print(f"{username}님의 기존 대화 기록을 불러왔습니다.")
                print(f"   - 저장된 대화 수: {len(chat_history)}개")
                print(f"   - 대화 요약 존재: {'예' if conversation_summary else '아니오'}")
                print(f"   - 마지막 접속: {last_login}")
                
                return chat_history, conversation_summary, True
                
            except Exception as e:
                print(f"메모리 파일 로드 중 오류 발생: {e}")
                return [], "", False
        else:
            print(f"👋 {username}님, 처음 뵙겠습니다! 새로운 대화를 시작하겠습니다.")
            return [], "", False
    
    def save_user_memory(self, username: str, chat_history: List[Dict[str, Any]], 
                        conversation_summary: str) -> bool:
        try:
            memory_file = self.get_memory_file_path(username)
            
            user_data = {
                'username': username,
                'chat_history': chat_history,
                'conversation_summary': conversation_summary,
                'last_login': datetime.now().isoformat(),
                'total_conversations': len(chat_history),
                'created_at': datetime.now().isoformat() if not memory_file.exists() else None
            }
            
            if memory_file.exists():
                with open(memory_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    user_data['created_at'] = existing_data.get('created_at', datetime.now().isoformat())
            
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(user_data, f, ensure_ascii=False, indent=2)
            
            return True
            
        except Exception as e:
            print(f"메모리 저장 중 오류 발생: {e}")
            return False
    
    def summarize_old_conversations(self, username: str, chat_history: List[Dict[str, Any]], 
                                  conversation_summary: str) -> tuple[List[Dict[str, Any]], str]:
        if len(chat_history) <= self.summary_threshold:
            return chat_history, conversation_summary
        
        try:
            conversations_to_summarize = chat_history[:-self.summary_threshold]
            
            conversation_text = ""
            for conv in conversations_to_summarize:
                conversation_text += f"사용자: {conv['human']}\n"
                conversation_text += f"챗봇: {conv['ai']}\n\n"
            
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
            
            if conversation_summary:
                conversation_text = f"[이전 대화 요약]\n{conversation_summary}\n\n[새로운 대화]\n{conversation_text}"
            
            summary_chain = summary_prompt | self.llm | StrOutputParser()
            new_summary = summary_chain.invoke({
                "username": username,
                "conversation_text": conversation_text
            })
            
            new_chat_history = chat_history[-self.summary_threshold:]
            
            print(f"대화 요약 완료: {len(conversations_to_summarize)}개 대화가 요약되었습니다.")
            
            return new_chat_history, new_summary
            
        except Exception as e:
            print(f"대화 요약 중 오류 발생: {e}")
            return chat_history, conversation_summary
    
    def format_chat_history(self, chat_history: List[Dict[str, Any]], 
                           conversation_summary: str) -> str:
        formatted_parts = []
        
        if conversation_summary:
            formatted_parts.append("[이전 대화 요약]")
            formatted_parts.append(conversation_summary)
            formatted_parts.append("\n[최근 대화]")
        
        if not chat_history:
            if not conversation_summary:
                formatted_parts.append("이전 대화 내용이 없습니다.")
        else:
            for entry in chat_history:
                formatted_parts.append(f"사용자: {entry['human']}")
                formatted_parts.append(f"챗봇: {entry['ai']}")
        
        return "\n".join(formatted_parts)
    
    def list_all_users(self) -> List[Dict[str, Any]]:
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
        try:
            memory_file = self.get_memory_file_path(username)
            if memory_file.exists():
                memory_file.unlink()
                print(f"{username}님의 대화 기록이 삭제되었습니다.")
                return True
            else:
                print(f"{username}님의 대화 기록을 찾을 수 없습니다.")
                return False
        except Exception as e:
            print(f"메모리 삭제 중 오류 발생: {e}")
            return False
    
    def save_chat_history_export(self, username: str, chat_history: List[Dict[str, Any]], 
                                conversation_summary: str, filename: str = None):
        if filename is None:
            safe_username = "".join(c for c in username if c.isalnum() or c in ('-', '_')).strip()
            filename = self.export_dir / f"{safe_username}_chat_export.json"
        
        try:
            export_data = {
                'username': username,
                'export_date': datetime.now().isoformat(),
                'total_conversations': len(chat_history),
                'conversation_summary': conversation_summary,
                'chat_history': chat_history,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            print(f"대화 기록이 {filename}에 저장되었습니다.")
        except Exception as e:
            print(f"대화 기록 저장 중 오류 발생: {e}")
    
    def get_export_file_info(self, username: str) -> Dict[str, Any]:
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
    
    def get_user_statistics(self, username: str, chat_history: List[Dict[str, Any]], 
                           conversation_summary: str) -> Dict[str, Any]:
        memory_file = self.get_memory_file_path(username)
        
        try:
            if memory_file.exists():
                with open(memory_file, 'r', encoding='utf-8') as f:
                    user_data = json.load(f)
                
                return {
                    'username': username,
                    'total_conversations': len(chat_history),
                    'has_summary': bool(conversation_summary),
                    'summary_length': len(conversation_summary) if conversation_summary else 0,
                    'first_visit': user_data.get('created_at', 'Unknown'),
                    'last_login': user_data.get('last_login', 'Unknown'),
                    'current_session_messages': len(chat_history)
                }
            else:
                return {
                    'username': username,
                    'total_conversations': len(chat_history),
                    'has_summary': bool(conversation_summary),
                    'summary_length': len(conversation_summary) if conversation_summary else 0,
                    'first_visit': 'Current session',
                    'last_login': 'Current session',
                    'current_session_messages': len(chat_history)
                }
        except Exception as e:
            return {"error": f"통계 정보 조회 중 오류 발생: {e}"}