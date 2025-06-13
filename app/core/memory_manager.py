import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class MemoryManager:
    def __init__(self, memory_dir: str, llm: ChatOpenAI):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir = Path("chat_history")
        self.export_dir.mkdir(exist_ok=True)
        self.llm = llm
        self.max_conversation_length = 10
        self.summary_threshold = 8
    
    def get_memory_file_path(self, username: str) -> Optional[Path]:
        """사용자명을 안전한 파일명으로 변환하여 경로 반환. 유효하지 않으면 None 반환"""
        if not username or not username.strip():
            return None
            
        safe_username = "".join(c for c in username if c.isalnum() or c in ('-', '_')).strip()
        if not safe_username:
            return None
            
        return self.memory_dir / f"{safe_username}_memory.json"
    
    def user_exists(self, username: str) -> bool:
        """사용자 메모리 파일이 실제로 존재하는지 확인"""
        memory_file = self.get_memory_file_path(username)
        return memory_file is not None and memory_file.exists()
    
    def _repair_json_file(self, file_path: Path) -> bool:
        """손상된 JSON 파일 복구 시도"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 백업 생성
            backup_path = file_path.with_suffix('.json.backup')
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"손상된 파일 백업 생성: {backup_path}")
            
            # 일반적인 JSON 오류 수정 시도
            content = content.strip()
            
            # 후행 쉼표 제거
            content = content.replace(',\n}', '\n}').replace(',\n]', '\n]')
            content = content.replace(', }', ' }').replace(', ]', ' ]')
            
            # 불완전한 JSON 완성
            if not content.endswith('}'):
                if content.endswith(','):
                    content = content[:-1] + '\n}'
                else:
                    content += '\n}'
            
            # 복구된 내용으로 파일 덮어쓰기
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # 복구 확인
            json.loads(content)
            print(f"JSON 파일 복구 성공: {file_path}")
            return True
            
        except Exception as e:
            print(f"JSON 파일 복구 실패: {e}")
            return False
    
    def _create_default_user_data(self, username: str) -> Dict[str, Any]:
        """기본 사용자 데이터 구조 생성"""
        return {
            'username': username,
            'chat_history': [],
            'conversation_summary': "",
            'last_login': datetime.now().isoformat(),
            'total_conversations': 0,
            'created_at': datetime.now().isoformat()
        }
    
    def load_user_memory(self, username: str) -> Tuple[List[Dict[str, Any]], str, bool]:
        # 유효하지 않은 사용자명 처리
        memory_file = self.get_memory_file_path(username)
        if memory_file is None:
            print(f"유효하지 않은 사용자명: '{username}'")
            return [], "", False
        
        if not memory_file.exists():
            print(f"신규 사용자 {username}님, 처음 뵙겠습니다!")
            return [], "", False
        
        # 여러 시도로 파일 로드
        for attempt in range(3):
            try:
                with open(memory_file, 'r', encoding='utf-8') as f:
                    user_data = json.load(f)
                
                # 데이터 검증
                if not isinstance(user_data, dict):
                    raise ValueError("유효하지 않은 데이터 형식")
                
                chat_history = user_data.get('chat_history', [])
                conversation_summary = user_data.get('conversation_summary', "")
                last_login = user_data.get('last_login', 'Unknown')
                
                # 채팅 히스토리 타입 검증
                if not isinstance(chat_history, list):
                    chat_history = []
                
                print(f"{username}님의 기존 대화 기록 로드 완료")
                print(f"   - 저장된 대화 수: {len(chat_history)}개")
                print(f"   - 대화 요약 존재: {'예' if conversation_summary else '아니오'}")
                print(f"   - 마지막 접속: {last_login}")
                
                return chat_history, conversation_summary, True
                
            except json.JSONDecodeError as e:
                print(f"JSON 파싱 오류 (시도 {attempt + 1}/3): {e}")
                
                if attempt == 0:
                    # 첫 번째 시도: 파일 복구
                    if self._repair_json_file(memory_file):
                        continue
                elif attempt == 1:
                    # 두 번째 시도: 다른 인코딩으로 읽기
                    try:
                        with open(memory_file, 'r', encoding='utf-8-sig') as f:
                            user_data = json.load(f)
                        
                        chat_history = user_data.get('chat_history', [])
                        conversation_summary = user_data.get('conversation_summary', "")
                        
                        print(f"{username}님의 대화 기록 로드 완료 (인코딩 변경)")
                        return chat_history, conversation_summary, True
                        
                    except Exception:
                        continue
                else:
                    # 세 번째 시도 실패: 파일 재생성
                    print(f"파일 복구 불가능. 새로운 기록으로 시작합니다.")
                    self._backup_corrupted_file(memory_file)
                    return [], "", False
                    
            except Exception as e:
                print(f"메모리 파일 로드 중 일반 오류: {e}")
                if attempt == 2:
                    self._backup_corrupted_file(memory_file)
                    return [], "", False
                continue
        
        return [], "", False
    
    def _backup_corrupted_file(self, memory_file: Path):
        """손상된 파일 백업 후 제거"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{memory_file.stem}_corrupted_{timestamp}.json"
            backup_path = memory_file.parent / backup_name
            
            memory_file.rename(backup_path)
            print(f"손상된 파일 백업: {backup_path}")
            
        except Exception as e:
            print(f"손상된 파일 백업 실패: {e}")
            try:
                memory_file.unlink()
                print("손상된 파일 삭제됨")
            except:
                pass
    
    def save_user_memory(self, username: str, chat_history: List[Dict[str, Any]], 
                        conversation_summary: str) -> bool:
        try:
            memory_file = self.get_memory_file_path(username)
            if memory_file is None:
                print(f"유효하지 않은 사용자명으로 저장 불가: '{username}'")
                return False
            
            # 기존 데이터 로드 (created_at 보존용)
            created_at = datetime.now().isoformat()
            if memory_file.exists():
                try:
                    with open(memory_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                        created_at = existing_data.get('created_at', created_at)
                except:
                    pass
            
            # 새 데이터 생성
            user_data = {
                'username': username,
                'chat_history': chat_history,
                'conversation_summary': conversation_summary,
                'last_login': datetime.now().isoformat(),
                'total_conversations': len(chat_history),
                'created_at': created_at
            }
            
            # 임시 파일에 먼저 저장
            temp_file = memory_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(user_data, f, ensure_ascii=False, indent=2)
            
            # JSON 유효성 검증
            with open(temp_file, 'r', encoding='utf-8') as f:
                json.load(f)
            
            # 검증 성공 시 원본 파일로 이동
            temp_file.replace(memory_file)
            return True
            
        except Exception as e:
            print(f"메모리 저장 중 오류: {e}")
            
            # 임시 파일 정리
            temp_file = self.get_memory_file_path(username)
            if temp_file is not None:
                temp_file = temp_file.with_suffix('.tmp')
                if temp_file.exists():
                    temp_file.unlink()
            
            return False
    
    def summarize_old_conversations(self, username: str, chat_history: List[Dict[str, Any]], 
                                  conversation_summary: str) -> Tuple[List[Dict[str, Any]], str]:
        if len(chat_history) <= self.summary_threshold:
            return chat_history, conversation_summary
        
        try:
            conversations_to_summarize = chat_history[:-self.summary_threshold]
            
            conversation_text = ""
            for conv in conversations_to_summarize:
                if isinstance(conv, dict) and 'human' in conv and 'ai' in conv:
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
            print(f"대화 요약 중 오류: {e}")
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
                if isinstance(entry, dict) and 'human' in entry and 'ai' in entry:
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
            if memory_file is None:
                print(f"유효하지 않은 사용자명: '{username}'")
                return False
                
            if memory_file.exists():
                # 삭제 전 백업
                backup_path = memory_file.with_suffix('.deleted_backup')
                memory_file.rename(backup_path)
                print(f"{username}님의 대화 기록이 삭제되었습니다. (백업: {backup_path})")
                return True
            else:
                print(f"{username}님의 대화 기록을 찾을 수 없습니다.")
                return False
        except Exception as e:
            print(f"메모리 삭제 중 오류: {e}")
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
            print(f"대화 기록 저장 중 오류: {e}")
    
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
        if memory_file is None:
            return {"error": f"유효하지 않은 사용자명: '{username}'"}
        
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
            return {"error": f"통계 정보 조회 중 오류: {e}"}
    
    def repair_all_memory_files(self) -> Dict[str, Any]:
        """모든 메모리 파일 일괄 복구"""
        results = {
            'total_files': 0,
            'repaired_files': 0,
            'failed_files': 0,
            'backup_created': 0,
            'details': []
        }
        
        for memory_file in self.memory_dir.glob("*_memory.json"):
            results['total_files'] += 1
            
            try:
                with open(memory_file, 'r', encoding='utf-8') as f:
                    json.load(f)
                results['details'].append(f"정상: {memory_file.name}")
                
            except json.JSONDecodeError:
                if self._repair_json_file(memory_file):
                    results['repaired_files'] += 1
                    results['details'].append(f"복구 성공: {memory_file.name}")
                else:
                    self._backup_corrupted_file(memory_file)
                    results['failed_files'] += 1
                    results['backup_created'] += 1
                    results['details'].append(f"복구 실패 (백업 생성): {memory_file.name}")
            
            except Exception as e:
                results['failed_files'] += 1
                results['details'].append(f"오류: {memory_file.name} - {e}")
        
        return results