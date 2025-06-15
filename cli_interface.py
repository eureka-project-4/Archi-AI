import os
from main_chatbot import PricingPlanChatbot
from dotenv import load_dotenv

class CLIInterface:
    def __init__(self):
        load_dotenv(override=True)
        
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.pricing_data_file = os.getenv("PRICING_DATA_FILE")
        
        if not self.openai_api_key:
            print("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
            print("   .env 파일에 OPENAI_API_KEY=your-api-key 를 추가해주세요.")
            return
        
        print(f"OpenAI API Key 로드됨: {self.openai_api_key[:8]}...{self.openai_api_key[-4:]}")
        
        try:
            self.chatbot = PricingPlanChatbot(self.openai_api_key, self.pricing_data_file)
        except Exception as e:
            print(f"챗봇 초기화 실패: {e}")
            return
        
        self.show_welcome_message()
    
    def show_welcome_message(self):
        print("=== 할루시네이션 검증 기능이 포함된 요금제 추천 챗봇 ===")
        print("새로운 검증 기능:")
        print("  - 요금제 존재 여부 자동 검증")
        print("  - 정보 정확성 신뢰도 측정")
        print("  - 퍼지 매칭을 통한 유사 요금제 제안")
        print("  - 상세 검증 보고서 생성")
        print("")
        print("기본 명령어:")
        print("  - 'users': 저장된 사용자 목록 보기")
        print("  - 'login [이름]': 사용자 로그인")
        print("  - 'logout': 현재 사용자 로그아웃")
        print("  - 'stats': 현재 사용자 통계")
        print("  - 'summary': 대화 요약 보기")
        print("  - 'clear': 대화 기록 초기화")
        print("  - 'msg-stats': 메시지 타입별 통계")
        print("  - 'export': 대화 기록 내보내기")
        print("  - 'export-info': export 파일 정보 확인")
        print("  - 'delete [이름]': 사용자 데이터 삭제")
        print("")
        print("검증 명령어:")
        print("  - 'verify': 요금제 데이터베이스 정보")
        print("  - 'check [요금제명]': 특정 요금제 존재 확인")
        print("  - 'verified [질문]': 검증된 응답으로 대화")
        print("  - 'report [질문]': 상세 검증 보고서 생성")
        print("")
        print("  - 'quit' 또는 'exit': 종료")
        print("-" * 60)
    
    def handle_command(self, user_input: str) -> bool:
        if user_input.lower() in ['quit', 'exit', '종료']:
            if self.chatbot.current_user:
                self.chatbot.logout_user()
            print("챗봇을 종료합니다. 안녕히 가세요!")
            return True
        
        elif user_input.lower() == 'users':
            self.show_users()
        
        elif user_input.lower().startswith('login '):
            username = user_input[6:].strip()
            self.login_user(username)
        
        elif user_input.lower() == 'logout':
            self.chatbot.logout_user()
            
        elif user_input.lower() == 'msg-stats':
            self.show_message_statistics()
            
        elif user_input.lower().startswith('delete '):
            username = user_input[7:].strip()
            self.delete_user(username)
        
        elif user_input.lower() == 'stats':
            self.show_stats()
        
        elif user_input.lower() == 'summary':
            self.show_summary()
        
        elif user_input.lower() == 'export-info':
            self.show_export_info()
        
        elif user_input.lower() == 'clear':
            self.chatbot.clear_chat_history()
        
        elif user_input.lower() == 'export':
            self.chatbot.save_chat_history()
        
        elif user_input.lower() == 'verify':
            self.show_verification_info()
        
        elif user_input.lower().startswith('check '):
            plan_name = user_input[6:].strip()
            self.check_plan(plan_name)
        
        elif user_input.lower().startswith('verified '):
            question = user_input[9:].strip()
            self.verified_chat(question)
        
        elif user_input.lower().startswith('report '):
            question = user_input[7:].strip()
            self.generate_report(question)
        
        else:
            self.handle_chat(user_input)
        
        return False
    
    def show_users(self):
        users = self.chatbot.list_all_users()
        if users:
            print("\n저장된 사용자 목록:")
            for i, user in enumerate(users, 1):
                print(f"  {i}. {user['username']} "
                      f"(대화 {user['total_conversations']}개, "
                      f"최근 접속: {user['last_login'][:19]})")
        else:
            print("저장된 사용자가 없습니다.")
    
    def login_user(self, username: str):
        if username:
            result = self.chatbot.login_user(username)
            print(f"\n{result['message']}")
            if result['success'] and not result['is_new_user']:
                print(f"저장된 대화 수: {result['conversation_count']}개")
        else:
            print("사용자 이름을 입력해주세요. 예: login 홍길동")
    
    def delete_user(self, username: str):
        if username:
            confirm = input(f"{username}님의 모든 데이터를 삭제하시겠습니까? (y/N): ")
            if confirm.lower() == 'y':
                self.chatbot.delete_user_memory(username)
            else:
                print("삭제가 취소되었습니다.")
        else:
            print("삭제할 사용자 이름을 입력해주세요. 예: delete 홍길동")
    
    def show_stats(self):
        if not self.chatbot.current_user:
            print("먼저 로그인해주세요.")
            return
        
        stats = self.chatbot.get_user_statistics()
        if 'error' in stats:
            print(f"{stats['error']}")
        else:
            print(f"\n{stats['username']}님의 통계:")
            print(f"  - 총 대화 수: {stats['total_conversations']}개")
            print(f"  - 대화 요약 존재: {'예' if stats['has_summary'] else '아니오'}")
            if stats['has_summary']:
                print(f"  - 요약 길이: {stats['summary_length']}자")
            print(f"  - 첫 방문: {stats['first_visit'][:19]}")
            print(f"  - 마지막 접속: {stats['last_login'][:19]}")
    def show_message_statistics(self):
        """메시지 타입별 통계 표시"""
        if not self.chatbot.current_user:
            print("먼저 로그인해주세요.")
            return
        
        stats = self.chatbot.get_message_statistics()
        if 'error' in stats:
            print(f"{stats['error']}")
        else:
            print(f"\n{stats['username']}님의 메시지 통계:")
            print(f"  - 총 메시지 수: {stats['total_messages']}개")
            print(f"  - 일반 대화: {stats['chat_messages']}개")
            print(f"  - 요금제 추천: {stats['suggestion_messages']}개")
            print(f"  - 추천 비율: {stats['suggestion_ratio']:.1%}")
            if stats['average_confidence'] > 0:
                print(f"  - 평균 추천 신뢰도: {stats['average_confidence']:.1%}")
    def show_summary(self):
        if not self.chatbot.current_user:
            print("먼저 로그인해주세요.")
        elif self.chatbot.conversation_summary:
            print(f"\n{self.chatbot.current_user}님의 대화 요약:")
            print("-" * 40)
            print(self.chatbot.conversation_summary)
            print("-" * 40)
        else:
            print("저장된 대화 요약이 없습니다.")
    
    def show_export_info(self):
        if not self.chatbot.current_user:
            print("먼저 로그인해주세요.")
            return
        
        export_info = self.chatbot.get_export_file_info()
        if 'error' in export_info:
            print(f"{export_info['error']}")
        elif export_info['exists']:
            print(f"\n{self.chatbot.current_user}님의 export 파일 정보:")
            print(f"  - 파일명: {export_info['filename']}")
            print(f"  - 마지막 업데이트: {export_info['last_updated'][:19]}")
            print(f"  - 대화 수: {export_info['total_conversations']}개")
            print(f"  - 파일 크기: {export_info['file_size']:,} bytes")
        else:
            print(f"{export_info['filename']} 파일이 존재하지 않습니다.")
            print("   'export' 명령어로 파일을 생성할 수 있습니다.")
    
    def show_verification_info(self):
        if not self.chatbot.current_user:
            print("먼저 로그인해주세요.")
            return
        
        db_info = self.chatbot.get_plan_database_info()
        print(f"\n요금제 데이터베이스 정보:")
        print(f"  - 총 요금제 수: {db_info['total_plans']}개")
        print(f"  - 데이터베이스 항목: {db_info['total_entries']}개")
        print(f"  - 샘플 요금제: {', '.join(db_info['sample_plans'])}")
    
    def check_plan(self, plan_name: str):
        if not plan_name or not self.chatbot.current_user:
            print("사용법: check [요금제명]")
            return
        
        print(f"'{plan_name}' 요금제 검증 중...")
        check_result = self.chatbot.check_plan_hallucination(plan_name)
        
        print(f"\n검증 결과:")
        print(f"  - 요금제 존재: {'예' if check_result.plan_exists else '아니오'}")
        print(f"  - 신뢰도: {check_result.confidence_score:.1%}")
        
        if check_result.matched_plan:
            print(f"  - 가장 유사한 요금제: {check_result.matched_plan}")
        
        if check_result.evidence:
            print("  - 검증 근거:")
            for evidence in check_result.evidence:
                print(f"    • {evidence}")
        
        if check_result.discrepancies:
            print("  - 발견된 문제:")
            for disc in check_result.discrepancies:
                print(f"    • {disc}")
    
    def verified_chat(self, question: str):
        if not question or not self.chatbot.current_user:
            print("사용법: verified [질문]")
            return
        
        print("🔍 검증된 응답을 생성하고 있습니다...")
        response = self.chatbot.chat_with_verification(question)
        print(f"\n챗봇: {response}")
    
    def generate_report(self, question: str):
        if not question or not self.chatbot.current_user:
            print("사용법: report [질문]")
            return
        
        print("검증 보고서를 생성하고 있습니다...")
        report = self.chatbot.generate_verification_report(question)
        print(f"\n{report}")
    
    def handle_chat(self, user_input: str):
        if not self.chatbot.current_user:
            print("먼저 로그인을 진행하겠습니다.")
            result = self.chatbot.login_user(user_input)
            print(f"{result['message']}")
            if result['success'] and not result['is_new_user']:
                print(f"저장된 대화 수: {result['conversation_count']}개")
            return
        
        response = self.chatbot.chat(user_input)
        print(f"\n챗봇: {response}")
        
        if len(self.chatbot.chat_history) % 5 == 0 and self.chatbot.verification_system.plan_database:
            db_info = self.chatbot.get_plan_database_info()
            print(f"\n검증 힌트: 현재 {db_info['total_plans']}개의 검증된 요금제를 기반으로 응답합니다.")
            print("   더 정확한 정보를 원하시면 'verified [질문]' 또는 'report [질문]' 명령어를 사용해보세요.")
    
    def get_user_prompt(self) -> str:
        if self.chatbot.current_user:
            return f"\n[{self.chatbot.current_user}] 입력: "
        else:
            return "\n[로그인 필요] 입력: "
    
    def run(self):
        if not hasattr(self, 'chatbot'):
            return
        
        while True:
            try:
                user_input = input(self.get_user_prompt()).strip()
                
                if not user_input:
                    continue
                
                if self.handle_command(user_input):
                    break
                    
            except KeyboardInterrupt:
                print("\n\n프로그램을 종료합니다.")
                if self.chatbot.current_user:
                    self.chatbot.logout_user()
                break
            except Exception as e:
                print(f"오류 발생: {e}")


def main():
    cli = CLIInterface()
    cli.run()


if __name__ == "__main__":
    main()