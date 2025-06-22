import os
import re
import json
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
from langchain_core.documents import Document
from app.models.message import MessageType
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
    
    def initialize(self, force_rebuild=True):
        try:
            vector_path = Path(settings.VECTOR_STORE_DIR) / "faiss_index"
            if vector_path.exists() and not force_rebuild:
                self.vectorstore = FAISS.load_local(
                    str(vector_path), 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print("기존 벡터스토어 로드됨")
            else:
                if force_rebuild:
                    print("강제 재생성 모드: 벡터스토어 새로 생성")
                self._create_vectorstore_from_files()
            
            if self.vectorstore:
                self._setup_chain()
                print("RAG 체인 설정 완료")
                self.debug_vectorstore_metadata()
                
        except Exception as e:
            print(f"RAG 시스템 초기화 오류: {e}")
            self.vectorstore = None
    
    def debug_vectorstore_metadata(self):
        if not self.vectorstore:
            print("DEBUG: 벡터스토어가 없습니다")
            return
            
        try:
            test_queries = ["요금제", "5G", "플랜", "쿠폰", "서비스", "데이터"]
            
            print(f"DEBUG: 벡터스토어 메타데이터 확인")
            
            for query in test_queries:
                test_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
                test_docs = test_retriever.invoke(query)
                
                print(f"DEBUG: '{query}' 검색 결과 - 문서 수: {len(test_docs)}")
                
                type_counts = {}
                for doc in test_docs:
                    doc_type = doc.metadata.get('type', 'missing')
                    type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
                
                print(f"DEBUG: '{query}' 타입별 분포: {type_counts}")
                
                if test_docs:
                    first_doc = test_docs[0]
                    print(f"DEBUG: '{query}' 첫 문서 - type: {first_doc.metadata.get('type')}, file: {first_doc.metadata.get('source_file')}")
                print()
            
        except Exception as e:
            print(f"DEBUG: 메타데이터 확인 실패: {e}")
    
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
                
                data_type = self._determine_data_type(file_path)
                
                for doc in documents:
                    doc.metadata['type'] = data_type
                    doc.metadata['source_file'] = file_path.name
                    doc.metadata['file_path'] = str(file_path)
                    print(f"DEBUG: 문서 메타데이터 설정 - type: {data_type}, file: {file_path.name}")
                
                all_documents.extend(documents)
                print(f"로드됨: {file_path.name} ({data_type}) - {len(documents)}개 문서")
                
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
            
            type_counts = {}
            for doc in splits:
                doc_type = doc.metadata.get('type', 'unknown')
                type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
            print(f"타입별 분포: {type_counts}")
    
    def _determine_data_type(self, file_path: Path) -> str:
        filename_lower = file_path.name.lower()
        
        print(f"DEBUG: 파일 타입 결정 중 - 파일명: {file_path.name}")
        
        if 'plan' in filename_lower:
            print(f"DEBUG: plan 타입으로 분류")
            return 'plan'
        elif 'vass' in filename_lower:
            print(f"DEBUG: vass 타입으로 분류")
            return 'vass'
        elif 'coupon' in filename_lower:
            print(f"DEBUG: coupon 타입으로 분류")
            return 'coupon'
        else:
            print(f"DEBUG: 기본값 plan 타입으로 분류")
            return 'plan'
    
    def _setup_chain(self):
        system_prompt = """
        You are an expert telecommunications service recommendation specialist. 
        Provide personalized recommendations based on user needs and usage patterns.
        
        **Current User: {user_id}**
        
        **Guidelines:**
        1. Engage naturally and friendly with users
        2. Understand user's needs and provide accurate recommendations
        3. Recommend services based on the provided data and context
        4. Remember previous conversations and maintain context
        5. Clearly explain recommendation reasons
        6. Only recommend items that actually exist in the provided data
        
        **Context Information:**
        {context}
        
        **Previous Conversation:**
        {chat_history}
        """
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        
        self.combine_docs_chain = create_stuff_documents_chain(self.llm, self.prompt)
        
        default_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.RETRIEVAL_K}
        )
        self.rag_chain = create_retrieval_chain(default_retriever, self.combine_docs_chain)
    
    def analyze_query_intent(self, user_input: str) -> Dict[str, Any]:
        try:
            intent_prompt = ChatPromptTemplate.from_template("""
            Analyze the user's question and classify what type of information they are seeking.
            
            Classification categories:
            1. "plan": Telecommunications plans/products only (5G, LTE, data, calls, monthly fees)
            2. "vass": Additional services only (roaming, security, quality improvement)
            3. "coupon": Coupons/benefits only (movie discounts, shopping, lifestyle benefits)
            4. "comprehensive": Comprehensive recommendations (plans + services + coupons together)
            5. "general": General inquiries (greetings, customer service, policies)
            Important rules:
            - Questions about PREVIOUS recommendations or asking for EXPLANATIONS about past responses should be classified as "general"
            - Only classify as "comprehensive" when user is ASKING FOR NEW recommendations that combine multiple categories
            - Keywords like "방금", "아까", "네가 말한", "이유", "왜" usually indicate "general" intent
            Examples:
            - "드라마 보면서 커피 마시는 걸 좋아해" → general (daily life)
            - "5G 요금제 추천해줘" → plan (telecom product)
            - "방금 추천한 이유가 뭐야?" → general (asking for explanation)
            - "요금제랑 쿠폰 다 추천해줘" → comprehensive (multiple telecom products)
            data_category:
            - "5G": 5G related plans or services
            - "LTE": LTE related plans or services
            Keyword hints:
        - plan: 요금제, 플랜, 데이터, 통화, 문자, 월요금, 5G, LTE, 무제한, 가격
        - vass: 부가서비스, 로밍, 보안, 추가, 프리미엄, 애드온
        - coupon: 쿠폰, 할인, 혜택, 리워드, 영화, 쇼핑, 라이프스타일
        - comprehensive: 종합, 전체, 모든, 패키지, 토탈, 완전한, 함께, 조합, 추천
        - general: 안녕, 문의, 고객센터, 정책, 약관, 도움
            
            User input: {user_input}
            
            Respond in JSON format only:
            {{
                "intent": "plan|vass|coupon|comprehensive|general",
                "confidence": 0.0-1.0,
                "reasoning": "brief explanation of classification decision",
                "data_category": 5G|LTE|None
            }}
            """)
            
            intent_chain = intent_prompt | self.analysis_llm | StrOutputParser()
            result = intent_chain.invoke({"user_input": user_input})
            
            if '```json' in result:
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', result, re.DOTALL)
                if json_match:
                    result = json_match.group(1)
            
            intent_data = json.loads(result)
            
            return {
                "intent": intent_data.get("intent", "general"),
                "confidence": intent_data.get("confidence", 0.5),
                "reasoning": intent_data.get("reasoning", ""),
                "data_category": intent_data.get("data_category", "5G"),
                "method": "llm_analysis"
            }
            
        except Exception as e:
            print(f"의도 분석 실패: {e}")
            return {
                "intent": "general",
                "confidence": 0.3,
                "reasoning": "Failed to analyze intent, defaulting to general",
                "method": "fallback"
            }
    
    def extract_price_conditions(self, text: str) -> Dict[str, Any]:
        try:
            price_prompt = ChatPromptTemplate.from_template("""
            Extract price conditions from the user input.
            
            Look for:
            - "X원 이하" (under X won)
            - "X원 이상" (over X won)  
            - "X원대" (around X won)
            - "저렴한" (cheap)
            - "비싼" (expensive)
            
            User input: {user_input}
            
            Respond in JSON format only:
            {{
                "has_price_condition": true/false,
                "price_max": number or null,
                "price_min": number or null,
                "price_around": number or null,
                "condition_type": "under|over|around|cheap|expensive|none"
            }}
            """)
            
            price_chain = price_prompt | self.analysis_llm | StrOutputParser()
            result = price_chain.invoke({"user_input": text})
            
            if '```json' in result:
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', result, re.DOTALL)
                if json_match:
                    result = json_match.group(1)
            
            return json.loads(result)
            
        except Exception as e:
            print(f"가격 조건 추출 실패: {e}")
            return {"has_price_condition": False}
    
    def search_plans_by_price(self, price_condition: Dict[str, Any], data_type: str = None, data_category: str = "5G") -> List[Dict[str, Any]]:
        if not self.csv_verifier or not price_condition.get("has_price_condition"):
            return []
        
        print(f"DEBUG: 가격 검색 - 데이터 타입 필터: {data_type}")
        
        criteria = {}
        
        if price_condition.get("price_max"):
            criteria["price_range"] = (0, price_condition["price_max"])
        elif price_condition.get("price_min"):
            criteria["price_range"] = (price_condition["price_min"], 999999)
        elif price_condition.get("price_around"):
            around_price = price_condition["price_around"]
            criteria["price_range"] = (around_price - 10000, around_price + 10000)
        if data_category and data_type == "plan":
            if data_category.upper() == "5G":
                criteria["category_code"] = 1
            elif data_category.upper() == "LTE":
                criteria["category_code"] = 2
        all_results = self.csv_verifier.find_plans_by_criteria(**criteria)
        if price_condition.get("condition_type") == "expensive":
            all_results.sort(key=lambda x: x['price'], reverse=True) 
        elif price_condition.get("condition_type") == "cheap":
            all_results.sort(key=lambda x: x['price'])
        if data_type is None:
            print(f"DEBUG: 전체 {len(all_results)}개 상품 반환 (모든 타입)")
            return all_results
        
        filtered_results = []
        for plan in all_results:
            plan_type = plan.get('type', 'unknown')
            print(f"DEBUG: 상품 '{plan['name']}' - 타입: {plan_type}, 가격: {plan['price']}")
            if plan_type == data_type:
                filtered_results.append(plan)
        
        print(f"DEBUG: 전체 {len(all_results)}개 중 {data_type} 타입 {len(filtered_results)}개 필터링됨")
        return filtered_results
    
    def extract_plan_names_from_input(self, text: str) -> List[str]:
        try:
            extraction_prompt = ChatPromptTemplate.from_template("""
            Extract actual product names or plan names from the user input.
            
            DO NOT extract:
            - Price conditions (e.g., "under 50,000 won", "cheap")
            - Generic terms with conditions (e.g., "which plan", "cheap plan", "plan under 30,000 won")
            - General recommendation requests (e.g., "recommend", "available", "tell me", "inquiry")
            
            ONLY extract:
            - Specific product names or service names (noun form only)
            - Examples: "5G 프리미어 에센셜", "T플랜 스페셜", "LTE 베이직", "지니뮤직 마음껏듣기 월정액" , "배스킨라빈스 파인트 4천원 할인쿠폰"
            
            User input: {user_input}
            
            Respond in JSON format only:
            {{
                "extracted_names": ["list of specific product names"],
                "confidence": 0.0-1.0
            }}
            """)
            
            extraction_chain = extraction_prompt | self.analysis_llm | StrOutputParser()
            result = extraction_chain.invoke({"user_input": text})
            
            if '```json' in result:
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', result, re.DOTALL)
                if json_match:
                    result = json_match.group(1)
            
            extracted_data = json.loads(result)
            return extracted_data.get("extracted_names", [])
            
        except Exception as e:
            print(f"요금제명 추출 실패: {e}")
            return []
    
    def get_search_filter(self, intent: str) -> Optional[Dict[str, str]]:
        filter_map = {
            "plan": {"type": "plan"},
            "vass": {"type": "vass"}, 
            "coupon": {"type": "coupon"},
            "comprehensive": None,
            "general": None
        }
        return filter_map.get(intent, None)
    
    def get_filtered_retriever(self, intent: str):
        if not self.vectorstore:
            return None
            
        search_filter = self.get_search_filter(intent)
        
        search_k = settings.RETRIEVAL_K * 3
        if intent == "plan" or "vass" or "coupon":
            search_k = max(15, search_k)
        elif intent == "comprehensive":
            search_k = max(20, search_k)
        
        print(f"DEBUG: get_filtered_retriever 호출됨 - intent: {intent}, filter: {search_filter}, k: {search_k}")
        
        if search_filter:
            try:
                retriever = self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={
                        "k": search_k,
                        "filter": search_filter
                    }
                )
                
                test_docs = retriever.invoke("테스트")
                print(f"DEBUG: 필터 적용 후 검색된 문서 수: {len(test_docs)}")
                if test_docs:
                    print(f"DEBUG: 첫 번째 문서 타입: {test_docs[0].metadata.get('type', 'unknown')}")
                
                return retriever
                
            except Exception as e:
                print(f"DEBUG: 필터링 실패, 기본 검색 사용: {e}")
                return self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": search_k}
                )
        else:
            return self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": search_k}
            )
    
    def load_user_context(self, user_id: str) -> tuple:
        user_id = str(user_id)
        chat_history, conversation_summary, is_existing_user = self.memory_manager.load_user_memory(user_id)
        return chat_history, conversation_summary, is_existing_user
    
    def chat(self, user_id: str, message: str) -> Dict[str, Any]:
        user_id = str(user_id)
        return self.chat_with_verification(user_id, message)
    
    def chat_with_verification(self, user_id: str, message: str) -> Dict[str, Any]:
        try:
            user_id = str(user_id)
            intent_analysis = self.analyze_query_intent(message)
            intent = intent_analysis["intent"]
            data_category = intent_analysis.get("data_category")
            
            print(f"DEBUG: 질문: {message}")
            print(f"DEBUG: 분석된 의도: {intent} (신뢰도: {intent_analysis['confidence']})")
            print(f"DEBUG: 분석 근거: {intent_analysis['reasoning']}")
            
            asked_plans = self.extract_plan_names_from_input(message)
            price_conditions = self.extract_price_conditions(message)
            
            print(f"DEBUG: 추출된 요금제명: {asked_plans}")
            print(f"DEBUG: 추출된 가격 조건: {price_conditions}")
            
            chat_history, conversation_summary, _ = self.memory_manager.load_user_memory(user_id)
            chat_history_str = self.memory_manager.format_chat_history(chat_history, conversation_summary)
            
            # 변수 초기화
            ai_response = ""
            
            # 가격 조건이 있으면 CSV에서 직접 검색
            if price_conditions.get("has_price_condition") and self.csv_verifier:
                print(f"DEBUG: 가격 조건 기반 검색 실행")
                
                if intent == "plan":
                    data_type_filter = "plan"
                    search_description = "통신 요금제"
                elif intent == "vass":
                    data_type_filter = "vass"
                    search_description = "부가서비스"
                elif intent == "coupon":
                    data_type_filter = "coupon"
                    search_description = "쿠폰/혜택"
                elif intent == "comprehensive":
                    data_type_filter = None
                    search_description = "모든 상품"
                else:
                    data_type_filter = "plan"
                    search_description = "통신 요금제"
                
                print(f"DEBUG: 검색 대상: {search_description} (타입: {data_type_filter})")
                
                if data_type_filter:
                    matching_plans = self.search_plans_by_price(price_conditions, data_type_filter, data_category)
                else:
                    matching_plans = self.search_plans_by_price(price_conditions, None, data_category)
                
                if matching_plans:
                    print(f"DEBUG: 조건에 맞는 {search_description} {len(matching_plans)}개 발견")
                    requested_count = 1 
                    if any(word in message.lower() for word in ['여러', '몇개', '비교', '옵션들', '선택지', '여러개']):
                        requested_count = 3
                    elif any(word in message.lower() for word in ['하나', '한개', '1개', '단일']):
                        requested_count = 1
                    print("*"*40)
                    print(requested_count)
                    top_plans = matching_plans[:requested_count]
                    
                    condition_text = ""
                    if price_conditions.get("price_max"):
                        condition_text = f"월 {price_conditions['price_max']:,}원 이하"
                    elif price_conditions.get("price_around"):
                        condition_text = f"월 {price_conditions['price_around']:,}원대"
                    
                    ai_response = f"{condition_text} 조건에 맞는 {search_description}를 추천해드리겠습니다.\n\n"
                    
                    for i, plan in enumerate(top_plans, 1):
                        ai_response += f"**{i}. {plan['name']}**\n"
                        ai_response += f"- 월 요금: {plan['price']:,}원\n"
                        
                        if data_type_filter == "plan":
                            ai_response += f"- 데이터: {plan['data']}\n"
                            ai_response += f"- 통화: {plan['calls']}\n"
                            ai_response += f"- 문자: {plan['sms']}\n"
                        
                        ai_response += f"- 혜택: {plan['benefit']}\n\n"
                    
                    ai_response += "더 자세한 정보나 다른 조건의 상품이 궁금하시면 언제든지 문의해주세요!"
                    
                    conversation_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "human": message,
                        "ai": ai_response,
                        "message_type": MessageType.SUGGESTION,  
                        "query_intent": intent
                    }
                    
                    chat_history.append(conversation_entry)
                    self.memory_manager.save_user_memory(user_id, chat_history, conversation_summary)
                    
                    return {
                        "response": ai_response,
                        "user_id": user_id,
                        "message_type": "SUGGESTION",
                        "confidence_score": 1.0,
                        "verification_status": f"정상 처리 - {search_description} 가격 조건 검색",
                        "query_intent": intent,
                        "price_condition_match": True,
                        "found_plans": len(matching_plans),
                        "search_type": data_type_filter or "all"
                    }
                else:
                    print(f"DEBUG: 조건에 맞는 {search_description}가 없음")
            
            # 특정 요금제명이 언급된 경우 검증
            if asked_plans and self.csv_verifier:
                for plan_name in asked_plans:
                    verification = self.csv_verifier.verify_plan_exists(plan_name)
                    
                    print(f"DEBUG: 요금제 '{plan_name}' 검증 결과:")
                    print(f"DEBUG: - 존재 여부: {verification['exists']}")
                    print(f"DEBUG: - 신뢰도: {verification['confidence']}")
                    print(f"DEBUG: - 매치 타입: {verification['match_type']}")
                    
                    if verification['confidence'] < 0.5:
                        print(f"DEBUG: 요금제 '{plan_name}' 차단됨 (신뢰도 {verification['confidence']})")
                        return {
                            "response": f"죄송합니다. '{plan_name}' 요금제는 현재 제공되지 않는 요금제입니다. 다른 요금제를 추천해드릴까요?",
                            "user_id": user_id,
                            "message_type": MessageType.BLOCKED_MESSAGE,
                            "mentioned_plans": [plan_name],
                            "confidence_score": verification['confidence'],
                            "verification_status": "차단됨 - 존재하지 않는 요금제",
                            "query_intent": intent
                        }
                    elif verification['exists'] and verification['confidence'] >= 0.8:
                        print(f"DEBUG: 요금제 '{plan_name}' 정확히 매칭됨 - CSV 직접 사용")
                        plan_info = verification['matched_plan']
                        # ai_response = f"""
                        #     {plan_name} 상품에 대해 안내해드리겠습니다.

                        #     **{plan_name}**
                        #     - 월 요금: {plan_info['price']:,}원
                        #     - 데이터: {plan_info['data']}
                        #     - 통화: {plan_info['calls']}
                        #     - 문자: {plan_info['sms']}
                        #     - 혜택: {plan_info['benefit']}

                        #     이 상품에 대해 더 궁금한 점이 있으시면 언제든지 문의해주세요.
                        # """
                        if len(plan_info['benefit']) > 3:
                            
                            ai_response = f"""
                                {plan_name} 상품에 대해 안내해드리겠습니다.

                                - 혜택: {plan_info['benefit']}

                                이 상품에 대해 더 궁금한 점이 있으시면 언제든지 문의해주세요.
                            """
                        else:
                            ai_response = f"""
                            {plan_name} 상품에 대해 안내해드리겠습니다.
                            
                            이 상품에 대해 더 궁금한 점이 있으시면 언제든지 문의해주세요.
                        """
                        
                        conversation_entry = {
                            "timestamp": datetime.now().isoformat(),
                            "human": message,
                            "ai": ai_response.strip(),
                            "message_type": "SUGGESTION",
                            "query_intent": intent
                        }
                        
                        chat_history.append(conversation_entry)
                        self.memory_manager.save_user_memory(user_id, chat_history, conversation_summary)
                        
                        return {
                            "response": ai_response.strip(),
                            "user_id": user_id,
                            "message_type": "SUGGESTION",
                            "confidence_score": 1.0,
                            "verification_status": "정상 처리 - CSV 직접 매칭",
                            "query_intent": intent,
                            "mentioned_plans": [plan_name],
                            "direct_csv_match": True
                        }
                    else:
                        print(f"DEBUG: 요금제 '{plan_name}' 통과됨 (신뢰도 {verification['confidence']})")
            else:
                print(f"DEBUG: 추출된 요금제명이 없거나 CSV 검증 시스템이 없음")
            # general intent는 RAG 검색 없이 바로 응답
            if intent == "general":
                print(f"DEBUG: General intent - RAG 검색 건너뜀")
                
                # 이전 대화 기록은 포함시켜야 함!
                general_prompt = ChatPromptTemplate.from_template("""
                당신은 친근한 대화 상대입니다. 사용자와 자연스럽게 대화하세요.
                이전 대화 내용을 기억하고 참고하여 응답하세요.
                
                이전 대화:
                {chat_history}
                
                사용자: {message}
                응답:
                """)
                
                general_chain = general_prompt | self.llm | StrOutputParser()
                ai_response = general_chain.invoke({
                    "message": message,
                    "chat_history": chat_history_str  # 이전 대화 포함!
                })
                
                # 대화 기록 저장
                conversation_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "human": message,
                    "ai": ai_response,
                    "message_type": "GENERAL_RESPONSE",
                    "query_intent": intent
                }
                
                chat_history.append(conversation_entry)
                self.memory_manager.save_user_memory(user_id, chat_history, conversation_summary)
                
                return {
                    "response": ai_response,
                    "user_id": user_id,
                    "message_type": "GENERAL_RESPONSE",
                    "confidence_score": 1.0,
                    "verification_status": "정상 처리",
                    "query_intent": intent
                }
            # RAG 체인으로 일반 응답 생성
            if self.rag_chain is None:
                return {
                    "response": "죄송합니다. AI 시스템이 초기화되지 않았습니다.",
                    "user_id": user_id,
                    "message_type": "error",
                    "confidence_score": 0.0
                }
            
            filtered_retriever = self.get_filtered_retriever(intent)
            search_filter = self.get_search_filter(intent)
            
            print(f"DEBUG: 적용된 검색 필터: {search_filter}")
            
            if filtered_retriever:
                temp_rag_chain = create_retrieval_chain(
                    filtered_retriever, 
                    self.combine_docs_chain
                )
                
                response = temp_rag_chain.invoke({
                    "input": message,
                    "chat_history": chat_history_str,
                    "user_id": user_id
                })
                
                ai_response = response.get("answer", str(response))
                used_sources = response.get("context", [])
                
                print(f"DEBUG: 검색된 소스 개수: {len(used_sources)}")
                # comprehensive intent processing
                # comprehensive intent processing
                if intent == 'comprehensive':
                    print("DEBUG: Comprehensive mode - executing each intent separately")
                    
                    # 각 카테고리별 결과를 저장할 딕셔너리
                    category_results = {
                        'plan': None,
                        'vass': None,
                        'coupon': None
                    }
                    
                    # 각 카테고리별로 독립적으로 처리
                    if intent == 'comprehensive':
                        print("DEBUG: Comprehensive mode - executing each intent separately")
                        
                        # 각 카테고리별 결과를 저장할 딕셔너리
                        category_results = {
                            'plan': None,
                            'vass': None,
                            'coupon': None
                        }
                        
                        # 각 카테고리별로 독립적으로 처리
                        for category in ['plan', 'vass', 'coupon']:
                            print(f"DEBUG: Processing {category} in comprehensive mode")
                            category_retriever = self.get_filtered_retriever(category)
                            if category_retriever:
                                temp_category_chain = create_retrieval_chain(
                                    category_retriever,
                                    self.combine_docs_chain
                                )
                                extracted_condition = "없음" 
                                try:
                                    category_condition_chain = self.get_category_condition_chain()
                                    extracted_condition = category_condition_chain.invoke({
                                        "user_message": message,
                                        "category": category
                                    }).strip()
                                except Exception as e:
                                    print(f"DEBUG: 조건 추출 실패: {e}")
                                    extracted_condition = "없음"
                                
                                if category == 'plan':
                                    search_message = "요금제 추천"
                                elif category == 'vass':
                                    search_message = "부가서비스 추천"
                                else:
                                    search_message = "쿠폰 추천"
                                
                                category_response = temp_category_chain.invoke({
                                    "input": search_message,
                                    "chat_history": chat_history_str,
                                    "user_id": user_id
                                })
                                
                                category_sources = category_response.get("context", [])
                                category_ai_response = ""  # 기본값
                                
                                print(f"DEBUG: {category} - 검색된 문서 수: {len(category_sources)}")
                                
                                if category_sources:
                                    filtered_context = "\n\n".join([doc.page_content for doc in category_sources])
                                    
                                    if category == "vass":
                                        print("DEBUG: VASS enhanced_prompt 생성 중")
                                        if extracted_condition == "없음" or not extracted_condition:
                                            category_message = f"부가서비스 하나만 추천해줘"
                                        else:
                                            category_message = f"{extracted_condition} 조건에 맞는 부가서비스 하나만 추천해줘"
                                        
                                        enhanced_prompt = f"""다음 검증된 데이터를 기반으로 부가서비스를 추천하세요.

                    검증된 부가서비스 데이터:
                    {filtered_context}

                    사용자 질문: {category_message}

                    지시사항:
                    1. 반드시 한국어로 답변하세요
                    2. 위 데이터에서 부가서비스 하나를 선택하여 추천하세요
                    3. 서비스명, 혜택을 포함하여 20자 이내로 설명하세요
                    4. 영어로 답변하지 마세요

                    추천:"""

                                    elif category == "coupon":
                                        print("DEBUG: COUPON enhanced_prompt 생성 중")
                                        print(f"DEBUG: Coupon context length: {len(filtered_context)}")
                                        print(f"DEBUG: Coupon context preview: {filtered_context[:200]}...")
                                        
                                        if extracted_condition == "없음" or not extracted_condition:
                                            category_message = f"쿠폰 하나만 추천해줘"
                                        else:
                                            category_message = f"{extracted_condition} 조건에 맞는 쿠폰 하나만 추천해줘"
                                    
                                        enhanced_prompt = f"""다음은 쿠폰 데이터입니다. CSV 형식으로 되어 있으며, 각 줄은 하나의 쿠폰을 나타냅니다.
                                            
                                            형식: ID,쿠폰명,설명,코드,카테고리,생성일,만료일

                                            쿠폰 데이터:
                                            {filtered_context}

                                            지시사항:
                                            1. 반드시 한국어로 답변하세요
                                            2. 위 데이터에서 쿠폰 하나를 선택하여 추천하세요
                                            3. 쿠폰명, 혜택을 포함하여 20자 이내로 설명하세요
                                            4. 영어로 답변하지 마세요
                                            사용자 질문: {category_message}

                                            """
                                    
                                    else:  # category == "plan"
                                        
                                        print("DEBUG: PLAN enhanced_prompt 생성 중")
                                        if extracted_condition == "없음" or not extracted_condition:
                                            category_message = f"요금제 하나만 추천해줘"
                                        else:
                                            category_message = f"{extracted_condition} 조건에 맞는 요금제 하나만 추천해줘"
                                        
                                        enhanced_prompt = f"""다음은 통신 요금제 데이터입니다. 각 요금제의 정보를 정확히 읽고 추천하세요.

                                    예시 데이터 형식:
                                    - 요금제명: 5G 프리미어 에센셜
                                    - 가격: 89000
                                    - 데이터: 200GB
                                    - 통화: 무제한
                                    - 문자: 무제한
                                    - 혜택: 넷플릭스 무료

                                    중요: 데이터가 "무제한"이라고 명시되지 않은 경우, 절대 무제한이라고 말하지 마세요.
                                    무제한은 -1이 무제한입니다.
                                    예: "95"는 95GB를 의미하며, 무제한이 아닙니다, "-1"은 데이터 무제한을 의미합니다.

                                    검증된 요금제 데이터:
                                    {filtered_context}

                                    사용자 질문: {category_message}

                                    지시사항:
                                    1. 데이터에 적힌 그대로만 설명하세요
                                    2. 숫자만 있으면 GB 단위입니다 (예: 95 → 95GB)
                                    3. "무제한"이라고 명시된 경우만 무제한으로 설명
                                    4. 가격은 원 단위로 표시 (예: 68000 → 68,000원)
                                    5. 혜택에 대한 부분은 설명하지 마세요.
                                    6. 설명 이유는 20자 이내로 설명하세요
                                    7. 한국어로 답변하세요

                                    추천:"""
                                    
                                    # 모든 카테고리에 대해 enhanced_prompt로 응답 생성
                                    filtered_response = self.combine_docs_chain.invoke({
                                        "input": enhanced_prompt,
                                        "chat_history": chat_history_str,
                                        "context": category_sources,
                                        "user_id": user_id
                                    })
                                    category_ai_response = str(filtered_response)
                                    print(f"DEBUG: {category} AI 응답: {category_ai_response}")
                                else:
                                    # 검색 결과가 없을 때 기본 메시지
                                    if category == 'plan':
                                        category_ai_response = "현재 추천 가능한 요금제를 찾을 수 없습니다."
                                    elif category == 'vass':
                                        category_ai_response = "현재 추천 가능한 부가서비스를 찾을 수 없습니다."
                                    else:
                                        category_ai_response = "현재 추천 가능한 쿠폰을 찾을 수 없습니다."
                                
                                # 결과 저장
                                category_results[category] = {
                                    'response': category_ai_response,
                                    'sources': category_sources
                                }
                        
                        # 가격 조건 처리
                        price_conditions = self.extract_price_conditions(message)
                        if price_conditions.get("has_price_condition") and self.csv_verifier:
                            print(f"DEBUG: Comprehensive mode with price conditions")
                            
                            for category in ['plan', 'vass', 'coupon']:
                                matching_items = self.search_plans_by_price(price_conditions, category, data_category)
                                if matching_items and matching_items[0]:
                                    item = matching_items[0]
                                    
                                    if category == 'plan':
                                        price_response = f"**{item['name']}**\n"
                                        price_response += f"- 월 요금: {item['price']:,}원\n"
                                        price_response += f"- 데이터: {item['data']}\n"
                                        price_response += f"- 통화: {item['calls']}\n"
                                        price_response += f"- 문자: {item['sms']}\n"
                                        price_response += f"- 혜택: {item['benefit']}"
                                    else:
                                        price_response = f"**{item['name']}**\n"
                                        price_response += f"- 가격: {item['price']:,}원\n"
                                        price_response += f"- 혜택: {item['benefit']}"
                                    
                                    # 가격 조건 결과가 있으면 우선 사용
                                    category_results[category] = {
                                        'response': price_response,
                                        'sources': []
                                    }
                        
                        # 최종 응답 생성
                        ai_response = ""
                        
                        # 요금제
                        ai_response += "##추천 통신 요금제\n"
                        if category_results['plan'] and category_results['plan']['response']:
                            ai_response += category_results['plan']['response'] + "\n\n"
                        else:
                            ai_response += "현재 추천 가능한 요금제를 찾을 수 없습니다.\n\n"
                        
                        # 부가서비스
                        ai_response += "##추천 부가서비스\n"
                        if category_results['vass'] and category_results['vass']['response']:
                            ai_response += category_results['vass']['response'] + "\n\n"
                        else:
                            ai_response += "##현재 추천 가능한 부가서비스를 찾을 수 없습니다.\n\n"
                        
                        # 쿠폰
                        ai_response += "##추천 쿠폰/혜택\n"
                        if category_results['coupon'] and category_results['coupon']['response']:
                            ai_response += category_results['coupon']['response'] + "\n"
                        else:
                            ai_response += "현재 추천 가능한 쿠폰을 찾을 수 없습니다.\n"
                        
                        # 모든 소스 문서 결합
                        used_sources = []
                        for category_data in category_results.values():
                            if category_data and category_data.get('sources'):
                                used_sources.extend(category_data['sources'])
                        
                        print(f"DEBUG: Comprehensive mode completed - used {len(used_sources)} total sources")
                 
                 
                elif search_filter and intent != "comprehensive" and intent != "general":
                    print(f"DEBUG: 사후 필터링 적용 - 타겟 타입: {search_filter.get('type')}")
                    original_count = len(used_sources)
                    used_sources = [doc for doc in used_sources if doc.metadata.get('type') == search_filter.get('type')]
                    print(f"DEBUG: 필터링 후 소스 개수: {len(used_sources)} (원래: {original_count})")
                    
                    if used_sources:
                        filtered_context = "\n\n".join([doc.page_content for doc in used_sources])
                        
                        if intent == "vass":
                            enhanced_prompt = f"""다음 검증된 데이터를 기반으로 부가서비스를 추천하세요.

                                검증된 부가서비스 데이터:
                                {filtered_context}

                                사용자 질문: {message}

                                지시사항:
                                1. 반드시 한국어로 답변하세요
                                2. 위 데이터에서 부가서비스 하나를 선택하여 추천하세요
                                3. 서비스명, 가격, 혜택을 포함하여 20자 이내로 설명하세요
                                4. 영어로 답변하지 마세요

                                추천:"""

                        elif intent == "coupon":
                            enhanced_prompt = f"""다음은 쿠폰 데이터입니다. CSV 형식으로 되어 있으며, 각 줄은 하나의 쿠폰을 나타냅니다.
                                            
                                            형식: ID,쿠폰명,설명,코드,카테고리,생성일,만료일

                                            쿠폰 데이터:
                                            {filtered_context}

                                            지시사항:
                                            1. 반드시 한국어로 답변하세요
                                            2. 위 데이터에서 쿠폰 하나를 선택하여 추천하세요
                                            3. 쿠폰명, 가격, 혜택을 포함하여 20자 이내로 설명하세요
                                            4. 영어로 답변하지 마세요
                                            사용자 질문: {message}

                                            """
                        
                        else:  # intent == "plan"
                            enhanced_prompt = f"""다음은 통신 요금제 데이터입니다. 각 요금제의 정보를 정확히 읽고 추천하세요.

                                    예시 데이터 형식:
                                    - 요금제명: 5G 프리미어 에센셜
                                    - 가격: 89000
                                    - 데이터: 200GB
                                    - 통화: 무제한
                                    - 문자: 무제한
                                    - 혜택: 넷플릭스 무료
                                    - 카테고리 코드: 001 or 002

                                    중요: 데이터가 "무제한"이라고 명시되지 않은 경우, 절대 무제한이라고 말하지 마세요.
                                    무제한은 -1이 무제한입니다.
                                    예: "95"는 95GB를 의미하며, 무제한이 아닙니다, "-1"은 데이터 무제한을 의미합니다.
                                    카테고리 코드 001은 5G, 002는 LTE입니다. 절대 001, 002라고 말하지 마세요.
                                    예 카테고리 코드 001 -> 5G, 002 -> LTE

                                    검증된 요금제 데이터:
                                    {filtered_context}

                                    사용자 질문: {message}

                                    지시사항:
                                    1. 데이터에 적힌 그대로만 설명하세요
                                    2. 숫자만 있으면 GB 단위입니다 (예: 95 → 95GB)
                                    3. "무제한"이라고 명시된 경우만 무제한으로 설명
                                    4. 5G 요금제를 물어봤다면 카테코리 코드 001에서만 대답하세요.
                                    5. 가격 조건이 있다면 반드시 가격 조건에 맞는 요금제만 추천하세요.
                                    6. 가격은 원 단위로 표시 (예: 68000 → 68,000원)
                                    7. 혜택에 대한 부분은 설명하지 마세요.
                                    8. 설명 이유는 20자 이내로 설명하세요
                                    9. 한국어로 답변하세요

                                    추천:"""
                        
                        filtered_response = self.combine_docs_chain.invoke({
                            "input": enhanced_prompt,
                            "chat_history": chat_history_str,
                            "context": used_sources,
                            "user_id": user_id
                        })
                        ai_response = str(filtered_response)
                        print(f"DEBUG: 엄격한 검증으로 재생성된 응답 (intent: {intent})")

                    for i, source in enumerate(used_sources[:3]):
                        print(f"DEBUG: 소스 {i+1} 타입: {source.metadata.get('type', 'unknown')}")
                        print(f"DEBUG: 소스 {i+1} 파일: {source.metadata.get('source_file', 'unknown')}")
                        print(f"DEBUG: 소스 {i+1} 내용: {source.page_content[:100]}...")
                else:
                    response = self.combine_docs_chain.invoke({
                        "input": message,
                        "chat_history": chat_history_str,
                        "context": [],
                        "user_id": user_id
                    })
                    ai_response = str(response)
                    used_sources = []
            
            # 메시지 분류
            classification = self.message_classifier.classify_message(message, ai_response)
            message_type = classification["message_type"]
            if classification.get("mentioned_plans"):
                message_type = MessageType.SUGGESTION
                classification["message_type"] = MessageType.SUGGESTION
            # 신뢰도 계산
            final_confidence_score = 1.0
            if message_type == "SUGGESTION" and self.csv_verifier:
                mentioned_in_response = self.csv_verifier.find_mentioned_plans(ai_response)
                print(f"DEBUG: AI 응답에서 발견된 요금제: {mentioned_in_response}")
                
                response_confidence_scores = []
                for mentioned_plan in mentioned_in_response:
                    verification = self.csv_verifier.verify_plan_exists(mentioned_plan)
                    response_confidence_scores.append(verification['confidence'])
                    print(f"DEBUG: 응답 내 요금제 '{mentioned_plan}' 검증: {verification['confidence']}")
                
                if response_confidence_scores:
                    final_confidence_score = min(response_confidence_scores)
                    print(f"DEBUG: 최종 신뢰도: {final_confidence_score}")
                    
                    if final_confidence_score < 0.7:
                        ai_response += f"\n\n일부 정보의 정확성을 확인해주세요. 정확한 요금제 정보는 공식 홈페이지에서 확인 가능합니다."
            
            # 대화 기록 저장
            conversation_entry = {
                "timestamp": datetime.now().isoformat(),
                "human": message,
                "ai": ai_response,
                "message_type": message_type,
                "query_intent": intent
            }
            
            chat_history.append(conversation_entry)
            
            if len(chat_history) > self.memory_manager.max_conversation_length:
                chat_history, conversation_summary = self.memory_manager.summarize_old_conversations(
                    user_id, chat_history, conversation_summary
                )
            
            self.memory_manager.save_user_memory(user_id, chat_history, conversation_summary)
            user_id = int(user_id)
            return {
                "response": ai_response,
                "user_id": user_id,
                "message_type": message_type,
                "confidence_score": final_confidence_score,
                "verification_status": "정상 처리" if final_confidence_score >= 0.7 else "낮은 신뢰도 - 일부 불일치 가능",
                "query_intent": intent,
                "intent_confidence": intent_analysis["confidence"],
                "search_filter_applied": self.get_search_filter(intent),
                "used_knowledge": [doc.page_content[:100] + "..." for doc in used_sources]
            }
            
        except Exception as e:
            print(f"채팅 처리 오류: {e}")
            return {
                "response": f"죄송합니다. 오류가 발생했습니다: {e}",
                "user_id": user_id,
                "message_type": "error",
                "confidence_score": 0.0,
                "verification_status": "오류"
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
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    def get_category_condition_chain(self):
        # 한 번만 생성해서 self에 저장(최적화)
        if hasattr(self, "_category_condition_chain"):
            return self._category_condition_chain

        category_prompt = ChatPromptTemplate.from_template("""
        아래는 사용자의 종합 추천 요청 메시지입니다.

        - 요청: "{user_message}"
        - 카테고리: {category}  (plan=요금제, vass=부가서비스, coupon=쿠폰/혜택)

        위 요청 중 '{category}'(이)에 해당하는 조건(예: 가격, 특징, 용도 등)만 한국어 한 문장으로 명확히 정리해서 추출하세요.
        - 만약 '{category}'에 해당하는 조건이 명확히 없으면 '없음'이라고만 답하세요.
        - 예시: 
        * 요청: "넷플릭스 가능한 요금제, 영화 쿠폰, 데이터 많이 주는 부가서비스 추천해줘"
        * 카테고리: plan → "넷플릭스 가능"
        * 카테고리: vass → "데이터 많이 주는"
        * 카테고리: coupon → "영화 쿠폰"
        - 답변은 조건 키워드나 짧은 구로만. 불필요한 말은 포함하지 마세요.

        [조건만]: 
        """)
        self._category_condition_chain = category_prompt | self.analysis_llm | StrOutputParser()
        return self._category_condition_chain

    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        user_id = str(user_id)
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
    
    def update_vectorstore(self, file_paths: List[str], force_rebuild: bool = True) -> dict:
        try:
            print("DEBUG: update_vectorstore 시작")
            
            if force_rebuild:
                print("DEBUG: 강제 재생성 모드 시작")
                vector_path = Path(settings.VECTOR_STORE_DIR) / "faiss_index"
                if vector_path.exists():
                    print(f"DEBUG: 기존 벡터스토어 삭제 중: {vector_path}")
                    import shutil
                    shutil.rmtree(vector_path)
                    print("DEBUG: 기존 벡터스토어 삭제 완료")
            
            print("DEBUG: _update_vectorstore_internal 호출")
            result = self._update_vectorstore_internal(file_paths)
            print(f"DEBUG: update_vectorstore 완료: {result}")
            return result
            
        except Exception as e:
            print(f"DEBUG: update_vectorstore 오류: {e}")
            return {"success": False, "message": f"업데이트 오류: {e}"}
    
    def _update_vectorstore_internal(self, file_paths: List[str]) -> dict:
        print(f"DEBUG: _update_vectorstore_internal 시작, 파일 수: {len(file_paths)}")
        print(f"DEBUG: 파일 경로들: {file_paths}")
        
        all_documents = []
        for i, file_path in enumerate(file_paths):
            print(f"DEBUG: 파일 {i+1}/{len(file_paths)} 처리 중: {file_path}")
            path = Path(file_path)
            if path.exists():
                print(f"DEBUG: 파일 존재 확인됨: {file_path}")
                try:
                    loader = TextLoader(str(path), encoding='utf-8')
                    print(f"DEBUG: TextLoader 생성 완료")
                    documents = loader.load()
                    print(f"DEBUG: 문서 로드 완료, 문서 수: {len(documents)}")
                    
                    data_type = self._determine_data_type(path)
                    print(f"DEBUG: 데이터 타입 결정: {data_type}")
                    
                    for doc in documents:
                        doc.metadata['type'] = data_type
                        doc.metadata['source_file'] = path.name
                        doc.metadata['file_path'] = str(path)
                        print(f"DEBUG: 문서 메타데이터 설정 완료")
                    
                    all_documents.extend(documents)
                    print(f"DEBUG: 파일 {file_path} 처리 완료")
                except Exception as e:
                    print(f"DEBUG: 파일 로드 오류 {file_path}: {e}")
            else:
                print(f"DEBUG: 파일을 찾을 수 없습니다: {file_path}")
        
        print(f"DEBUG: 전체 문서 수: {len(all_documents)}")
        
        if not all_documents:
            print("DEBUG: 로드할 문서가 없음")
            return {"success": False, "message": "로드할 문서가 없습니다."}
        
        print("DEBUG: text_splitter 생성 중")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        print("DEBUG: 문서 분할 시작")
        splits = text_splitter.split_documents(all_documents)
        print(f"DEBUG: 문서 분할 완료, 청크 수: {len(splits)}")
        
        print("DEBUG: FAISS 벡터스토어 생성 시작")
        self.vectorstore = FAISS.from_documents(
            documents=splits, 
            embedding=self.embeddings
        )
        print("DEBUG: FAISS 벡터스토어 생성 완료")
        
        print("DEBUG: 벡터스토어 저장 중")
        vector_path = Path(settings.VECTOR_STORE_DIR) / "faiss_index"
        self.vectorstore.save_local(str(vector_path))
        print("DEBUG: 벡터스토어 저장 완료")
        
        print("DEBUG: RAG 체인 재설정 중")
        self._setup_chain()
        print("DEBUG: RAG 체인 재설정 완료")
        
        result = {
            "success": True, 
            "message": "벡터스토어 업데이트 완료",
            "chunks_created": len(splits)
        }
        print(f"DEBUG: 최종 결과: {result}")
        return result
    
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
            "memory_manager": "사용 가능",
            "metadata_filtering": "활성화" if self.vectorstore else "비활성화"
        }