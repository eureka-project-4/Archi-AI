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
        You are an expert telecommunications plan recommendation specialist. 
        Provide personalized recommendations based on user needs and usage patterns.
        
        **Current User: {user_id}**
        
        **CRITICAL INSTRUCTIONS:**
        - ONLY recommend actual telecommunications plans (mobile phone plans, data plans, 5G/LTE plans)
        - DO NOT recommend coupons, vouchers, gift cards, or lifestyle services
        - Focus on monthly mobile service plans with data, calls, and SMS
        - If context contains non-telecommunications items, ignore them completely
        
        **Guidelines:**
        1. Engage naturally and friendly with users
        2. Understand user's call volume, data usage, and budget requirements
        3. Provide accurate recommendations based on the provided plan information
        4. Remember previous conversations and maintain context
        5. Clearly explain recommendation reasons
        6. For returning users, provide personalized service based on conversation history
        7. **Important**: Only recommend plans that actually exist. Verify plan names and information accurately.
        
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
            2. "service": Additional services only (roaming, security, quality improvement)
            3. "coupon": Coupons/benefits only (movie discounts, shopping, lifestyle benefits)
            4. "comprehensive": Comprehensive recommendations (plans + services + coupons together)
            5. "general": General inquiries (greetings, customer service, policies)
            
            Keyword hints:
            - plan: plan, data, calls, SMS, monthly fee, 5G, LTE, unlimited, pricing
            - service: service, roaming, security, additional, premium, add-on
            - coupon: coupon, discount, benefit, reward, movie, shopping, lifestyle
            - comprehensive: comprehensive, all, everything, package, total, complete, together
            - general: hello, inquiry, customer service, policy, terms, help
            
            User input: {user_input}
            
            Respond in JSON format only:
            {{
                "intent": "plan|service|coupon|comprehensive|general",
                "confidence": 0.0-1.0,
                "reasoning": "brief explanation of classification decision"
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
    
    def search_plans_by_price(self, price_condition: Dict[str, Any], data_type: str = None) -> List[Dict[str, Any]]:
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
        
        all_results = self.csv_verifier.find_plans_by_criteria(**criteria)
        
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
            - Examples: "5G Premier Essential", "T Plan Special", "LTE Basic"
            
            User input: {user_input}
            
            Respond in JSON format only:
            {{
                "extracted_plans": ["list of specific plan names"],
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
            return extracted_data.get("extracted_plans", [])
            
        except Exception as e:
            print(f"요금제명 추출 실패: {e}")
            return []
    
    def get_search_filter(self, intent: str) -> Optional[Dict[str, str]]:
        filter_map = {
            "plan": {"type": "plan"},
            "service": {"type": "service"}, 
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
        if intent == "plan":
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
        chat_history, conversation_summary, is_existing_user = self.memory_manager.load_user_memory(user_id)
        return chat_history, conversation_summary, is_existing_user
    
    def chat(self, user_id: str, message: str) -> Dict[str, Any]:
        return self.chat_with_verification(user_id, message)
    
    def chat_with_verification(self, user_id: str, message: str) -> Dict[str, Any]:
        try:
            
            intent_analysis = self.analyze_query_intent(message)
            intent = intent_analysis["intent"]
            
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
                elif intent == "service":
                    data_type_filter = "service"
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
                    matching_plans = self.search_plans_by_price(price_conditions, data_type_filter)
                else:
                    matching_plans = self.search_plans_by_price(price_conditions, None)
                
                if matching_plans:
                    print(f"DEBUG: 조건에 맞는 {search_description} {len(matching_plans)}개 발견")
                    
                    top_plans = matching_plans[:3]
                    
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
                        "message_type": MessageType.SUGGESTION.value,  # Enum을 문자열로 변환
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
                            "message_type": MessageType.BLOCKED_MESSAGE.value,  # Enum을 문자열로 변환
                            "mentioned_plans": [plan_name],
                            "confidence_score": verification['confidence'],
                            "verification_status": "차단됨 - 존재하지 않는 요금제",
                            "query_intent": intent
                        }
                    elif verification['exists'] and verification['confidence'] >= 0.8:
                        print(f"DEBUG: 요금제 '{plan_name}' 정확히 매칭됨 - CSV 직접 사용")
                        plan_info = verification['matched_plan']
                        ai_response = f"""
    {plan_name} 요금제 정보를 안내해드리겠습니다.

    **{plan_name}**
    - 월 요금: {plan_info['price']:,}원
    - 데이터: {plan_info['data']}
    - 통화: {plan_info['calls']}
    - 문자: {plan_info['sms']}
    - 혜택: {plan_info['benefit']}

    이 요금제에 대해 더 궁금한 점이 있으시면 언제든지 문의해주세요.
                        """
                        
                        conversation_entry = {
                            "timestamp": datetime.now().isoformat(),
                            "human": message,
                            "ai": ai_response.strip(),
                            "message_type": MessageType.SUGGESTION.value,  # Enum을 문자열로 변환
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
                
                if search_filter and intent != "comprehensive" and intent != "general":
                    print(f"DEBUG: 사후 필터링 적용 - 타겟 타입: {search_filter.get('type')}")
                    original_count = len(used_sources)
                    used_sources = [doc for doc in used_sources if doc.metadata.get('type') == search_filter.get('type')]
                    print(f"DEBUG: 필터링 후 소스 개수: {len(used_sources)} (원래: {original_count})")
                    
                    if used_sources:
                        filtered_context = "\n\n".join([doc.page_content for doc in used_sources])
                        
                        # 특정 요금제를 물어보는 경우와 추천을 요청하는 경우를 구분
                        if asked_plans:  # 특정 요금제명이 언급된 경우
                            enhanced_prompt = f"""Based ONLY on the following verified data, answer the user's question.

    CRITICAL RULE: If the user asks about a specific plan name that is NOT exactly found in the data below, you MUST respond with "죄송합니다. 해당 요금제를 현재 데이터에서 찾을 수 없습니다."

    DO NOT:
    - Make up plan names that don't exist in the data
    - Combine information from different plans 
    - Infer or guess pricing/features not explicitly stated
    - Use similar plan names as if they were the requested plan

    ONLY provide information about plans that are explicitly mentioned in the verified data below.

    Verified Data:
    {filtered_context}

    User Question: {message}

    Remember: Only answer about plans that are EXACTLY named in the verified data above. If the exact plan name is not found, clearly state it's not available."""
                        else:  # 추천이나 조건 기반 질문인 경우
                            enhanced_prompt = f"""Based on the following data, provide recommendations or information to answer the user's question.

    You are a helpful telecommunications plan advisor. Use the verified data below to:
    - Recommend suitable plans based on user needs
    - Compare different options
    - Explain features and benefits
    - Answer general questions about telecommunications services

    Verified Data:
    {filtered_context}

    User Question: {message}
    Previous Conversation: {chat_history_str}

    Please provide a helpful, personalized response based on the user's needs and the available data."""
                        
                        filtered_response = self.combine_docs_chain.invoke({
                            "input": enhanced_prompt,
                            "chat_history": chat_history_str,
                            "context": used_sources,
                            "user_id": user_id
                        })
                        ai_response = str(filtered_response)
                        print(f"DEBUG: 엄격한 검증으로 재생성된 응답")
                
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
                        ai_response += f"\n\n⚠️ 일부 정보의 정확성을 확인해주세요. 정확한 요금제 정보는 공식 홈페이지에서 확인 가능합니다."
            
            # 대화 기록 저장
            conversation_entry = {
                "timestamp": datetime.now().isoformat(),
                "human": message,
                "ai": ai_response,
                "message_type": message_type.value if hasattr(message_type, 'value') else message_type,  # Enum을 문자열로 변환
                "query_intent": intent
            }
            
            chat_history.append(conversation_entry)
            
            if len(chat_history) > self.memory_manager.max_conversation_length:
                chat_history, conversation_summary = self.memory_manager.summarize_old_conversations(
                    user_id, chat_history, conversation_summary
                )
            
            self.memory_manager.save_user_memory(user_id, chat_history, conversation_summary)
            
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