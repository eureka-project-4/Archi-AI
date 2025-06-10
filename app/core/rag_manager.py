import os
from pathlib import Path
from typing import List, Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from app.config import settings

class RAGManager:
    def __init__(self):
        os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
        
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS
        )
        
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        
        # 디렉토리 생성
        Path(settings.PRICING_DATA_DIR).mkdir(parents=True, exist_ok=True)
        Path(settings.VECTOR_STORE_DIR).mkdir(parents=True, exist_ok=True)
    
    def initialize(self):
        """초기 RAG 시스템 설정"""
        try:
            # 기존 벡터스토어 로드 시도
            vector_path = Path(settings.VECTOR_STORE_DIR) / "faiss_index"
            if vector_path.exists():
                self.vectorstore = FAISS.load_local(
                    str(vector_path), 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print("기존 벡터스토어 로드됨")
            else:
                # 요금제 파일들로 새 벡터스토어 생성
                self._create_vectorstore_from_files()
            
            if self.vectorstore:
                self.retriever = self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": settings.RETRIEVAL_K}
                )
                self._setup_chain()
                
        except Exception as e:
            print(f"RAG 시스템 초기화 오류: {e}")
            self.retriever = None
    
    def _create_vectorstore_from_files(self):
        """데이터 파일들로부터 벡터스토어 생성"""
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
            
            # 벡터스토어 저장
            vector_path = Path(settings.VECTOR_STORE_DIR) / "faiss_index"
            self.vectorstore.save_local(str(vector_path))
            
            print(f"벡터스토어 생성 완료: {len(splits)}개 청크")
    
    def _setup_chain(self):
        """챗봇 체인 설정"""
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
    
    def update_vectorstore(self, file_paths: List[str]) -> dict:
        """벡터스토어 업데이트"""
        try:
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
            
            # 새 벡터스토어 생성
            self.vectorstore = FAISS.from_documents(
                documents=splits, 
                embedding=self.embeddings
            )
            
            # 저장
            vector_path = Path(settings.VECTOR_STORE_DIR) / "faiss_index"
            self.vectorstore.save_local(str(vector_path))
            
            # 리트리버 및 체인 재설정
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
            
        except Exception as e:
            return {"success": False, "message": f"업데이트 오류: {e}"}
    
    def chat(self, user_id: str, message: str, chat_history: str = "") -> dict:
        """채팅 처리"""
        try:
            if self.rag_chain is None:
                return {
                    "response": "죄송합니다. AI 시스템이 초기화되지 않았습니다.",
                    "used_knowledge": []
                }
            
            if self.retriever:
                # RAG 체인 실행
                response = self.rag_chain.invoke({
                    "input": message,
                    "chat_history": chat_history,
                    "user_id": user_id
                })
                
                ai_response = response.get("answer", str(response))
                used_sources = [doc.page_content[:100] + "..." for doc in response.get("context", [])]
                
            else:
                # 기본 체인 실행
                response = self.rag_chain.invoke({
                    "input": message,
                    "chat_history": chat_history,
                    "context": "요금제 정보를 로드할 수 없습니다.",
                    "user_id": user_id
                })
                ai_response = str(response)
                used_sources = []
            
            return {
                "response": ai_response,
                "used_knowledge": used_sources
            }
            
        except Exception as e:
            return {
                "response": f"죄송합니다. 오류가 발생했습니다: {e}",
                "used_knowledge": []
            }