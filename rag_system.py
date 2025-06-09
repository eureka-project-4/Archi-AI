from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

class RAGSystem:
    def __init__(self, embeddings: OpenAIEmbeddings):
        self.embeddings = embeddings
        self.vectorstore = None
        self.retriever = None
    
    def setup_rag_system(self, pricing_data_file: str):
        try:
            loader = TextLoader(pricing_data_file, encoding='utf-8')
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            splits = text_splitter.split_documents(documents)
            
            self.vectorstore = FAISS.from_documents(
                documents=splits, 
                embedding=self.embeddings
            )
            
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            
            print(f"RAG 시스템 초기화 완료: {len(splits)}개의 문서 청크 생성됨")
            return True
            
        except Exception as e:
            print(f"RAG 시스템 초기화 중 오류 발생: {e}")
            self.retriever = None
            return False
    
    def get_retriever(self):
        return self.retriever
    
    def get_vectorstore(self):
        return self.vectorstore
    
    def is_available(self) -> bool:
        return self.retriever is not None