import streamlit as st
from langchain_openai import OpenAIEmbeddings
import os

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.vectorstores import Chroma


def load_documents(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f"Loading document {file} ...")
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f"Loading document {file} ...")
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        print(f"Loading document {file} ...")
        loader = TextLoader(file)
    else:
        print(f"Unsupported file type: {extension}")
        return None
    
    data = loader.load()
    return data


def chunk_data(data, chunk_size = 256, chaunk_overlap = 20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chaunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings(model = 'text-embedding-3-small', dimensions = 1536)
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory='./mychroma_db')
    return vector_store


def ask_and_get_answer(vector_store, q, k=3):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model = 'gpt-3.5-turbo', temperature =1)

    retriever = vector_store.as_retriever(search_type = 'similarity', search_kwargs = {'k': k})

    chain = RetrievalQA.from_chain_type(llm = llm, retriever = retriever, chain_type = 'stuff') # stuff is defaults - takes all the text in the input

    answer = chain.run(q)
    return answer


def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # print(f"Total tokens: {total_tokens}")
    # print(f"Embedding cost in USD : {total_tokens / 1000 * 0.0004:.6f}")
    return total_tokens, total_tokens / 1000 * 0.0004
    
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']



def process_question():
    if 'question' in st.session_state:
        del st.session_state['question']

if __name__ == '__main__':

    st.set_page_config(
    page_title="(RAG) Document Q&A App",  # This title will appear on the browser tab
    page_icon="📄",  # Optional: Set an icon for the browser tab

    )


    import os
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    st.image('./files/img.png')
    st.subheader("LLM Question Answering App")
    with st.sidebar:
        api_key = st.text_input('OpenAI API Key', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key

        uploaded_file = st.file_uploader("Upload a document", type=['pdf', 'docx', 'txt'])

        chunk_size = st.number_input('Chunk size', min_value=100, max_value=2048, value=512, on_change=clear_history)
        k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)
        add_data = st.button('Add data')
        if add_data:
            clear_history()
        
        if uploaded_file and add_data:
            with st.spinner('Reading, chunking and creating embeddings...'):
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./files', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_documents(file_name)
                chunks = chunk_data(data, chunk_size = chunk_size)
                st.write(f"Chunk size: {chunk_size}, Chunks: {len(chunks)}")

                tokens, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f"Total tokens: {tokens}, Embedding cost in USD : {embedding_cost:.6f}")

                vector_store = create_embeddings(chunks)

                st.session_state.vs = vector_store
                st.success('File uploaded, chunked and embedded successfully!')
    
    q = st.text_input('Ask a question about the content of the document', key='question')  
    
    if q:
    
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs
            # st.write(f'k: {k}')
            answer = ask_and_get_answer(vector_store, q, k=k)
            st.text_area('LLM Answer', value = answer)

            st.divider()
            if 'history' not in st.session_state:
                st.session_state.history = ''
            value = f'Q: {q}\nA: {answer}\n'
            st.session_state.history = f'{value}\n{ "-" *  100}\n{st.session_state.history}'
            h=st.session_state.history
            st.text_area('Chat History', value = h, key='history', height=400)

             

