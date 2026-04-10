import streamlit
from PyPDF2 import PdfReader
from langchain_core.prompts import PromptTemplate
from langchain_classic.text_splitter import  RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint

streamlit.title("PDF Summarize")
streamlit.markdown("Upload a PDF and get a quick AI-powered summary instantly.")
parser = StrOutputParser()
token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    huggingfacehub_api_token=token
)

model = ChatHuggingFace(llm=llm)

pdf = streamlit.file_uploader("UPLODE PDF",type="pdf")

templet = PromptTemplate(
    template="Summarize given text : {text} \n in length {length}",
    input_variables=['text'],
)

option = streamlit.selectbox(
    'select length of summary',
    ('long','medium','short'),
)

if streamlit.button("Summarize"):
    reader = PdfReader(pdf)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    split_text = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunk = split_text.split_text(text)

    chain = templet | model | parser

    out = chain.invoke({'text': chunk , 'length': option})

    streamlit.write(out)
