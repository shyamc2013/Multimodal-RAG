from llama_index.core.indices.multi_modal.base import (
    MultiModalVectorStoreIndex,
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import SimpleDirectoryReader, StorageContext, Settings
from llama_index.embeddings.clip import ClipEmbedding
from llama_index.multi_modal_llms.ollama import OllamaMultiModal
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import SimpleMultiModalQueryEngine

import qdrant_client

from fastapi import FastAPI, File, UploadFile
import uvicorn    #server used for running fastapi
from pydantic import BaseModel
import tempfile
import os
from dotenv import load_dotenv
import shutil
import pymupdf


load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")


app= FastAPI()

Settings.llm= OllamaMultiModal(model="llava")

qa_tmpl_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)
qa_tmpl = PromptTemplate(qa_tmpl_str)

#value of this variable will be set later
query_engine= None



def set_query_engine():
   
    # Create a local Qdrant vector store
    client = qdrant_client.QdrantClient(path="qdrant_mm_db")

    text_store = QdrantVectorStore(
        client=client, collection_name="text_collection"
    )
    image_store = QdrantVectorStore(
        client=client, collection_name="image_collection"
    )
    storage_context = StorageContext.from_defaults(
        vector_store=text_store, image_store=image_store
    )

    image_embed_model = ClipEmbedding()

    # Create the MultiModal index
    documents = SimpleDirectoryReader("./data/").load_data()
    index = MultiModalVectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        image_embed_model=image_embed_model
    )

    global query_engine
    query_engine = index.as_query_engine(
        text_qa_template=qa_tmpl
    )

#extract images from the pdf file
def extract_images(source, file):
    #open the file
    pdf_file = pymupdf.open(file) 
    image_counter = 0

    for page_index in range(0, len(pdf_file)):
        # get the page itself 
        page = pdf_file[page_index] 

        for image in page.get_images():
            image_id = image[7] # img<no>
            image_block_id = image[0] # block number 
            image_title_block_id = image_block_id+1 # image title block number
            image_dim = image[2],image[3] # image dimension details
            
            # print(f"[+] Page:{page_index}, Image : {image_id}, Block:{image_block_id}, Image Dim:{image_dim}")
            image_counter += 1

            # save the images to the local file system
            pix = pymupdf.Pixmap(pdf_file, image[0]) 
            # image file name contains image name 'img<no>' and block number
            pix.save(os.path.join(source, f"{image_id}_{image_block_id}.png"))
    
    return image_counter


#PDF reader and calls the set_retrieval_chain() fuction
#this function is called when api is called as soon as pdf is uploaded
@app.post("/file_upload/")
async def load_pdf(file: UploadFile):

    # Define the file path with .pdf extension
    file_location = f"./data/{os.path.splitext(file.filename)[0]}.pdf"
    
    # Save the file to the specified location
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    pwd = os.getcwd()
    # file path you want to extract images from 
    source = os.path.join(pwd,'data')
    filename= file.filename+ '.pdf'
    file_path = os.path.join(source, filename)

    total_images= extract_images(source, file_path)
    print(f'Total images extracted- {total_images}')

    set_query_engine()

    status= 'success'
    return {'status': status}


#Use of pydantic helps in automatic validation of incoming request body. Otherwise we have to do it by our own
#It also helps in Serialization: Converts models to and from dictionaries and JSON
class QuestionRequest(BaseModel):
    question: str

#get response for the question asked
#this function is called when api is called as soon as question is asked
@app.post("/ask_question/")
def get_response(request: QuestionRequest):
    print('get_response')
    question = request.question
    response = query_engine.query(question)

    answer= response.response

    page_nos= set()
    for text_node in response.metadata['text_nodes']:
        page_no= text_node.metadata['page_label']
        page_nos.add(int(page_no))

    page_nos= sorted(page_nos)
    page_nos_list= list(page_nos)

    img_names= []
    for img_node in response.metadata['image_nodes']:
        img_name= img_node.metadata['file_name']
        img_names.append(img_name)

    return {'answer': answer, 'page_list': page_nos_list, 'img_names': img_names}


if __name__== '__main__':
    uvicorn.run(app, host= 'localhost', port= 8000)