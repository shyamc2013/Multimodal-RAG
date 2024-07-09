import requests
import streamlit as st

#streamlit framework
st.title('Multimodal RAG Application: Upload file and ask question about it')

uploaded_file= st.file_uploader('Choose the file to upload')

file_uploaded= False

if(st.button('Submit')):
    files= {'file': uploaded_file}
    response = requests.post("http://localhost:8000/file_upload/", files= files)
    file_uploaded= response.json()['status']== 'success'

if(file_uploaded):
    st.write('Your file successfully uploaded')
    
question= st.text_input('Enter your question here')

if(question):
    response= requests.post('http://localhost:8000/ask_question/', json= {"question": question})
    st.write('**Answer is extracted from the following page numbers:** ', *(response.json()['page_list']))
    st.write('**Answer is extracted from the following file:** ', response.json()['file_name'])
    st.write('**Answer to your question is:** ')
    st.write(response.json()['answer'])

    st.write('**Some relevent images from the documents as per the query:**')
    for img_name in response.json()['img_names']:
        path= './data/'+ img_name
        st.image(path)


