import requests
import streamlit as st

#streamlit framework
st.title('Multimodal RAG Application: Upload file and ask question about it')

uploaded_file= st.file_uploader('Choose the file to upload')

file_upload= None

if(st.button('Submit')):
    files= {'file': uploaded_file.getvalue()}
    response = requests.post("http://localhost:8000/file_upload/", files= files)
    file_upload= response.json()['status']

if(file_upload== 'success'):
    st.write('Your file successfully uploaded')


question= st.text_input('Enter your question here')

if(question):
    response= requests.post('http://localhost:8000/ask_question/', json= {"question": question})
    st.write('Answer is extracted from the following page numbers: ', response.json()['page_list'])
    st.write('Answer to your question is: ')
    st.write(response.json()['answer'])

    st.write('Some relevent images from the documents as per the query:')
    for img_name in response.json()['img_names']:
        path= './data/'+ img_name
        st.image(path)


