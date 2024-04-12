import fitz  # PyMuPDF
import re
import chromadb
import openai
from langchain.prompts import PromptTemplate
from flask import Flask, jsonify, request
import os
import uuid
import time

app = Flask(__name__)

# chroma_client = chromadb.Client()
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

chroma_client = chromadb.PersistentClient(path="pdf_local")


pdf_list=[]
# Replace with your OpenAI API key
openai.api_key = "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"

def generate_text_summary(text_to_summarize):
  """
  Summarizes a given text using the OpenAI API and GPT-3.5 model.

  Args:
      text_to_summarize: The text to be summarized (string).

  Returns:
      A string containing the summarized text.
  """

  # Define a clear prompt template
  prompt_template = PromptTemplate.from_template(
      "summarize the following text \n{text} the summary should contain atleast 50 words\n\nSummary:"
  )

  # Construct the complete prompt with the user-provided text
  prompt = prompt_template.format(text=text_to_summarize)
  # Use openai.Completion.create for text generation

  response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ]
    )
  message_content = response.choices[0].message.content

  return message_content

def answer_user_query(user_query, query_related_text):
  """
  Attempts to answer a user query using the provided related text.

  Args:
      user_query: The user's query (string).
      query_related_text: Text related to the user's query (string).

  Returns:
      A string containing the generated answer or an error message.
  """

  try:
    # Craft a prompt that instructs the AI to find the answer in the text
    prompt_template = PromptTemplate.from_template(
        "The user asked: {query}\n"
        "Find a specific answer to the question from the following text:\n"
        "{text} and the answer should contain atleast 30 words\n\n"
        "Answer:"
    )
    # Construct the complete prompt with the user query and related text
    prompt = prompt_template.format(query=user_query, text=query_related_text)
    response = openai.chat.completions.create(
          model="gpt-3.5-turbo",
          messages=[
              {
                  "role": "user",
                  "content": prompt,
              },
          ]
      )
    return str(response) # converting the response into a string
  except Exception as e:
    # Handle potential errors during answer generation
    error_message = f"An error occurred while generating the answer: {e}"
    return error_message

def extract_text_from_pdf(pdf_path):
    text = ''
    page_no={}
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)

    # Iterate through each page
    for page_number in range(len(pdf_document)):
        # Get the page
        page = pdf_document[page_number]
        # Extract text from the page
        text = page.get_text()
        text=single_line_text(text)
        for chunk in get_text_chunks(text):
           page_no[str(chunk)]=page_number+1
    # Close the PDF document
    pdf_document.close()

    return page_no


def single_line_text(paragraph):
    # Remove line breaks and extra spaces
    single_line = ' '.join(paragraph.split())
    return single_line

def get_text_chunks(text):
    """Splits text into smaller chunks of 1000 characters each."""
    chunk_size = 1500
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def chroma_metadata(chunks,source,page_nos):
    chroma_metadata_file=[]
    for i in range(1,len(chunks)+1):
        metadata={f'source_{i}':str(source),'page_no':page_nos[i-1]}
        chroma_metadata_file.append(metadata)
    print("chroma_metadata_file::",chroma_metadata_file)
    return chroma_metadata_file
    
# pdf_files = ['fess101.pdf'] #,'harry-potter-book-collection-1-4.pdf','Harry Potter and The Chamber of Secrets.pdf']

# text=''
# chunks=[]
# chroma_metadatas=[]
# page_nos=[]
# # Iterate through each PDF file and extract text
# for pdf_file_path in pdf_files:
#     ids=[]
#     chunks_file=[]
#     extracted_text = extract_text_from_pdf(pdf_file_path)
#     print('extracted_text::',extracted_text)
#     for keys, values in  extracted_text.items():
#        chunks_file.append(str(keys))
#        ids.append(str(values))
#     print('chunks_file::',chunks_file)
#     # chroma_ids+=ast.literal_eval(ids)
#     print('ids::',ids)
#     print("lenght of chunks_file:", len(chunks_file))
#     chroma_metadatas+=chroma_metadata(chunks_file,pdf_file_path,ids)
#     chunks+=chunks_file
#     # print('-' * 80)

#     # print(justified_paragraph)
#     # print('-' * 80)

# chroma_documents=[]
# for txt in chunks:
#    chroma_documents.append(txt)

# # storings ids
# chroma_ids=[]
# for i in range(1,len(chunks)+1):
#    chroma_ids.append((f"doc_{i}"))

# collection.add(
#     documents=chroma_documents,
#     metadatas=chroma_metadatas,
#     ids=chroma_ids
# )

def get_related_txt(query,collection):
  results = collection.query(
      query_texts=[query],
      n_results=2
  )  
  return results


# user_query='What can we know about the past?'
# summaraized_query=generate_text_summary(user_query)
# print("\n\nUser Query:\t",user_query,"\nSummary:\t",summaraized_query)



# print("==="*50)
# print("Related Text before::",query_related_text)
# print("==="*50)







# print("Related Text ::",related_txt)
# print("Reference ::", file_name)



def get_reference_txt(result,collection):
  results = collection.query(
      query_texts=[result],
      n_results=1
  )  
  return results
def all_functions(collection,file):
        global pdf_list
        chunks=[]
        chroma_metadatas=[]
        pdf_files=[]
        # for pdf in file:
        #     pdf_files.append(str(pdf.filename))
        # print("pdf_files::",pdf_files)
        # Iterate through each PDF file and extract text
        # for pdf_file_path in pdf_files:
        ids=[]
        chunks_file=[]
        extracted_text = extract_text_from_pdf(file)
        print('extracted_text::',extracted_text)
        for keys, values in  extracted_text.items():
            chunks_file.append(str(keys))
            ids.append(str(values))
        print('chunks_file::',chunks_file)
        # chroma_ids+=ast.literal_eval(ids)
        print('ids::',ids)
        print("lenght of chunks_file:", len(chunks_file))
        chroma_metadatas+=chroma_metadata(chunks_file,file,ids)
        chunks+=chunks_file
        pdf_list.append(file)
        chroma_documents=[]
        for txt in chunks:
            chroma_documents.append(txt)
        chroma_ids = [str(uuid.uuid4()) for _ in chroma_documents]

        collection.add(
            documents=chroma_documents,
            metadatas=chroma_metadatas,
            ids = chroma_ids
        )

def get_answer(user_query,collection):
    query_related_text=get_related_txt(user_query,collection)
    documents_related_text=query_related_text['documents']
    documents_related_text=documents_related_text[0]
    print("==="*49)
    print('documents_related_text::',documents_related_text)
    related_txt=''
    for txt in documents_related_text:
        related_txt+=txt
    # To extract File name 
    file_name=query_related_text['metadatas'][0]
    sources=[]
    for i in file_name:
        for key, value in i.items():
            if value not in sources:
                sources.append(value)
    result=answer_user_query(user_query, related_txt)

    # Define the regular expression pattern to match the content field
    pattern = r"content='([^']*)'"

    # Search for the pattern in the result string
    match = re.search(pattern, result)
    if match :
        content=match.group(1)
        reference_txt=get_reference_txt(content,collection)
        reference_txt=reference_txt['documents']
        reference_txt=reference_txt[0]  #reference_txt=generate_text_summary(reference_txt[0])
        content = match.group(1) # Extract the content from the matched group
        response={"Answer":content,
                  "Related Text":related_txt,
                  "Reference":file_name}
        print("Answer:")
        print(content)
        # print("Reference:",reference_txt)
    else:
        response={"message":"No content found in the result string."}
    return response


# def reset_chromadb(client):
#     for collection in client.collections:  # Loop through all collections directly
#         collection.delete()

 # Sleep for remaining time in the cycle

@app.route("/pdf_chat",methods=['GET'])
def pdf_chat():
    global pdf_list
    user_query = request.args.get('user_query')
    user_id = request.args.get('user_id')
    file = request.files.getlist('file')
    print('files::',file)
    collections=chroma_client.list_collections()
    # Extract names from Collection objects
    collection_names = [collection.name for collection in collections]
    print('collection_names::',collection_names)
    if str(user_id) not in collection_names:
        for collection in collection_names:
            chroma_client.delete_collection(collection)
            print("deleted")
        collection = chroma_client.create_collection(name=user_id)
        pdf_list=[]
        for pdf in file:
            if str(pdf.filename) not in pdf_list:
                print("entered into if.......")
                print("pdf::",str(pdf.filename))
                all_functions(collection,str(pdf.filename)) 
    else:
        collection = chroma_client.get_collection(user_id)
        for pdf in file:
            print("pdf::",pdf)
            if str(pdf.filename) not in pdf_list:
                print('pdf_list::',pdf_list)
                print("str(pdf.filename)",str(pdf.filename))
                print("entered into else.......")
                all_functions(collection,str(pdf.filename)) 
    # print("collection::",collection)
    response = get_answer(user_query,collection)
    return jsonify(response)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=False)

# while True:
#     reset_chromadb(chroma_client)
#     time.sleep(60 * 30 - time.time() % (60 * 30)) 