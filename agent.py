import pandas as pd
from pymongo import MongoClient
from bson import ObjectId
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
import openai
from vanna.openai.openai_chat import OpenAI_Chat
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore
import os,json
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest
import duckdb
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
import uuid
import chromadb
from flask_apscheduler import APScheduler
import re
from langchain.prompts import PromptTemplate
import fitz
import requests
from csv_bot import description_data,generate_summary,generate_sql_query,get_answer,generate_answer,csv_gererate_graph_type


# To get the secrete keys from the .env file
load_dotenv()
# Getting connection of OPENAI API KEY credention 
API_KEY = os.getenv("OPENAI_API_KEY")

# Getting connection of mongodb credention 
# connection_string = os.getenv("mongo_conn_string")
connection_string = "mongodb+srv://ramprakashreddyk:ramPrakash1@cluster0.adfyuxw.mongodb.net/ahexchatbot?retryWrites=true&w=majority&appName=Cluster0"
print('mongo_conn_string:',connection_string)

# connection pool
connection_pools = {}
sql_objects = {}
MAX_POOL_TIMEOUT=24 * 60 * 60

# Connecting to the vector store
class VannaAgent(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)

vn = VannaAgent(config={'api_key':API_KEY, 'model': 'gpt-3.5-turbo-16k'})

upload_folder_path = 'uploads'
if not os.path.exists(upload_folder_path):
    os.makedirs(upload_folder_path)

chroma_client = chromadb.PersistentClient(path="local_chroma")

pdf_list=[]

# openai.api_key = "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"

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
    chunks_page_no={}
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
           chunks_page_no[str(chunk)]=page_number+1
    # Close the PDF document
    pdf_document.close()

    return chunks_page_no


def single_line_text(paragraph):
    # Remove line breaks and extra spaces
    single_line = ' '.join(paragraph.split())
    return single_line

def get_text_chunks(text):
    """Splits text into smaller chunks of 1000 characters each."""
    chunk_size = 15000
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def chroma_metadata(chunks,source,page_nos):
    chroma_metadata_file=[]
    for i in range(1,len(chunks)+1):
        metadata={f'source':str(source),'page_no':page_nos[i-1]}
        chroma_metadata_file.append(metadata)
    print("chroma_metadata_file::",chroma_metadata_file)
    return chroma_metadata_file
    

def get_related_txt(query,collection,file_name):
  print("entered")
  results = collection.query(
      query_texts=[query],
      n_results=2,
      where={'source':str(file_name)}
  )  
  return results


def all_functions(collection,file_name):
        global pdf_list
        chunks=[]
        chroma_metadatas=[]
        pdf_files=[]
        # for pdf in file:
        #     pdf_files.append(str(pdf.filename))
        # print("pdf_files::",pdf_files)
        # Iterate through each PDF file and extract text
        # for pdf_file_path in pdf_files:
        page_nos=[]
        chunks_file=[]
        extracted_text = extract_text_from_pdf(file_name)
        print('extracted_text::',extracted_text)
        if extracted_text == {}:
            return  {"message": "Text not extracted"}
        else:
            for keys, values in  extracted_text.items():
                chunks_file.append(str(keys))
                page_nos.append(str(values))
            print('chunks_file::',chunks_file)
            # chroma_ids+=ast.literal_eval(ids)
            print('ids::',page_nos)
            print("lenght of chunks_file:", len(chunks_file))
            chroma_metadatas+=chroma_metadata(chunks_file,file_name,page_nos)
            chunks+=chunks_file
            pdf_list.append(file_name)
            chroma_documents=[]
            for txt in chunks:
                chroma_documents.append(txt)
            chroma_ids = [str(uuid.uuid4()) for _ in chroma_documents]
            print('chroma_ids::',chroma_ids)
            collection.add(
                documents=chroma_documents,
                metadatas=chroma_metadatas,
                ids = chroma_ids
            )
            pdf_list.append(file_name)

def get_answer_pdf(user_query,collection,files):
    try:
        related_docs=''
        related_txt=''
        reference_file_names=[]
        docs_distances={}
        for name in files:
            print('names::',str(name))
            query_related_text=get_related_txt(user_query,collection,str(name))
            print('query_related_text::',query_related_text)
            if query_related_text!={}:
                documents_related_text=query_related_text['documents']
                documents_related_text=documents_related_text[0]
                print("==="*49)
                print('documents_related_text::',documents_related_text)
                similarity_distances=query_related_text['distances']
                similarity_distances=similarity_distances[0]
                print('similarity_distances::',similarity_distances)
                docs_distances[sum(similarity_distances)]=documents_related_text
                print("docs_distances",docs_distances)
                # if sum(similarity_distances):
                try:
                    for txt in documents_related_text:
                        related_txt+=txt
                    file_names=query_related_text['metadatas'][0] # To extract File name 
                    print("file_names::",file_names)
                    # for filename in file_names:
                    #     if filename not in reference_file_names:
                    #         reference_file_names.append(filename)
                    
                except Exception as e:
                    print("Exception::",e)
                    return {"message": "No related text found for the query."}
                if related_txt =='':
                    return {"message": "No related text found for the query."}
        min_distance_docs = docs_distances[min(docs_distances.keys())] 
        print("min_distance_docs::",min_distance_docs)
        for doc in min_distance_docs:
            reference=collection.get(include=["metadatas"],where_document={"$contains":doc})
            print("************************************")
            print("reference['metadatas']::",reference['metadatas'])
            print("************************************") 
            reference_file_names.append(reference['metadatas'])

        for txt in min_distance_docs:
            related_docs+=txt
        print("reference_file_names::",reference_file_names)
        print("related_txt::",related_docs)
    except Exception as e:
        print()
        return {e} 
    result=answer_user_query(user_query, related_docs)
    print("result::",result)
    # Define the regular expression pattern to match the content field
    pattern = r"content='([^']*)'"

    # Search for the pattern in the result string
    match = re.search(pattern, result)
    if match :
        content = match.group(1) # Extract the content from the matched group
        response={"Answer":content,
                  "Related Text":related_docs,
                  "Reference":reference_file_names}
        print("Answer:")
        print(content)
        # print("Reference:",reference_txt)
    else:
        response={"message":"No content found in the result string."}
    return response

# Convert data to Json format
def df_to_json(data_frame):
    df = data_frame
    json_data = df.to_json(orient="records")
    return json_data


def response_data(question):
    """Generate response data based on the provided question.

    Args:
        question (str): The user question for which response data is generated.

    Returns:
        tuple: A tuple containing the generated SQL query, JSON data, summary data, and DataFrame.
            - generated_sql_query (str): The SQL query generated based on the question.
            - json_data (str): The JSON-formatted data extracted from the DataFrame.
            - summary_data (str): The summary generated based on the question and DataFrame.
            - df (DataFrame): The DataFrame containing the query results.
    """
    generated_sql_query = vn.generate_sql(question)
    df = vn.run_sql(generated_sql_query)

    # Generate summary based on question and data frame
    summary_data = get_summary(question, df)

    # Convert data to Json format
    json_data = df_to_json(df)

    return generated_sql_query, json_data, summary_data, df


def get_summary(question, df):
    """Generate summary based on the provided question and DataFrame.

    Args:
        question (str): The user question for which the summary is generated.
        df (DataFrame): The DataFrame containing the data to be summarized.

    Returns:
        str: The summary data generated based on the provided question and DataFrame.
    """
    summary_data = vn.generate_summary(question, df=df)
    return summary_data

def generate_completion(prompt,gpt_model):
    """
    Generate a completion based on the provided prompt, question.
    Args:
    - prompt (str): The system understanding prompt.
    Returns:
    - str: The completion generated by the AI model.
    """
    completion = openai.chat.completions.create(
        # model="gpt-3.5-turbo-16k",
        model=gpt_model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ]
    )
    return completion.choices[0].message.content


def generate_system_understanding(question,gpt_model):
    """Generate system understanding response based on the provided user question.

    Args:
        question (str): The user question for which system understanding is generated.

    Returns:
        str: System understanding response generated based on the provided question.
    """
 
    system_understanding_prompt = f"System Understanding: Please provide an explanation or understanding '{question}'."

    system_understanding_completion = generate_completion(system_understanding_prompt,gpt_model)
    
    return system_understanding_completion


def gererate_graph_type(dataframe_info,gpt_model):
    """Generate the graph type based on the provided dataframe information.

    Args:
        dataframe_info (str): Information about the dataframe to determine the appropriate graph type.

    Returns:
        str: The suggested type of graph based on the provided dataframe information.
    """

    graph_type_prompt = f"Graph Type: What type of graph can be created based on the given information? '{dataframe_info}' If none, please specify 'None'. Give in one word"
    
    graph_type_completion = generate_completion(graph_type_prompt,gpt_model)

    return graph_type_completion


# Initialize Flask app
app = Flask(__name__)

@app.route('/sqlagent/v2/api', methods=['POST'])
def ask_question():
    print("helloo")
    """Endpoint to handle incoming questions.

    This function retrieves the question from the request, connects to MongoDB to fetch the required data, 
    establishes a connection to the database, generates SQL query, runs the query, 
    and returns a JSON response containing the SQL query, JSON data, summary, system understanding, status, 
    and chart presentation.

    Returns:
        Response: JSON response containing information about the SQL query, JSON data, summary, 
        system understanding, status, and chart presentation.
    """
    request_data = request.form.to_dict()
    if not request_data:
        raise BadRequest('Invalid request. Missing form data.')
    question = request_data.get('user_input')
    user_id = request_data.get('user_id')
    source_id = request_data.get('source_id')
    gpt_model = request_data.get('gpt_model')
    source_id_list = eval(source_id)
    # num_tokens = 0
    print("Hiiiiiiiiii")
    # Connect to MongoDB
    client = MongoClient(connection_string)
    db = client["test"]
    collection = db["dataitems"]
    # matching the source_id with respected to the user_id 
    pipeline=[
            {
                "$match": {
                    "user_id": user_id
                }
            },
            {
                "$project": {
                    "matchedContent": {
                        "$filter": {
                            "input": "$content",
                            "cond": { "$in": ["$$this._id", {
                                "$map": {
                                    "input": source_id_list,
                                    "as": "id",
                                    "in": { "$toObjectId": "$$id" }
                                }}
                            ]}
                        }
                    }
                }
            },
            {
                "$unwind": "$matchedContent"
            },
            {
                "$replaceRoot": {
                    "newRoot": "$matchedContent"
                }
            },
            {
                "$project": {
                    "filePath": 1,
                    "fileType": 1,
                    "_id": 1,
                    "sourceName":1
                }
            }
        ]

    # Perform aggregation
    result = list(collection.aggregate(pipeline))
    print('result:',result)
    print('result type:',type(result))
    print('user_id:',user_id)
    print('user_id type:',type(user_id))
    
    
    if result[0]["fileType"] == 'dbConnection':
        conn_path = result[0]["filePath"]
        print('mysql string:',conn_path)
        print('mysql string type:',type(conn_path))
        if source_id not in connection_pools:
            try:
                connection_pools[source_id] = create_engine(
                conn_path, pool_size=5, max_overflow=2, pool_timeout=MAX_POOL_TIMEOUT,poolclass=QueuePool)
                # engine = connection_pools[source_id]
                print("Connection to the database successful!")
            except OperationalError as e:
                # If an error occurs during connection, print the error message
                print(f"Error connecting to the database: {e}")

        engine = connection_pools[source_id]
        print('pool:',connection_pools)
        print('engine pool:',engine)
        # Test the connection
        conn = engine.connect()


        # running
        def generated_dataframes(sql: str) -> pd.DataFrame:
            df = pd.read_sql_query(sql, conn)
            return df

        vn.run_sql = generated_dataframes
        vn.run_sql_is_set = True
        # Get all the values
        generated_sql_query, json_data, summary_data, data_frame = response_data(question)
        # Get generated system_understanding
        system_understanding = generate_system_understanding(question,gpt_model)
        # Get generated Graph type
        graph_type = gererate_graph_type(data_frame,gpt_model)
        json_data = data_frame.to_dict(orient='records')
        

        response = {
                    'user_input': question,
                    'sql_query' : generated_sql_query,
                    'json_data' : json_data,
                    'response' : summary_data,
                    'system_understanding' : system_understanding,
                    'chart_presentation' : graph_type,
                    'gpt_model':gpt_model}
        return jsonify(response)
    elif result[0]['fileType'] == 'csv':
        col_lst=[]
        table_schema={}
        conn = duckdb.connect()
        for ids in result:
            print('ids:',ids)
            if str(ids['_id']) not in connection_pools:
                csvpath = ids['filePath']
                
                print('csvpath:',csvpath)
                key = str(ids['_id'])
                print('key:',key)
                connection_pools[str(ids['_id'])]=csvpath
                print('csv pool has created successfully ...')
            csv_engine = connection_pools[str(ids['_id'])]
            
            print(' csv_engine pool:',csv_engine)
            # conn = duckdb.connect()
            df = duckdb.from_csv_auto(csv_engine,conn)
            # Register DataFrames as temporary tables
            print('sourceName:',ids['sourceName'])
            table_name=ids['sourceName']
            conn.register(table_name, df)

        
            column_names = description_data(df)
            print('column_names:',column_names)
            table_schema[table_name]=column_names
            # col_lst.extend(column_names)
            # print('col:',col_lst)
        print("="*10)
        print('pool:',connection_pools)
        print("="*10)
        print('all table_schema:',table_schema)
        summary_data = generate_summary(  table_schema,gpt_model)
        print("Summary : ", summary_data)

        sql_query = generate_sql_query(question, table_schema,gpt_model)
        print("SQL Query : ", sql_query)
        sql_result = conn.execute(sql_query).fetchdf()
        # print(sql_result)
        # result_df = get_answer(sql_query)


        answer_data = generate_answer(question, sql_result,gpt_model)
        print("Answer : ", answer_data)
        

        chart_presentation = csv_gererate_graph_type(sql_result,gpt_model)

        json_data = sql_result.to_dict(orient='records')
        # json_data = df_to_json(answer_data)
       
        return jsonify({
            'user_input': question,
            'source_id' : source_id,
            'json_data' : json_data,
            'chart_presentation' : chart_presentation,
            'sql_query' : sql_query,
            'response' : answer_data,
            'summary' : summary_data,
            'gpt_model':gpt_model
        }), 200

    elif result[0]['fileType'] == 'pdf':
        global pdf_list
        if question is None:
            return jsonify({"message":"No query found"})
        elif user_id is None:
            return jsonify({"message":"No user id found"})
        else:
            collections=chroma_client.list_collections()
            collection_names = [collection.name for collection in collections] # Extract names from Collection objects
            print('collection_names::',collection_names)
            try:
                files_list=[]
                for pdf in result:
                    file_url=pdf['filePath']
                    files_list.append("uploads\\"+ str(file_url.split('/')[-1]))
                if str(user_id) not in collection_names:
                    collection = chroma_client.create_collection(name=user_id)
                    pdf_list=[]
                    for pdf in result:
                        file_url=pdf['filePath']
                        file_name=file_url.split('/')[-1]
                        if file_name not in pdf_list:
                            print("entered into if.......")
                            print("pdf::",file_name)
                            try:
                                response = requests.get(file_url)
                                local_file_path = os.path.join(upload_folder_path, os.path.basename(file_name))
                                if response.status_code == 200:
                                    with open(local_file_path, 'wb') as file:
                                        file.write(response.content)
                                    print(f"File downloaded successfully to: {local_file_path}")
                                else:
                                    print(f"Failed to download file. Status code: {response.status_code}")
                            except Exception as e:
                                print(f"Error downloading file: {e}") 
                            print("local file path::",local_file_path)                           
                            all_functions(collection,str(local_file_path)) 
                            os.remove(local_file_path)
                else:
                    collection = chroma_client.get_collection(user_id)
                    files_chroma=collection.get(include=["metadatas"],where={"source": {"$ne": 'None'}})
                    print('files_chroma::',files_chroma['metadatas'])
                    print("222222222222222222222222")
                    source_names=[]
                    for item in files_chroma['metadatas']:
                        if item['source'] not in source_names:
                            source_names.append(item['source'])
                    print(source_names)
                    for pdf in result:
                        file_url=pdf['filePath']
                        file_name=file_url.split('/')[-1]
                        print("file_name::",file_name)
                        file_name="uploads\\"+file_name
                        print("file_name::",file_name)
                        if str(file_name) not in source_names:
                            print("str(pdf.filename)",str(file_name))
                            print("entered into else.......")
                            try:
                                response = requests.get(file_url)
                                local_file_path = os.path.join(upload_folder_path, os.path.basename(file_name))
                                if response.status_code == 200:
                                    with open(local_file_path, 'wb') as file:
                                        file.write(response.content)
                                    print(f"File downloaded successfully to: {local_file_path}")
                                else:
                                    print(f"Failed to download file. Status code: {response.status_code}")
                            except Exception as e:
                                print(f"Error downloading file: {e}")
                            print("local file path::",local_file_path) 
                            all_functions(collection,str(local_file_path))
                        print("local_file_path",local_file_path) 
                        os.remove(local_file_path)
                # print("collection::",collection)
                print("files_list::",files_list)
                print("quest",question)
                response = get_answer_pdf(question,collection,files_list) 
                return jsonify(response)
            except Exception as e:
                print(e)
                return jsonify({'error': 'Something went wrong !!!'}), 400
    else:
        return jsonify({'error': 'filepath not found !!!'}), 400

@app.route('/hello', methods=['GET'])
def hello():
    print("helloo")
    return jsonify({'message': 'Helloo !!!'})
if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)







    