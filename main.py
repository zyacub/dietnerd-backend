from helper_functions import *

from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
from fastapi.responses import StreamingResponse
from starlette.responses import JSONResponse

import asyncio
from sse_starlette.sse import EventSourceResponse
from concurrent.futures import ThreadPoolExecutor

import uuid
import json
from urllib.parse import unquote

from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

from helper_functions import * 

import heapq

#Sim search
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from collections import defaultdict

update_queues = defaultdict(asyncio.Queue)

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryModel(BaseModel):
    user_query: str

disclaimer = """
DietNerd is an exploratory tool designed to enrich your conversations with a registered dietitian or registered dietitian nutritionist, who can then review your profile before providing recommendations.
Please be aware that the insights provided by DietNerd may not fully take into consideration all potential medication interactions or pre-existing conditions.
To find a local expert near you, use this website: https://www.eatright.org/find-a-nutrition-expert
"""

executor = ThreadPoolExecutor()

# Create a global event loop
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

def run_in_executor(func, *args):
    return loop.run_in_executor(executor, func, *args)

@app.get("/")
async def root():
    return "Hello! Go to /docs!'"

@app.get("/db_sim_search/{question:str}")
async def sim_search(question:str):
   decoded_query = unquote(question)
   result = await sim_score(decoded_query)
   return result

@app.get("/db_get/{query:str}")
async def db_get_endpoint(query: str):
   decoded_query = unquote(query)
   result = await query_db_final(decoded_query)
   return result

@app.get("/check_valid/{question:str}")
async def check_valid(question:str):
   question_validity = determine_question_validity(question)
   if question_validity == 'False - Meal Plan/Recipe':
    final_output = ("I'm sorry, I cannot help you with this question. For any questions or advice around meal planning or recipes, please speak to a registered dietitian or registered dietitian nutritionist.\n"
                    "To find a local expert near you, use this website: https://www.eatright.org/find-a-nutrition-expert.")
    print(final_output)
   elif question_validity == 'False - Animal':
    final_output = ("I'm sorry, I cannot help you with this question. For any questions regarding an animal, please speak to a veterinarian.\n"
                   "To find a local expert near you, use this website: https://vetlocator.com/.")
   else:
    final_output = "good"
   return {"response" : final_output}

@app.post("/process_query")
async def process_query(query: QueryModel, background_tasks: BackgroundTasks):
    session_id = str(uuid.uuid4())
    background_tasks.add_task(process_user_query, query.user_query, session_id)
    return JSONResponse({"session_id": session_id})

@app.get("/sse")
async def sse(session_id: str = Query(default=None)):
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    return EventSourceResponse(event_generator(session_id))

async def event_generator(session_id: str):
    queue = asyncio.Queue()
    update_queues[session_id] = queue
    try:
        while True:
            data = await queue.get()
            if isinstance(data, dict) and "final_output" in data:
                yield {"event": "message", "data": json.dumps(data)}
                break
            else:
                yield {"event": "message", "data": json.dumps({"update": data})}
    finally:
        del update_queues[session_id]

def process_user_query(user_query, session_id):
    # Query Generation 
    start_poc = time.time()
    general_query, query_contention, query_list = query_generation(user_query)
    end_poc = time.time()

    print("Generated strings")
    loop.run_until_complete(send_update(session_id, "Generated search strings..."))
    # Article Retrieval
    start_api = time.time()
    deduplicated_articles_collected = collect_articles(query_list)
    end_api = time.time()

    print("Retrieved Articles")
    loop.run_until_complete(send_update(session_id, f"Retrieved {len(deduplicated_articles_collected)} Articles..."))
    # Relevance Classifier
    start_relevant = time.time()
    relevant_articles, irrelevant_articles = concurrent_relevance_classification(deduplicated_articles_collected, user_query)
    end_relevant = time.time()

    print("relevant articles")
    loop.run_until_complete(send_update(session_id, f"Classified {len(relevant_articles)} Relevant Articles..."))

    # Article Match
    start_processing = time.time()
    reliability_analysis_df = connect_to_reliability_analysis_db()
    matched_articles, articles_to_process = article_matching(relevant_articles, reliability_analysis_df)

    print("matched articles")
    # Article Processing
    relevant_article_summaries = concurrent_article_processing(articles_to_process)

    # Write Processed Articles to DB
    write_articles_to_db(relevant_article_summaries, env)

    all_relevant_articles = list(itertools.chain(relevant_article_summaries, matched_articles))
    end_processing = time.time()

    print(f"Synthesizing {len(all_relevant_articles)} Articles...")
    loop.run_until_complete(send_update(session_id, f"Synthesizing {len(all_relevant_articles)} Articles..."))

    # Final Output
    start_output = time.time()
    final_output = generate_final_response(all_relevant_articles, user_query)
    end_output = time.time()

    poc_duration = end_poc - start_poc
    api_duration = end_api - start_api
    relevance_classifier_duration = end_relevant - start_relevant
    article_processing_duration = end_processing - start_processing
    final_output_duration = end_output - start_output
    total_runtime = poc_duration + api_duration + article_processing_duration + final_output_duration

    write_output_to_db(user_query, final_output, all_relevant_articles, total_runtime, env)
    end_output = time.time()

    print('-'*200)
    print(final_output)
    print('-'*20)
    print('User Question: ', user_query)
    print('-'*20)
    print('General Query: ', general_query)
    print('-'*20)
    print('Points of Contention: ', query_contention)
    print('-'*20)

    print('# Matched: ', len(matched_articles))
    print('# Processed: ', len(articles_to_process))
    print('# Relevant: ', len(all_relevant_articles))
    print('# Irrelevant: ', len(irrelevant_articles))
    print('Relevant Articles: ', all_relevant_articles)
    print('-'*20)
    print('Total Runtime: ', total_runtime)
    print(' -- ')
    print('[Section 1] Points of Contention: ', poc_duration)
    print('[Section 2] PubMed API Call: ', api_duration)
    print('[Section 3] Relevance Classification: ', relevance_classifier_duration)
    print('[Section 4] Reliability Analysis: ', article_processing_duration)
    print('[Section 5] Final Synthesis: ', final_output_duration)


    return_obj = {
       "end_output": final_output,
       "relevant_articles": all_relevant_articles
    }

    main_output, citations = split_end_output(return_obj["end_output"])
    relevant_articles = return_obj.get("relevant_articles", [])
    updated_citations = match_citations_with_articles(citations, all_relevant_articles)
    return_obj["end_output"] = final_output
    return_obj["citations_obj"] = updated_citations
    return_obj["citations"] = citations

    loop.run_until_complete(send_update(session_id, return_obj))

    return return_obj

async def send_update(session_id, data):
    if session_id in update_queues:
        await update_queues[session_id].put(data)

async def query_db_final(query: str):
   load_dotenv("ATT81274.env")
   mydb = mysql.connector.connect(
    host=os.getenv('host'),
    port=os.getenv('port'),
    user=os.getenv('user'),
    password=os.getenv('password'),
    database=os.getenv('database')
    )

   mycursor = mydb.cursor()
   sql = f"SELECT * FROM finaloutputs WHERE question = '{query}'"

   mycursor.execute(sql)

   myresult = mycursor.fetchall()
   with open("output.json", "w") as f:
      json.dump(myresult, f, indent=4)
   return myresult

def match_citations_with_articles(citations, articles):
    citation_dict = {}
    
    # Create a dictionary of articles keyed by a normalized version of their citation
    article_dict = {normalize_citation(article["citation"]): article for article in articles}
    
    for citation in citations:
        normalized_citation = normalize_citation(citation)
        citation_slice = normalized_citation[10:22]  # Extract the slice
        
        for article_citation in article_dict:
            if citation_slice in article_citation:
                article = article_dict[article_citation]
                citation_dict[citation] = {
                    "PMID": article["PMID"],
                    "PMCID": article["PMCID"],
                    "URL": article["url"],
                    #"full_text": article["full_text"],
                    "Summary": article["summary"]
                }
                break  # Stop searching once a match is found
    
    return citation_dict

def normalize_citation(citation):
    # Remove the citation number at the beginning, if present
    citation = re.sub(r'^\s*\[?\d+\.?\]?\s*', '', citation)
    
    # Remove all punctuation and convert to lowercase
    citation = re.sub(r'[^\w\s]', '', citation.lower())
    
    # Remove extra whitespace
    citation = ' '.join(citation.split())
    
    return citation

def split_end_output(end_output: str):
    # Remove substring "disclaimer" ignoring line breaks
    cleaned_output = end_output.replace(disclaimer, "")
    
    # Split the modified cleaned_output into main response and citations
    main_response, citations_section = cleaned_output.split("References:", 1)
    
    # Extract individual citations
    citations = re.split(r'\n{1,2}', citations_section.strip())
    
    return main_response.strip(), citations

async def sim_score(question: str):
   mydb = mysql.connector.connect(
    host=os.getenv("host"),
    port=os.getenv("port"),
    user=os.getenv("user"),
    password=os.getenv("password"),
    database=os.getenv("database")
  )
   mycursor = mydb.cursor()
   sql = f"SELECT question FROM finaloutputs;"
   mycursor.execute(sql)
   myresult = mycursor.fetchall()
   resultdict = []

   for x in myresult:
      resultdict.append(x[0])

   scores_dict = calculate_similarity(resultdict, question)
   print(scores_dict)

   min_heap = []
      
   for item in scores_dict:
      score = item[0]
      sentence = item[1]
      if (score > 0.23):
         heapq.heappush(min_heap, (score, sentence))
      if (len(min_heap) > 3):
         heapq.heappop(min_heap)
   top_k_sentences = [(score, sentence) for score, sentence in sorted(min_heap, reverse=True)]
   print(top_k_sentences)
   return top_k_sentences

def calculate_similarity(sentences, source_sentence):
    # Combine source sentence with the list of sentences
    all_sentences = sentences + [source_sentence]
    
    # Create the TF-IDF vectorizer and transform the sentences
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_sentences)
    
    # Calculate the cosine similarity between the source sentence and all other sentences
    cosine_similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1]).flatten()
    
    # Combine the similarity scores with the sentences
    similarity_scores = [(score, sentence) for score, sentence in zip(cosine_similarities, sentences)]
    
    return similarity_scores


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)