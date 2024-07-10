import os
# Question to Query
import pandas as pd
import time
import numpy as np
import openai
from openai import OpenAI
from urllib.parse import unquote

# Database
import ast
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv
from scipy import spatial # for calculating vector similarities for search
import json
import itertools
import fitz

# Information Retrieval
from Bio import Entrez
from Bio.Entrez import efetch, esearch
from metapub import PubMedFetcher
import re
import requests
from bs4 import BeautifulSoup

# Summarizer
from concurrent.futures import ThreadPoolExecutor, as_completed
import string
from tenacity import retry # Exponential Backoff
# wait_random_exponential stop_after_attempt

# Output Synthesis
import textwrap

# Fast API and server imports
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Any, Dict

#Sim search
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Other
import json
import heapq

load_dotenv('ATT81274.env')
client = OpenAI(api_key=os.getenv("openai"))


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return "Hello! Go to /docs!'"


@app.get("/db_get/{query:str}")
async def db_get_endpoint(query: str):
   decoded_query = unquote(query)
   result = await query_db_final(decoded_query)
   return result

@app.get("/db_sim_search/{question:str}")
async def sim_search(question:str):
   decoded_query = unquote(question)
   result = await sim_score(decoded_query)
   return result

@app.get("/check_valid/{question:str}")
async def check_valid(question:str):
   question_validity = await determine_question_validity(question)
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


@app.get("/query_generation/{user_query}")
def query_generation_endpoint(user_query: str):
    general_query, query_contention, query_list = query_generation(user_query)
    return {
        "general_query": general_query,
        "query_contention": query_contention,
        "query_list": query_list
    }

@app.post("/collect_articles/")
def collect_articles_endpoint(
   query_list: Any = Body(None)
):
    deduplicated_articles_collected = collect_articles(query_list)
    return {"deduplicated_articles_collected": deduplicated_articles_collected}


@app.post("/article_matching/")
async def article_matching_endpoint(
   deduplicated_articles_collected: Any = Body(None)
):
    reliability_analysis_df = connect_to_reliability_analysis_db()
    matched_articles, articles_to_process = article_matching(deduplicated_articles_collected, reliability_analysis_df)
    return {
        "matched_articles": matched_articles,
        "articles_to_process": articles_to_process
    }


@app.post("/reliability_analysis_processing/")
def reliability_analysis_processing_endpoint(
    body: Dict[str, Any] = Body(...)
):
    user_query = body.get("user_query")
    articles_to_process = body.get("articles_to_process")
    relevant_article_summaries, irrelevant_article_summaries = reliability_analysis_processing(articles_to_process, user_query)
    return {
        "relevant_article_summaries": relevant_article_summaries,
        "irrelevant_article_summaries": irrelevant_article_summaries
    }


@app.post("/write_articles_to_db/")
def write_articles_to_db_endpoint(
   relevant_article_summaries: Any = Body(None)
):
    write_articles_to_db(relevant_article_summaries, "ATT81274.env")
    return {"status": "success"}


@app.post("/get_all_articles/")
def write_articles_to_db_endpoint(
   relevant_article_summaries: Any = Body(None),
   matched_articles: Any = Body(None)
):
    all_relevant_articles = list(itertools.chain(relevant_article_summaries, matched_articles))
    return {"all_relevant_articles": all_relevant_articles}


@app.post("/generate_final_response/")
async def generate_final_response_endpoint(
    body: Dict[str, Any] = Body(...)
):
    user_query = body.get("user_query")
    all_relevant_articles = body.get("all_relevant_articles")
    final_output = generate_final_response(all_relevant_articles, user_query)
    return_obj = {
       "end_output": final_output,
       "relevent_articles": all_relevant_articles,
       "total_runtime": 60.0
       }
    main_output, citations = split_end_output(return_obj["end_output"])
    relevent_articles = return_obj.get("relevent_articles", [])
    updated_citations = match_citations_with_articles(citations, all_relevant_articles)
    return_obj["end_output"] = final_output
    return_obj["citations_obj"] = updated_citations
    return_obj["citations"] = citations
    write_output_to_db(user_query, final_output, all_relevant_articles, 60.00, "ATT81274.env")
    return {"final_output": final_output, "response_obj": return_obj}




@app.get("/generate/{question:str}")
def generate_endpoint(question: str):
   result = generate(question)
   return result


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
      if (score > 0):
         heapq.heappush(min_heap, (score, sentence))
      if (len(min_heap) > 3):
         heapq.heappop(min_heap)
   top_k_sentences = [(score, sentence) for score, sentence in sorted(min_heap, reverse=True)]
   print(top_k_sentences)
   return top_k_sentences

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

   return myresult

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

async def determine_question_validity(query):
  valid_question_response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
      {
        "role": "system",
        "content": """You are an expert in classifying user questions. Your task is to determine whether a user's question involves meal planning, recipe creation, or is asking on behalf of an animal. Meal planning questions typically focus on organizing meals, dietary schedules, and nutritional balance over a period. Recipe creation questions usually involve specific ingredients, cooking methods, and detailed instructions for preparing a dish. If the user's question is about meal planning or recipe creation, return "False - Meal Plan/Recipe". If the question is asking on behalf of an animal, return "False - Animal". If the question does not involve any of these topics, return "True". Provide only "True", "False - Meal Plan/Recipe", or "False - Animal" based on the criteria and no other text.

        Here are some examples:

        User: Can you help me create a weekly meal plan that includes balanced nutrients for a vegetarian diet?
        AI: False - Meal Plan/Recipe

        User: How do I make a low-carb lasagna?
        AI: False - Meal Plan/Recipe

        User: What are some ideas for healthy snacks I can prepare for my kids?
        AI: False - Meal Plan/Recipe

        User: What is a good strategy for planning meals for someone with diabetes?
        AI: False - Meal Plan/Recipe

        User: What are the health benefits of intermittent fasting?
        AI: True

        User: What is the best diet for my cat?
        AI: False - Animal

        User: Can dogs eat raw meat?
        AI: False - Animal
        """
      },
      {
        "role": "user",
        "content": query
      }
    ],
    temperature=0.2,
    top_p=1
  )

  question_validity = valid_question_response.choices[0].message.content
  return question_validity

def query_generation(query):
  #### CREATING GENERAL QUERY FOR CONTEXT
  general_query_response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
      {
        "role": "system",
        "content": """You are an expert in generating precise and effective PubMed queries to help researchers find relevant scientific articles. Your task is to create a broad query that will retrieve articles related to a specific topic provided by the user. The queries should be optimized to ensure they return the most relevant results. Use Boolean operators and other search techniques as needed. Format the query in a way that can be directly used in PubMed's search bar. Return only the query and no other text.

        Here are some examples:

        User: Is resveratrol effective in humans?
        AI: (resveratrol OR "trans-3,5,4'-trihydroxystilbene") AND human

        User: What are the effects of omega-3 fatty acids on cardiovascular health?
        AI: (omega-3 OR "omega-3 fatty acids") AND "cardiovascular health"

        User: What does the recent research say about the role of gut microbiota in diabetes management?
        AI: ("gut microbiota") AND ("diabetes management") AND ("recent"[Publication Date])
        """
      },
      {
        "role": "user",
        "content": query
      }
    ],
    temperature=0.7,
    top_p=1
  )

  general_query = general_query_response.choices[0].message.content


  #### IDENTIFYING POINTS OF CONTENTION
  poc_response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
      {
        "role": "system",
        "content": """You are an expert in generating precise and effective PubMed queries to help researchers find relevant scientific articles. Your task is to list up to 4 of the top points of contention around the given question, making sure each point is relevant and framed back to the original question.
        Each point should be as specific as possible and have a title and a brief summary of what the conversation is around this point of contention. The points should be ranked in order of how controversial the point is (how much debate and conversation is happening), where 1 is the most controversial.
        For each and every point of contention provided, generate 1 broad PubMed search query. Use Boolean operators and other search techniques as needed. Format each query in a way that can be directly used in PubMed's search bar.

        Format the response like the following and do not include any other words:
        * Point of Contention 1: <title>
        Summary: <summary>
        Query: <search_query>

        Here is an example:

        User: Is resveratrol effective in humans?
        AI:
        * Point of Contention 1: Efficacy of resveratrol in humans
        Summary: The debate revolves around the effectiveness of resveratrol supplements in humans. Some studies suggest that resveratrol may have various health benefits, such as cardiovascular protection and anti-aging effects, while others argue that the evidence is inconclusive or insufficient
        Query: (resveratrol OR "trans-3,5,4'-trihydroxystilbene") AND human

        * Point of Contention 2: Dosage and Timing of Resveratrol Intake
        Summary: This point of contention focuses on the optimal dosage and timing of resveratrol intake for life span extension. Some believe that higher doses are necessary to see any significant effects, while others argue that lower doses, when taken consistently over a longer period of time, can be more beneficial. Additionally, there is debate about whether resveratrol should be taken in a fasting state or with food to maximize its absorption and potential benefits.
        Query: (resveratrol OR "trans-3,5,4'-trihydroxystilbene") AND dose

        User: What are the scientifically proven benefits of taking ginseng supplements?
        AI:
        * Point of Contention 1: Efficacy of Ginseng in Cognitive Function
        Summary: The debate revolves around the effectiveness of ginseng supplements in enhancing cognitive function. Some studies suggest that ginseng may have various cognitive benefits, such as improving memory and concentration, while others argue that the evidence is inconclusive or insufficient.
        Query: (ginseng OR "Panax ginseng") AND cognition

        * Point of Contention 2: Ginseng for Immune System Enhancement
        Summary: This point of contention focuses on the role of ginseng in immune system enhancement. Some believe that ginseng can significantly boost immune system function, while others argue that the evidence is not strong enough to make such claims.
        Query: (ginseng OR "Panax ginseng") AND immune

        * Point of Contention 3: Ginseng for Energy and Stamina
        Summary: The efficacy of ginseng in increasing energy and stamina is a common point of debate. While some studies suggest that ginseng can help to combat fatigue and increase physical performance, others argue that these effects are not consistently observed across studies.
        Query: (ginseng OR "Panax ginseng") AND energy

        * Point of Contention 4: Safety and side effects of Gingko supplements
        Summary: The safety of Gingko supplements is a point of contention, with some concerns raised about potential side effects such as dizziness, upset stomach, and increased bleeding risk. While some studies suggest that Gingko supplements are generally safe, others argue that caution should be exercised, especially when combined with certain medications or in individuals with specific health conditions.
        Query: (Gingko OR "Gingko Biloba") AND (safety OR "side effects")
        """
      },
      {
        "role": "user",
        "content": query
      }
    ],
    temperature=0.6,
    top_p=1
  )

  query_contention = poc_response.choices[0].message.content

  #### AGGREGATE ALL 5 QUERIES
  pattern = r"Query: (.*)"
  matches = re.findall(pattern, query_contention)
  query_list = []
  for match in matches:
      query_list.append(match)

  query_list.append(general_query)
  return general_query, query_contention, query_list

def article_retrieval(query):
  Entrez.email = os.getenv('ENTREZ_EMAIL')

  # Set if you run the query multiple times due to stochastic output
  search_ids     = set()
  search_queries = set()
  num_query_attempts = 1

  for _ in range(num_query_attempts):
    search_queries.add(query)
    search_results = esearch(db="pubmed", term=query, retmax=10, sort="relevance")
    retrieved_ids = Entrez.read(search_results)["IdList"]
    search_ids = search_ids.union(retrieved_ids)

  search_ids_list = list(search_ids)

  if search_ids_list == []:
    article_data = ''
    return article_data
  else:
    articles = efetch(db="pubmed", id=search_ids_list, rettype="xml")
    article_data = Entrez.read(articles)["PubmedArticle"]
    return article_data

def collect_articles(query_list):
  articles_collected = []

  for query in query_list:
    article_group = article_retrieval(query)
    if article_group is None:
      break
    else:
      articles_collected.append(article_group)

  key_path = ['MedlineCitation', 'PMID']

  # Deduplicate collected articles
  seen = set()
  deduplicated_articles_collected = []

  for sublist in articles_collected:
    deduplicated_sublist = []
    for item in sublist:
      nested_key_value = item
      for key in key_path:
        nested_key_value = nested_key_value[key]

      if nested_key_value not in seen:
        seen.add(nested_key_value)
        deduplicated_sublist.append(item)
      deduplicated_articles_collected.append(deduplicated_sublist)

  return deduplicated_articles_collected

#@title string_to_dict
# Define a function to safely evaluate the string as a dictionary
def string_to_dict(string):
    try:
        return ast.literal_eval(string)
    except ValueError:
        return {}  # Return an empty dictionary in case of an error
    

#@title connect_to_reliability_analysis_db
def connect_to_reliability_analysis_db():
  mydb = mysql.connector.connect(
    host=os.getenv('host'),
    port=os.getenv('port'),
    user=os.getenv('user'),
    password=os.getenv('password'),
    database=os.getenv('database')
  )

  mycursor = mydb.cursor()

  sql = f"SELECT * FROM reliabilityanalysis"

  mycursor.execute(sql)

  # output: list of tuples
  myresult = mycursor.fetchall()

  reliability_analysis_mysql = pd.DataFrame(myresult, columns=['article_id', 'article_json', 'id'])

  # Apply the function to each row in the "Summary" column to convert from string to dictionary
  reliability_analysis_mysql['article_json'] = reliability_analysis_mysql['article_json'].apply(string_to_dict)

  # # Normalize the column of dictionaries to a DataFrame
  articles_df = pd.json_normalize(reliability_analysis_mysql['article_json'])

  # # Concatenate the new DataFrame with the original one, excluding the original "Summary" column
  reliability_analysis_df = pd.concat([reliability_analysis_mysql.drop(columns=['article_json']), articles_df], axis=1)
  return reliability_analysis_df

#@title deduplicate_by_pmid
#### Necessary as the database sometimes still has duplicates, despite upsert
def deduplicate_by_pmid(dict_list):
    seen_pmids = set()
    deduplicated_list = []

    for item in dict_list:
        pmid = item.get('PMID')
        if pmid and pmid not in seen_pmids:
            deduplicated_list.append(item)
            seen_pmids.add(pmid)

    return deduplicated_list

#@title article_matching
def article_matching(articles_collected, reliability_analysis_df):
  matched_articles = []
  articles_to_process = []

  seen_article_ids = {}
  for article_data in articles_collected:
    for article in article_data:
      article_id = str(article['MedlineCitation']['PMID'])
      if reliability_analysis_df['PMID'].isin([str(article_id)]).any():
        ### create article dictionary
        article_row = reliability_analysis_df[reliability_analysis_df['PMID'] == str(article_id)]
        article_dict = article_row.to_dict(orient='records')[0]
        matched_articles.append(article_dict)
      elif article_id not in seen_article_ids:
        seen_article_ids[article_id] = article  # Use article_id as a key to ensure distinctness
        articles_to_process.append(article)

  deduplicated_matched_articles = deduplicate_by_pmid(matched_articles)
  return deduplicated_matched_articles, articles_to_process

#@title generate_ama_citation
def generate_ama_citation(article):
  try:
    authors = article["MedlineCitation"]["Article"]["AuthorList"]
    author_names = ", ".join([f"{author['LastName']} {author['Initials']}" for author in authors])
  except KeyError:
    author_names = ""

  try:
    title = article["MedlineCitation"]["Article"]["ArticleTitle"]
  except KeyError:
    title = ""

  try:
    journal = article["MedlineCitation"]["Article"]["Journal"]["Title"]
  except KeyError:
    journal = ""

  try:
    pub_date = article["MedlineCitation"]["Article"]["Journal"]["JournalIssue"]["PubDate"]['Month'] + ' ' + article["MedlineCitation"]["Article"]["Journal"]["JournalIssue"]["PubDate"]['Day'] + ', ' + article["MedlineCitation"]["Article"]["Journal"]["JournalIssue"]["PubDate"]['Year']#article["PubmedData"]["History"][0]["Year"]
  except KeyError:
    pub_date = ""

  try:
    volume = article["MedlineCitation"]["Article"]["Journal"]["JournalIssue"]["Volume"]
  except KeyError:
    volume = ""

  try:
    issue = article["MedlineCitation"]["Article"]["Journal"]["JournalIssue"]["Issue"]
  except KeyError:
    issue = ""

  try:
    pages = article["MedlineCitation"]["Article"]["Pagination"]["MedlinePgn"]
  except KeyError:
    pages = ""

  return f"{author_names}. {title}. {journal}. {pub_date};{volume}({issue}):{pages}."

#@title all_full_text_options
def all_full_text_options(url):
  """Capture all full-text options as linked on a PubMed article website.
  If no buttons, it will return NULL.
  """

  headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
  response = requests.get(url, headers=headers)

  soup = BeautifulSoup(response.content, 'html.parser')

  full_text_links_section = soup.find('div', class_='full-text-links-list')

  # Extract all the 'a' tags within Full Text section
  links = full_text_links_section.find_all('a') if full_text_links_section else []

  links_dict = {}

  # Populate the dictionary with data-ga-action as keys and URLs as values
  for link in links:
      data_ga_action = link.get('data-ga-action', 'No action found')
      link_url = link['href']
      links_dict[data_ga_action] = link_url

  return links_dict

#@title clean_extracted_text
def clean_extracted_text(text):
    """
    Cleans the extracted text from a PDF to improve readability.

    Parameters:
        text (str): The extracted text from the PDF.

    Returns:
        str: Cleaned text.
    """
    # Replace newline characters with spaces
    cleaned_text = text.replace('\n', ' ')

    # Remove any strange unicode characters (like \u202f, \u2002, \xa0)
    cleaned_text = re.sub(r'[\u202f\u2002\xa0]', ' ', cleaned_text)

    # Fix hyphenated words at the end of lines
    cleaned_text = re.sub(r'-\s+', '', cleaned_text)

    # Replace multiple spaces with a single space
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

    # Strip leading/trailing whitespace
    cleaned_text = cleaned_text.strip()

    return cleaned_text

#@title Full Article Text - PubMed
def text_dictionary(article_html):
  sections_dict = {}
  current_h2 = None

  for header in article_html.find_all(['h2', 'h3']):
      section_name = header.text.strip()  # Section name from the header text
      section_text = []  # Initialize an empty list for the section text
      if header.name == 'h2':
          current_h2 = section_name
          sections_dict[current_h2] = {'text': '', 'subsections': {}}
      elif header.name == 'h3' and current_h2:
          # Ensure there is a current H2 to nest this H3 under
          if 'subsections' not in sections_dict[current_h2]:
              sections_dict[current_h2]['subsections'] = {}

      next_element = header.find_next_sibling()

      # Continue until there are no more siblings or another header is found
      while next_element and next_element.name not in ['h2', 'h3']:
          if next_element.name == 'p':
              section_text.append(next_element.text.strip())
          next_element = next_element.find_next_sibling()

      # Combine the text and store in the appropriate place in the dictionary
      if header.name == 'h2':
          sections_dict[current_h2]['text'] = ' '.join(section_text)
      elif header.name == 'h3' and current_h2:
          sections_dict[current_h2]['subsections'][section_name] = ' '.join(section_text)
  return sections_dict

def process_table(table):
  processed_table = []
  rowspan_placeholders = [0] * 100  # Assuming max 100 columns, adjust as needed

  for row in table.find_all('tr'):
    processed_row = []
    cells = row.find_all(['th', 'td'])
    cell_idx = 0

    for cell in cells:
      while rowspan_placeholders[cell_idx] > 0:
        processed_row.append('')
        rowspan_placeholders[cell_idx] -= 1
        cell_idx += 1

      cell_text = cell.get_text(strip=True)
      processed_row.append(cell_text)

      colspan = int(cell.get('colspan', 1))
      for _ in range(1, colspan):
        processed_row.append('')
        cell_idx += 1

      rowspan = int(cell.get('rowspan', 1))
      if rowspan > 1:
        for offset in range(colspan):
          rowspan_placeholders[cell_idx - colspan + 1 + offset] = rowspan - 1
      cell_idx += 1
    processed_table.append(processed_row)
  return processed_table

def table_dictionary(article_html):
  tables = article_html.find_all('table', {'class': 'default_table'})

  # Store each table's dataframes
  dataframes = []

  # Iterate over each table found
  for table in tables:
      processed_table = process_table(table)
      df = pd.DataFrame(processed_table)
      dataframes.append(df)

  tables_dict = {}
  # Iterate through the list of DataFrames and save each into the dictionary
  for index, df in enumerate(dataframes, start=1):
      # Use a formatted string for the key to identify each table
      key = f"Table {index}"
      tables_dict[key] = df.to_string(index=False)
  return tables_dict

# Function to check if all required titles are in the list of strings, case-insensitively, and output matched titles
def section_match(list_of_strings, required_titles):
    # Convert all strings in the list to lower case and keep original strings in a dictionary for lookup
    lower_to_original = {title.lower(): title for title in list_of_strings}

    # Check if all required titles are present (case-insensitively) in the list
    all_titles_present = all(title.lower() in lower_to_original for title in required_titles)

    if all_titles_present:
        # If all required titles are present, collect the matched titles from the list
        sections_to_pull = [lower_to_original[title.lower()] for title in required_titles if title.lower() in lower_to_original]
        return sections_to_pull
    else:
        ### Identify the most important columns via LLM
        list_of_strings_str = ', '.join(list_of_strings)

        relevant_sections_response = client.chat.completions.create(
          model="gpt-3.5-turbo-0125",
          messages=[
            {
              "role": "system",
              "content": """Of the given list of sections within the research paper, choose which sections most closely map to an "Abstract", "Background", "Methods", "Results", "Discussion", "Conclusion", "Sources of Funding", "Conflicts of Interest", "References", and "Table" section? Only use section names provide in the list to map. Multiple sections can map to each category. If there are multiple sections, separate them using the character |.
                Format must follow:
                Abstract: <sections>
                Background: <sections>
                Methods: <sections>
                Results: <sections>
                Discussion: <sections>
                Conclusion: <sections>
                Sources of Funding: <sections>
                Conflicts of Interest: <sections>
                Table: <sections>
                References: <sections>
              """
            },
            {
              "role": "user",
              "content": list_of_strings_str
            }
          ],
          temperature=0.1,
          top_p=1
        )

        relevant_sections = relevant_sections_response.choices[0].message.content
        print(relevant_sections)
        print("\n")

        # Split the text into lines
        lines = relevant_sections.split('\n')

        # Initialize a set to hold distinct values
        sections_to_pull = []

        # Iterate over each line
        for line in lines:
            # Check if line contains ':'
            if ':' in line:
                # Split the line at ':' and strip whitespace from the result
                value = line.split(':', 1)[1].strip()
                # Process and add the values
                # Split the value by ',' and strip whitespace and quotes
                split_values = [val.strip(" '") for val in value.split("',")]
                # Add each trimmed value to the set of distinct values
                for val in split_values:
                    if val not in sections_to_pull:
                        sections_to_pull.append(val)
        return sections_to_pull

### Identify the most relevant sections via LLM
def relevant_sections_capture(article_text):
  available_cols = article_text.keys()
  sections_of_interest = ["Abstract", "Background", "Results", "Conclusions", "Discussion", "Methods", "Source of Funding", "Conflicts of Interest", "Table", "References"]
  relevant_sections_identified = section_match(available_cols, sections_of_interest)
  true_sections_to_pull = [element for element in relevant_sections_identified if element in available_cols and "None" not in element]
  return true_sections_to_pull


def get_full_text_pubmed(article_json):
  url = "https://www.ncbi.nlm.nih.gov/pmc/articles/" + article_json['PMCID'] + '/'
  headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
  response = requests.get(url, headers=headers)

  soup = BeautifulSoup(response.content, 'html.parser')
  sections_dict = text_dictionary(soup)
  tables_dict = table_dictionary(soup)
  full_text_article = sections_dict | tables_dict

  sections_to_pull = relevant_sections_capture(sections_dict)

  concat_sections = ""

  for i in range(len(sections_to_pull)):
    title_str = sections_to_pull[i]
    if title_str == 'References':
      text_str = str(sections_dict[title_str])
      section_cleaned = '[' + title_str + '] ' + text_str
    else:
      text_str = sections_dict[title_str]['text']
      subsection_str = str(sections_dict[title_str]['subsections'])
      section_cleaned = '[' + title_str + '] ' + text_str + ' ' + str(subsection_str)
    concat_sections += section_cleaned

  article_content = concat_sections + ' ' + str(tables_dict)
  return article_content

#@title Full Article Text - Elsevier
def rank_links_by_preference(links_dict, preferred_sources):
    """
    Searches through the links dictionary to find and return the URL
    associated with the highest priority source available based on substring matching.

    :param links_dict: Dictionary with sources as keys and URLs as values.
    :param preferred_sources: List of source keywords in order of preference.
    :return: The URL of the highest priority source found, or None if none found.
    """
    for preferred_source in preferred_sources:
        for key in links_dict.keys():
            if preferred_source.lower() in key.lower():  # Case-insensitive matching
                return links_dict[key]
    return None

def get_preferred_link(url):
  links_dict = all_full_text_options(url)
  preferred_sources = ["Elsevier", "Springer", "JAMA", "Silverchair Information Systems", "Wiley",  "MDPI", "Taylor & Francis", "Cambridge University Press"]

  # Call the function with the dictionary and the list of preferred sources
  preferred_link = rank_links_by_preference(links_dict, preferred_sources)
  return preferred_link

def extract_pii(url):
    # Use regular expression to find the PII in the URL
    match = re.search(r'/pii/([^/]+)', url)
    if match:
        # Return the PII if found
        return match.group(1)
    else:
        # Return None if no PII is found in the URL
        return None

def get_full_text_elsevier(pii):
    api_key = os.getenv('ELSEVIER_API_KEY')
    if not api_key:
        raise ValueError("API key is not set in the environment variables.")

    url = f"https://api.elsevier.com/content/article/pii/{pii}"
    headers = {
        "X-ELS-APIKey": api_key,
        "Accept": "application/json"
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()  # Returns the full article text in JSON format
    else:
        return response.status_code, response.text  # Returns the error status and message
    
#@title Full Article Text - Springer
def extract_doi_springer(url):
    """
    Extracts the DOI from a Springer article URL.

    Parameters:
        url (str): The URL of the Springer article.

    Returns:
        str: The extracted DOI or an informative error message if DOI cannot be found.
    """
    # Define the base part of the Springer URL that precedes the DOI
    base_url = "https://link.springer.com/article/"

    # Extract the DOI by splitting the URL on the base URL and taking the latter part
    doi = url.split(base_url)[-1]
    return doi


def get_full_text_springer(url):
    """
    Attempts to fetch the full text PDF of a Springer article.

    Parameters:
        doi (str): The DOI of the article.
        api_key (str): Your Springer API key.

    Returns:
        Either the PDF content as bytes or an error message if the download fails.
    """
    # Retrieve the API key from the environment variable
    api_key = os.getenv('SPRINGER_API_KEY')
    doi = extract_doi_springer(url)
    if not api_key:
        return {"error": "API key is not set in the environment variables"}

    # URL to the Springer API endpoint for accessing article metadata
    url = f"https://link.springer.com/content/pdf/{doi}.pdf"

    # Custom headers for the request, including your API key
    headers = {
        'Accept': 'application/pdf',
        'X-API-Key': api_key
    }

    # Make the request for the full text PDF
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        # Attempt to convert the PDF content to text using PyMuPDF
        try:
            # Load PDF from bytes
            document = fitz.open("pdf", response.content)
            text = ""
            # Extract text from each page
            for page in document:
                text += page.get_text()
            document.close()
            return text
        except Exception as e:
            return {"error": "Failed to convert PDF to text", "message": str(e)}
    else:
        # Return an error with the status code
        return {"error": "Failed to fetch full text", "status_code": response.status_code}

#@title Full Article Text - JAMA
def get_full_text_jama(url):
    """
    Fetches and parses all text from the provided URL of a JAMA article, including paragraphs and headers.

    Parameters:
        url (str): The URL of the JAMA article.

    Returns:
        str: All extracted text from the article, including headers and paragraphs.
    """
    # Headers to mimic a browser visit
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}

    # Send a GET request to the URL with the headers
    response = requests.get(url, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all text within paragraph and header tags
        content_tags = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])

        # Extract text from each tag and combine into a single string, ensuring order is preserved
        article_text = '\n'.join(tag.text for tag in content_tags)

        return article_text
    else:
        return "Failed to retrieve the webpage. Status code: {}".format(response.status_code)
    
#@title Full Article Text - Wiley
def extract_doi_wiley(url):
    """
    Extracts the DOI from a given Wiley URL that contains 'doi/' followed by the DOI.

    Parameters:
        url (str): The Wiley URL containing the DOI.

    Returns:
        str: The extracted DOI or an empty string if DOI is not found.
    """
    # Check if 'doi/' is part of the URL
    if 'doi/' in url:
        # Split the URL at 'doi/' and return the part after it
        try:
            return url.split('doi/')[1]
        except IndexError:
            return ""  # Return empty string if 'doi/' is at the end with no DOI after
    else:
        return ""  # Return empty string if 'doi/' is not in the URL


def get_full_text_wiley(url):
    """
    Fetches the full text of an article from Wiley's API given a DOI.

    Parameters:
        doi (str): The DOI of the article.
        client_token (str): Wiley-TDM-Client-Token for authorized access.

    Returns:
        str: The full text of the article if successful, else an error message.
    """
    # URL encoding the DOI as it appears in the example URL format
    doi = extract_doi_wiley(url)
    encoded_doi = requests.utils.quote(doi)
    wiley_url = f"https://api.wiley.com/onlinelibrary/tdm/v1/articles/{encoded_doi}"
    client_token = os.getenv('WILEY_CLIENT_TOKEN')

    headers = {
        'Wiley-TDM-Client-Token': client_token,
        'Accept': 'application/pdf'  # Assuming PDF is the required format; adjust if XML or others needed
    }

    # Making the GET request
    response = requests.get(wiley_url, headers=headers, allow_redirects=True)

    # Check if the request was successful
    if response.status_code == 200:
        document = fitz.open("pdf", response.content)
        text = ""
        # Extract text from each page and clean it up
        for page in document:
            page_text = page.get_text("text")
            cleaned_text = " ".join(line.strip() for line in page_text.splitlines())
            text += cleaned_text + " "
        document.close()
        return text
    else:
        return f"Failed to retrieve full text. Status code: {response.status_code}, Message: {response.text}"
    
#@title process_article
def process_article(article, user_query):
  """ Helper function for ThreadPoolExecutor"""

  try:
    ### Retrieve the abstract ###
    abstract = article["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]


    ### Clean-Up Abstract ###
    reconstructed_abstract = ""
    for element in abstract:
        label = element.attributes.get("Label", "")
        if reconstructed_abstract:
          reconstructed_abstract += "\n\n"
        if label:
          reconstructed_abstract += f"{label}:\n"
        reconstructed_abstract += str(element)


    ### Pointwise-Relevance of Article to Query ###
    relevance_response = client.chat.completions.create(
      model="gpt-3.5-turbo-0125",
      messages=[
        {
          "role": "system",
          "content": """You are an expert medical researcher who's task is to determine whether research articles and studies are relevant to the question or that may be useful to know for safety reasons.
          Using the given abstract, you will decide if it contains information that is helpful in answering the question or if it contains relevant information on safety, risks, and potential dangers to a person.
          Please answer with a yes/no only. If the article is about an animal (e.g. hamster, mice), you must answer with "no".
          """
        },
        {
          "role": "user",
          "content": f"""
          Question: {user_query}
          Abstract: {reconstructed_abstract}
          """
        }
      ],
      temperature=0.8,
      top_p=1
    )

    answer_relevance = relevance_response.choices[0].message.content
    first_word = answer_relevance.split()[0].strip(string.punctuation).lower()
    article_is_relevant = first_word not in {"no", "n"}



    ### Citation ###
    citation = generate_ama_citation(article)


    ### Article JSON ###
    title = article["MedlineCitation"]["Article"]["ArticleTitle"]
    url = (f"https://pubmed.ncbi.nlm.nih.gov/"
              f"{article['MedlineCitation']['PMID']}/")

    types_html = article['MedlineCitation']['Article']['PublicationTypeList']
    publication_types = []
    for pub_type in types_html:
      publication_types.append(str(pub_type))

    pmc_id = next((element for element in article['PubmedData']['ArticleIdList'] if element.attributes.get('IdType') == 'pmc'), None)

    article_json =  {
                      "title": title,
                      "publication_type": publication_types,
                      "url": url,
                      "abstract": reconstructed_abstract,
                      "is_relevant": article_is_relevant,
                      "citation": citation,
                      "PMID": str(article['MedlineCitation']['PMID']),
                      "PMCID": str(pmc_id)
                    }

    # print(title, ' --- ', url)

    if article_is_relevant:
      preferred_link = get_preferred_link(article_json['url'])

      ### Bring in Full Text, if PMC text Available ###
      if (article_json['PMCID'] != None) & (article_json['PMCID'] != "None"):
        article_content = get_full_text_pubmed(article_json)
        article_json["full_text"] = True
      elif preferred_link and "elsevier" in preferred_link:
        pii = extract_pii(preferred_link)
        article_data_json = get_full_text_elsevier(pii)
        if (article_data_json['full-text-retrieval-response']['coredata']['openaccess'] == 1) | (article_data_json['full-text-retrieval-response']['coredata']['openaccess'] == '1'):
          article_content = clean_extracted_text(str(article_data_json['full-text-retrieval-response']['originalText']))
          article_json["full_text"] = True
        else:
          article_content = reconstructed_abstract
          article_json["full_text"] = False
      elif preferred_link and "springer" in preferred_link:
        article_content = clean_extracted_text(str(get_full_text_springer(preferred_link)))
        article_json["full_text"] = True
      elif preferred_link and "jamanetwork" in preferred_link:
        article_content = clean_extracted_text(str(get_full_text_jama(preferred_link)))
        article_json["full_text"] = True
      elif preferred_link and "wiley" in preferred_link:
        article_content = clean_extracted_text(str(get_full_text_wiley(preferred_link)))
        article_json["full_text"] = True
      else:
        article_content = reconstructed_abstract
        article_json["full_text"] = False

      if len(article_content) > 1048576:
             article_content = article_content[:1044000]

      ### Summarize only the relevant articles and assess strength of work ###
      study_types = set(['Adaptive Clinical Trial',
                    'Case Reports',
                    'Clinical Study',
                    'Clinical Trial',
                    'Clinical Trial Protocol',
                    'Clinical Trial, Phase I',
                    'Clinical Trial, Phase II',
                    'Clinical Trial, Phase III',
                    'Clinical Trial, Phase IV',
                    'Clinical Trial, Veterinary',
                    'Comparative Study',
                    'Controlled Clinical Trial',
                    'Equivalence Trial',
                    'Evaluation Study',
                    # 'Journal Article',
                    'Multicenter Study',
                    'Observational Study',
                    'Observational Study, Veterinary',
                    'Pragmatic Clinical Trial',
                    'Preprint',
                    'Published Erratum',
                    'Randomized Controlled Trial',
                    'Randomized Controlled Trial, Veterinary',
                    'Technical Report',
                    'Twin Study',
                    'Validation Study'])

      article_type = set(article_json['publication_type'])

      if article_type.isdisjoint(study_types):
        # review type paper
        system_prompt_summarize = """Given the following literature review paper, extract the following information and summarize it, being technical, detailed, and specific, while also explaining concepts for a layman audience. Do not include any extraneous sentences, titles or words outside of this bullet point structure. As often as possible, directly include metrics and numbers (especially significance level, confidence intervals, t-test scores, effect size). Follow the instructions in the parantheses::
            1. Purpose (What is the review seeking to address or answer? What methods were used? If relevant and mentioned, include dosages.):
            2. Main Conclusions (What are the conclusions and main claims made? What are its implications?):
            3. Risks (Are there any risks mentioned (e.g. risk of addiction, risk of death)?):
            4. Benefits (Are there any benefits purported?):
            5. Search Methodology and Scope (What was the search strategy used to identify relevant literature? Assess the breadth and depth of the literature included. Is the scope clearly defined, and does it encompass relevant research in the field?):
            6. Selection Criteria (Evaluate the criteria used for selecting the studies included in the review. What types of studies were included and which were excluded? Were diverse perspectives incorporated? Are contradictory findings or alternative theories addressed?):
            7. Quality Assessment of Included Studies (Were quality assessment methods applied? How were the methodologies, results, and reliability of the studies assessed?):
            8. Synthesis and Analysis (Evaluate how the findings from different studies are synthesized and analyzed. Is there a clear structure and methodology for synthesizing the literature? What statistical tests were used and for what purpose? Include all mention of statistical metrics and interpret what they mean, especially significance levels/p-values, confidence intervals, t-test scores, or effect size):
            9. Sources of Funding or Conflict of Interest (Identify any sources of funding and possible conflicts of interest.):
            """
      else:
        # study type paper
        system_prompt_summarize = """Given the following research paper, extract only the following information enumerated below and summarize it, being technical, detailed, and specific, while also explaining concepts for a layman audience. Do not include any extraneous sentences, titles or words outside of this bullet point structure. As often as possible, directly include metrics and numbers (especially significance level, confidence intervals, t-test scores, effect size). Follow the instructions in the parantheses:
            1. Purpose & Design (What is the study seeking to address or answer? What methods were used? Were there any exclusions or considerations? Include dosages if mentioned.):
            2. Main Conclusions (What claims are made?):
            3. Risks (Are there any risks mentioned (e.g. risk of addiction, risk of death)?):
            4. Benefits (Are there any benefits purported?):
            5. Type of Study ((e.g. observational, randomized). If randomized, mention if it was placebo controlled or double-blinded.):
            6. Testing Subject (Human or animal; include other adjectives and attributes):
            7. Size of Study (May be written as "N="):
            8. Length of Experiment:
            9. Statistical Analysis of Results (What tests were conducted? Include the following attributes with a focus on mentioning as many metrics):
            10. Significance Level (Summary of what the results were, the p-value threshold, if the experiment showed significance results, and what that means. Mention as many significant p-value numbers as available.):
            11. Confidence Interval (May be expressed as a percentage):
            12. Effect Size (Did the study aim for a certain effect size? May be expressed as Cohen's d, Pearson's r, or SMD. Include % power if mentioned):
            13. Sources of Funding or Conflict of Interest (Identify any sources of funding and possible conflicts of interest.):
            """

      reliability_analysis_response = client.chat.completions.create(
        model="gpt-4o",
        messages = [
            {
                "role": "system",
                "content": system_prompt_summarize
            },
            {
                "role": "user",
                "content": f"Paper: {article_content}"
            }
        ],
        temperature=0.6,
        top_p=1
      )

      # Extract the generated summary
      answer_summary = reliability_analysis_response.choices[0].message.content
      article_json["summary"] = answer_summary

    return article_json
  except KeyError:
    print("No abstract provided")

#@title process_article_with_retry
def process_article_with_retry(article, user_query):
    try:
        return process_article(article, user_query)
    except Exception as e:
        print("Error processing article:", e, "- waiting 10 secs")
        time.sleep(10)
        print("Trying again")
        return process_article(article, user_query)
    
#@title reliability_analysis_processing
def reliability_analysis_processing(articles_to_process, user_query):
  # summaries_collected = []
  relevant_article_summaries = []
  irrelevant_article_summaries = []

  with ThreadPoolExecutor(max_workers=8) as executor:
      futures = [executor.submit(process_article_with_retry, article, user_query) for article in articles_to_process]
      for future in as_completed(futures):
          try:
              result = future.result()
              # Bucket articles as relevant vs irrelevant
              if result["is_relevant"]:
                  relevant_article_summaries.append(result)
                  print(result)
                  print('-----------------------------------------------------------')
              else:
                  irrelevant_article_summaries.append(result)
          except Exception as e:
              print("Error processing article:", e)
  return relevant_article_summaries, irrelevant_article_summaries

#@title dict_to_tuple
def dict_to_tuple(data):
    transformed_data = []
    for article_json in data:
        # Extracting the PMID
        pmid = article_json['PMID']

        # Creating a new dictionary with selected fields and renaming 'publication_type' to 'article_type'
        new_dict = {
            'url': article_json['url'],
            'PMID': pmid,
            'PMCID': article_json.get('PMCID'),  # Using .get() to handle cases where 'PMCID' might be missing
            'summary': article_json['summary'],
            'citation': article_json['citation'],
            'article_type': article_json['publication_type']  # Renaming 'publication_type' to 'article_type'
        }

        # Appending the tuple (PMID, new_dict) to the transformed data list
        transformed_data.append((pmid, json.dumps(new_dict)))
    return transformed_data

#@title upload_to_db
def upload_to_db(credentials, table_name, data):
    """
    Adds a new row to a specified MySQL table.

    Parameters:
    - credentials (str): Name of env file with host, port, user, password, database
    - table_name (str): Name of the table to insert the new row into.
    - data (str): Tuple with 2 elements (str, JSON file)
    """

    # Load Credentials
    load_dotenv(credentials)

    try:
        connection = mysql.connector.connect(
            host=os.getenv('host'),
            port=os.getenv('port'),
            user=os.getenv('user'),
            password=os.getenv('password'),
            database=os.getenv('database'))
        if connection.is_connected():
            print('Connected to MySQL database')

            # Create a Cursor Object
            cursor = connection.cursor()

            # Upsert
            query = f"INSERT INTO {table_name} (article_id, article_json) VALUES (%s, %s) ON DUPLICATE KEY UPDATE article_id = VALUES(article_id)"

            #Prepare the data for insertion
            prepared_data = [(item[0], item[1]) for item in data]

            # Executing the query
            cursor.executemany(query, prepared_data)

            # Committing the transaction
            connection.commit()
            print(f"Row inserted into {table_name}")

    except Error as e:
        print(f"Error: {e}")

    finally:
        # Step 6: Close the Connection
        if connection.is_connected():
            cursor.close()
            connection.close()
            print('MySQL connection is closed')

#@title write_articles_to_db
def write_articles_to_db(relevant_article_summaries, env_file):
  data_to_insert = dict_to_tuple(relevant_article_summaries)
  upload_to_db(env_file, 'reliabilityanalysis', data_to_insert)\
  
#@title Few Shot Examples - Final Synthesis
example_1_question = "Can increasing omega-3 fatty acid intake improve cognitive function and what are common fish-free sources suitable for vegetarians?"
example_1_response = '''Increasing omega-3 fatty acid intake has been studied for potential benefits to brain health and cognitive function. While omega-3s like docosahexaenoic acid (DHA) and eicosapentaenoic acid (EPA) are essential for brain health, evidence from clinical trials presents a nuanced picture.

**Varying Cognitive Effects Across Conditions and Populations**
* **Benefits in Early Cognitive Decline:** A comprehensive literature review suggests that omega-3 fatty acids, especially DHA, may help protect against mild cognitive impairment (MCI) and early Alzheimer's disease (AD). Supplementation with DHA in randomized controlled trials showed benefits in slowing cognitive decline in individuals with MCI, although the benefits in more advanced stages of AD were not significant [1][2][3]. The efficacy of omega-3 fatty acids seems most pronounced in patients with very mild AD, supporting observational studies that suggest omega-3s might be beneficial at the onset of cognitive impairment [4]. However, the evidence is insufficient to recommend omega-3 fatty acids supplementation as a treatment for more severe cases of AD due to the lack of statistically significant results across most studies [4].
* **Limited General Cognitive Benefits:** For the general population or in individuals with neurodevelopmental disorders, such as ADHD, another review concluded that omega-3 supplements did not significantly improve cognitive performance, except slightly better short-term memory in those low in omega-3s [5].
* **Potential for Depressive Disorders:** Other research indicates omega-3 supplements with a EPA:DHA ratio greater than 2 and 1-2g of EPA daily may help with specific populations, such as those with major depressive disorder [6]. While not directly about cognitive function improvements, this highlights omega-3s' importance for mental health, which can be intricately linked to cognitive health.

**Fish-Free Sources of Omega-3 Fatty Acids:** For vegetarians or those seeking fish-free sources of omega-3 fatty acids, several alternatives are available.
* **ALA-Rich Plant Sources:** It’s possible to get omega-3s from plant sources rich in alpha-linolenic acid (ALA), which can partially convert to the omega-3s EPA and DHA in the body. Good ALA sources are flaxseeds, chia seeds, walnuts, and their oils [7][8]. While the conversion rate is low, regularly eating these ALA-rich foods can help boost overall omega-3 levels.
* **Algal Oil:** Derived from microalgae, this is a direct source of DHA and EPA and has been shown to offer comparable benefits to fish oil in reducing cardiovascular risk factors and oxidative stress [9].

**Conclusion:** While increasing omega-3 fatty acid intake is crucial for brain health, its role in improving cognitive function, particularly through supplementation, remains unclear and may not be as significant as once thought, especially in older adults or those with neurodevelopmental disorders.  Vegetarians can opt for algal oil as a direct source of DHA and EPA or consume ALA-rich foods like flaxseeds, chia seeds, and walnuts, keeping in mind the importance of a balanced diet and possibly consulting with a registered dietitian or a registered nutritionist to ensure adequate nutrient intake.

References:
[1] Welty FK. Omega-3 fatty acids and cognitive function. Current opinion in lipidology. Feb 01, 2023;34(1):12-21.
[2] Sala-Vila A, Fleming J, Kris-Etherton P, Ros E. Impact of α-Linolenic Acid, the Vegetable ω-3 Fatty Acid, on Cardiovascular Disease and Cognition. Advances in nutrition (Bethesda, Md.). Oct 02, 2022;13(5):1584-1602.
[3] Wysoczański T, Sokoła-Wysoczańska E, Pękala J, Lochyński S, Czyż K, Bodkowski R, Herbinger G, Patkowska-Sokoła B, Librowski T. Omega-3 Fatty Acids and their Role in Central Nervous System - A Review. Current medicinal chemistry. ;23(8):816-31.
[4] Canhada S, Castro K, Perry IS, Luft VC. Omega-3 fatty acids' supplementation in Alzheimer's disease: A systematic review. Nutritional neuroscience. ;21(8):529-538.
[5] Burckhardt M, Herke M, Wustmann T, Watzke S, Langer G, Fink A. Omega-3 fatty acids for the treatment of dementia. Cochrane Database Syst Rev. 2016;4(4):CD009002. Published 2016 Apr 11. doi:10.1002/14651858.CD009002.pub3
[6] Guu TW, Mischoulon D, Sarris J, et al. International Society for Nutritional Psychiatry Research Practice Guidelines for Omega-3 Fatty Acids in the Treatment of Major Depressive Disorder. Psychother Psychosom. 2019;88(5):263-273. doi:10.1159/000502652
[7] Doughman SD, Krupanidhi S, Sanjeevi CB. Omega-3 fatty acids for nutrition and medicine: considering microalgae oil as a vegetarian source of EPA and DHA. Current diabetes reviews. ;3(3):198-203.
[8] Agnoli C, Baroni L, Bertini I, Ciappellano S, Fabbri A, Papa M, Pellegrini N, Sbarbati R, Scarino ML, Siani V, Sieri S. Position paper on vegetarian diets from the working group of the Italian Society of Human Nutrition. Nutrition, metabolism, and cardiovascular diseases: NMCD. ;27(12):1037-1052.
[9] Salman HB, Salman MA, Yildiz Akal E. The effect of omega-3 fatty acid supplementation on weight loss and cognitive function in overweight or obese individuals on weight-loss diet. Nutricion hospitalaria. Aug 25, 2022;39(4):803-813.
'''

example_2_question = "What are the scientifically proven benefits of taking ginseng supplements?"
example_2_response = '''The scientifically proven benefits of taking ginseng supplements include improvements in cognitive function, physical performance, energy levels, immune system strength, and potential benefits in treating and managing chronic fatigue, diabetes, and its complications. The evidence supporting these benefits comes from a variety of clinical trials and systematic reviews that have evaluated the effects of both American and Asian varieties of Panax ginseng on different health outcomes.

* **Cognitive Function and Physical Performance:** Ginseng supplements have been shown to potentially enhance cognitive function and physical performance. Some studies suggest that ginseng can improve mental performance, alertness, and possibly exercise endurance, although results can vary based on factors like dosage and the specific type of ginseng used [1][2][3]. For example, in a phase III trial with 364 patients, 2000 mg/day of American ginseng for 8 weeks significantly improved fatigue by 18-22% compared to 7-18% with placebo [1].
* **Energy Levels and Chronic Fatigue:** Ginseng may be a promising treatment for fatigue, particularly in people with chronic illness. Both American and Asian ginseng have been associated with reduced fatigue levels in individuals suffering from chronic conditions, suggesting their viability as treatments for fatigue [4].
* **Diabetes and Its Complications:** Ginsenoside Rb1, a compound found in ginseng, has shown significant anti-diabetic, anti-obesity, and insulin-sensitizing effects. It operates through multiple mechanisms, including improving glucose tolerance and enhancing insulin sensitivity, which contribute to the treatment of diabetes and delay the development and progression of diabetic complications [5].
* **Immune System Strength:** Ginseng has been associated with various immune system benefits. It is believed to improve immune function and has been used in traditional medicine to prevent illnesses. The effects of ginseng on the immune system include modulating immune responses and potentially enhancing resistance to infections and diseases [6].
* **Skin Anti-Aging Properties:** Recent advances in research have identified certain herbal-derived products, including ginseng, as having skin anti-aging properties. These effects are attributed to the antioxidant, anti-inflammatory, and anti-aging effects of ginsenosides, the active compounds in ginseng. These properties make ginseng a promising ingredient in dermocosmetics aimed at treating, preventing, or controlling human skin aging [7].

**Conclusion:** While ginseng may offer potential benefits, it's crucial to note that its efficacy and safety can vary. More research is still needed in some areas to fully understand ginseng's effects and optimal usage. Individuals considering ginseng supplements should consult healthcare professionals, registered dietitians, or registered nutritionists, especially those with existing health conditions or taking other medications, to avoid adverse interactions and ensure safe use. Ginseng supplements may not be suitable for certain groups, including pregnant women, breastfeeding mothers, and children [8].

References:
[1] Arring NM, Barton DL, Brooks T, Zick SM. Integrative Therapies for Cancer-Related Fatigue. Cancer journal (Sudbury, Mass.). ;25(5):349-356.
[2] Roe AL, Venkataraman A. The Safety and Efficacy of Botanicals with Nootropic Effects. Current neuropharmacology. ;19(9):1442-1467.
[3] Arring NM, Millstine D, Marks LA, Nail LM. Ginseng as a Treatment for Fatigue: A Systematic Review. Journal of alternative and complementary medicine (New York, N.Y.). ;24(7):624-633.
[4] Zhou P, Xie W, He S, Sun Y, Meng X, Sun G, Sun X. Ginsenoside Rb1 as an Anti-Diabetic Agent and Its Underlying Mechanism Analysis. Cells. Feb 28, 2019;8(3):.
[5] Costa EF, Magalhães WV, Di Stasi LC. Recent Advances in Herbal-Derived Products with Skin Anti-Aging Properties and Cosmetic Applications. Molecules (Basel, Switzerland). Nov 03, 2022;27(21):.
[6] Kim JH, Kim DH, Jo S, Cho MJ, Cho YR, Lee YJ, Byun S. Immunomodulatory functional foods and their molecular mechanisms. Experimental & molecular medicine. ;54(1):1-11.
[7] Mancuso C, Santangelo R. Panax ginseng and Panax quinquefolius: From pharmacology to toxicology. Food and chemical toxicology : an international journal published for the British Industrial Biological Research Association. ;107(Pt A):362-372.
[8] Malík M, Tlustoš P. Nootropic Herbs, Shrubs, and Trees as Potential Cognitive Enhancers. Plants (Basel, Switzerland). Mar 18, 2023;12(6):.
'''

disclaimer = """
DietNerd is an exploratory tool designed to enrich your conversations with a registered dietitian or registered dietitian nutritionist, who can then review your profile before providing recommendations.
Please be aware that the insights provided by DietNerd may not fully take into consideration all potential medication interactions or pre-existing conditions.
To find a local expert near you, use this website: https://www.eatright.org/find-a-nutrition-expert
"""

def generate_final_response(all_relevant_articles, query):
  system_prompt_controversy_overview = """
      You are an expert in evaluating research articles and summarizing findings based on the strength of evidence. Your task is to review multiple research summaries using ONLY the provided Evidence and Claims and use those that have strong evidence to answer the user's question. You must choose at least 6 articles and use only the information in these articles to answer the question. Strong evidence means the research is well-conducted, peer-reviewed, and widely accepted in the scientific community. Provide a direct, research-backed answer to the question and focus on identifying the pros and cons of the topic in question. The answer should highlight when there are potential risks or dangers present.
      If the user question is dangeorus, harmful, or malicious, absolutely do not offer advice or strategies and absolutely do not address the pros, benefits, or potential results/outcomes. You must only focus on deterring this behavior, addressing the risks, and offering safe alternatives. The answer should also try to include as many different demographics as possible. Absolutely NO animal studies should be referenced or included in the final response. Mention dosage amounts when the information is available. Medical terms and technical concepts must be explained to a layman audience. Be sure to emphasize that you should always go and see a registered dietitian or a registered dietitian nutritionist.
      Articles must be cited in-line in Vancouver style using brackets. References listed must be numerically listed using brackets. Include section titles like "Conclusion" and organize sections as a bulleted list using an asterisk. List each and every one of the cited articles mentioned at the end using the citations in Evidence and Claims. Do not list duplicate references.

      The output must follow this format:
      <summary_of_evidence>

      References:
      [1] <citation_1>
      [2] <citation_2>
      [3] <citation_3>

      Here are some examples:

      User: {example_1_question}
      AI: {example_1_response}

      User: {example_2_question}
      AI: {example_2_response}
      """

  # Define the human prompt
  human_prompt = f"""
      Evidence and Claims: {all_relevant_articles}
      User Question: {query}
  """

  output_response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages = [
      {
          "role": "system",
          "content": system_prompt_controversy_overview
      },
      {
          "role": "user",
          "content": human_prompt
      }
  ],
    temperature=0.5,
    top_p=1
  )

  output = output_response.choices[0].message.content
  final_output = output + "\n" + disclaimer
  return final_output

#### Functions to write output to database

def split_end_output(end_output: str):
    # Remove substring "disclaimer" ignoring line breaks
    cleaned_output = end_output.replace(disclaimer, "")
    
    # Split the modified cleaned_output into main response and citations
    main_response, citations_section = cleaned_output.split("References:", 1)
    
    # Extract individual citations
    citations = re.split(r'\n{1,2}', citations_section.strip())
    
    return main_response.strip(), citations


def clean_citation(citation: str):
    # Remove the citation number (e.g., [1]) from the citation
    cleaned_citation = re.sub(r'^\[\d+\]\s*', '', citation).strip()
    return cleaned_citation

def parse_str(input_string: str) -> str:
    # Replace double periods with a single period
    cleaned_string = input_string.replace("..", ".")

    # Remove unnecessary semicolons and empty parentheses
    cleaned_string = re.sub(r';\s*', ' ', cleaned_string)
    cleaned_string = re.sub(r'\(\s*\)', '', cleaned_string)

    # Remove extra spaces that may result from the above replacements
    cleaned_string = re.sub(r'\s+', ' ', cleaned_string).strip()
    return cleaned_string


def upload_to_final(credentials, question, obj):
    load_dotenv(credentials)

    try:
        connection = mysql.connector.connect(
            host=os.getenv('host'),
            port=os.getenv('port'),
            user=os.getenv('user'),
            password=os.getenv('password'),
            database=os.getenv('database'))
        if connection.is_connected():
            # Create a Cursor Object
            cursor = connection.cursor()
            json_string = json.dumps(obj)
            # Upsert
            query = f"INSERT INTO finaloutputs (question, answer) VALUES (%s, %s) ON DUPLICATE KEY UPDATE question = VALUES(question)"
            # Executing the query
            cursor.execute(query, (question, json_string))
            # Committing the transaction
            connection.commit()
    except Error as e:
        print(f"Error: {e}")
    finally:
        # Step 6: Close the Connection
        if connection.is_connected():
            cursor.close()
            connection.close()
            #print('MySQL connection is closed')


def match_citations_with_articles(citations, articles):
    # Create a dictionary to store the matched citation information
    citation_dict = {}
    for citation in citations:
        trim_citation = clean_citation(citation)
        #print(trim_citation)
        for article in articles:
            # #print("ART: " + article["citation"])
            to_match = parse_str(article["citation"])[20:70]
            if to_match in citation:
                #print("Match Found")
                citation_dict[citation] = {
                    "PMID": article["PMID"],
                    "PMCID": article["PMCID"],
                    "URL": article["url"],
                    "Summary": article["summary"]
                }
    return citation_dict

def write_output_to_db(user_query, final_output, all_relevant_articles, total_runtime, env_file):
  return_obj = {
        "end_output": final_output,
        "relevent_articles": all_relevant_articles,
        "total_runtime": total_runtime
      }


  main_output, citations = split_end_output(return_obj["end_output"])

  relevent_articles = return_obj.get("relevent_articles", [])
  updated_citations = match_citations_with_articles(citations, all_relevant_articles)


  return_obj["end_output"] = final_output
  return_obj["citations_obj"] = updated_citations
  return_obj["citations"] = citations

  # with open("output.json", "w") as f:
  #     json.dump(return_obj, f, indent=4)

  upload_to_final(env_file, user_query, return_obj)
  return return_obj



def generate(user_query):
    env = 'ATT81274.env'
    question_validity = determine_question_validity(user_query)

    if question_validity == 'False - Meal Plan/Recipe':
        final_output = ("I'm sorry, I cannot help you with this question. For any questions or advice around meal planning or recipes, please speak to a registered dietitian or registered dietitian nutritionist.\n"
                        "To find a local expert near you, use this website: https://www.eatright.org/find-a-nutrition-expert.")
        print(final_output)
    elif question_validity == 'False - Animal':
        final_output = ("I'm sorry, I cannot help you with this question. For any questions regarding an animal, please speak to a veterinarian.\n"
                        "To find a local expert near you, use this website: https://vetlocator.com/.")
        print(final_output)
    else:
        # Query Generation
        start_poc = time.time()
        general_query, query_contention, query_list = query_generation(user_query)
        end_poc = time.time()

        # Article Retrieval
        start_api = time.time()
        deduplicated_articles_collected = collect_articles(query_list)
        end_api = time.time()

        # Article Match
        start_processing = time.time()
        reliability_analysis_df = connect_to_reliability_analysis_db()
        matched_articles, articles_to_process = article_matching(deduplicated_articles_collected, reliability_analysis_df)

        # Article Processing
        relevant_article_summaries, irrelevant_article_summaries = reliability_analysis_processing(articles_to_process, user_query)

        # Write Processed Articles to DB
        write_articles_to_db(relevant_article_summaries, env)

        # All Final Eligible Articles to Reference
        all_relevant_articles = list(itertools.chain(relevant_article_summaries, matched_articles))
        end_processing = time.time()

        # Final Output
        start_output = time.time()
        final_output = generate_final_response(all_relevant_articles, user_query)
        end_output = time.time()

        # Runtime Calculation
        poc_duration = end_poc - start_poc
        api_duration = end_api - start_api
        article_processing_duration = end_processing - start_processing
        final_output_duration = end_output - start_output
        total_runtime = poc_duration + api_duration + article_processing_duration + final_output_duration

        # Write Output to DB
        write_output_to_db(user_query, final_output, all_relevant_articles, total_runtime, env)
        end_output = time.time()
        return final_output
    



