import enum
from collections import OrderedDict
from typing import List, Dict, Tuple, Union

import numpy as np
import pandas as pd
import streamlit as st
from langchain_community.graphs.neo4j_graph import Neo4jGraph
from langchain_community.vectorstores.neo4j_vector import Neo4jVector, SearchType
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from pandas import DataFrame

NEO4J_URI = st.secrets['RESUME_NEO4J_URI']
NEO4J_USERNAME = st.secrets['RESUME_NEO4J_USERNAME']
NEO4J_PASSWORD = st.secrets['RESUME_NEO4J_PASSWORD']

embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

retrievers = dict()

retrievers["retrieve_by_description"] = person_retriever = Neo4jVector.from_existing_index(
    embedding=embedding_model,
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    index_name='person_text_embedding',
    search_type=SearchType.HYBRID,
    keyword_index_name='person_full_text',
    retrieval_query='''RETURN "" AS text, score, {entityId: node.entityId, name: node.name, role: node.role, 
            description: node.description, hitData:{role:node.role, description:node.description}, score:score} AS metadata'''
)


@tool
def retrieve_by_description(prompt: str) -> List[Document]:
    """Retrieve resumes based on Person summary and description"""
    return person_retriever.similarity_search(prompt, k=10)


retrievers["retrieve_by_skills"] = skill_retriever = Neo4jVector.from_existing_index(
    embedding=embedding_model,
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    index_name='skill_text_embedding',
    retrieval_query='''WITH node AS skill, score
MATCH(skill)<-[:HAS_SKILL]-(person)
WITH skill.entityId AS skill, count(*) AS totalWithSkill, collect(person) AS people, score
UNWIND people AS person
WITH person, collect(skill) AS skills, sum(score) AS score ORDER BY score DESC LIMIT 10
RETURN "" AS text, score, {entityId: person.entityId, name: person.name, role: person.role, 
description: person.description, hitData:skills, score:score} AS metadata'''
)


@tool
def retrieve_by_skills(prompt: str) -> List[Document]:
    """Retrieve resumes based on skills or technical experience"""
    return skill_retriever.similarity_search(prompt, k=10)


retrievers["retrieve_by_positions"] = position_retriever = Neo4jVector.from_existing_index(
    embedding=embedding_model,
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    index_name='position_text_embedding',
    search_type=SearchType.HYBRID,
    keyword_index_name='position_full_text',
    retrieval_query='''WITH node AS position, score
MATCH (position)<-[:HAS_POSITION]-(person)
WITH person, 
    collect({title:position.title, company:position.company, location:position.location, startDate:position.startDate, 
    endDate:position.endDate,  description:position.description}) AS positions, 
    sum(score) AS score // currently not taking length of position into account
ORDER BY score DESC LIMIT 10
RETURN "" AS text, score, {entityId: person.entityId, name: person.name, role: person.role, 
description: person.description, hitData:positions, score:score} AS metadata'''
)


@tool
def retrieve_by_positions(prompt: str) -> List[Document]:
    """Retrieve resumes based on past job roles/positions"""
    return position_retriever.similarity_search(prompt, k=10)


tools = [retrieve_by_skills, retrieve_by_positions, retrieve_by_description]
llm_with_tools = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0).bind_tools(tools)


def format_doc(doc: Document) -> Dict:
    res = OrderedDict()
    res.update(doc.metadata)
    return res


def docs_to_df(docs) -> DataFrame:
    return pd.DataFrame.from_records([format_doc(d) for d in docs])


def retrieve_and_format(prompt: str, retrieval_type: str):
    res = retrievers[retrieval_type].similarity_search(prompt, k=10)
    res_df = docs_to_df(res)
    res_df['score'] = max_scale_vector(res_df['score'])
    res_df['retrievalType'] = retrieval_type
    res_df['subPrompt'] = prompt
    return res_df


def sum_normalize_vector(x):
    v = np.array(x)
    return v / sum(v)


def max_scale_vector(x):
    v = np.array(x)
    return v / max(v)


def weighted_rank_and_filter(dfs: List[DataFrame], weights=None) -> DataFrame:
    if weights is not None:
        w = sum_normalize_vector(weights)
        for i in range(len(dfs)):
            dfs[i].score = dfs[i].score * w[i]
    df = pd.concat(dfs)
    group_by_cols = [i for i in df.columns if i not in ['hitData', 'retrievalType', 'subPrompt', 'score']]
    res_df = (df.groupby(group_by_cols).agg({'hitData': list, 'retrievalType': list, 'score': sum})
              .reset_index()
              .sort_values('score', ascending=False))
    res_df = res_df.set_index('entityId', drop=False)
    return res_df


def format_hit_data_item(r_type: str, hit_data) -> str:
    res = ''
    if r_type == "retrieve_by_skills":
        res = f"#### Relevant Skills:  \n{', '.join([x for x in hit_data])}*  \n"
    elif r_type == "retrieve_by_positions":
        res = "#### Relevant Positions:  \n"
        for p in hit_data:
            res = (res + f"__{p['title']}__, __{p['company']}__:  \n" + f"*{p['startDate']} - {p['endDate']}*  \n" +
                   f"{p['description']}  \n")
    return res


def format_hit_data(row: Dict) -> List[str]:
    res = ''
    for i in range(len(row['retrievalType'])):
        res = res + format_hit_data_item(row['retrievalType'][i], row['hitData'][i])
    return res


class Ranking(BaseModel):
    entityIds: List[str]


ranking_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert recruiter and researcher for resumes from job aspirants. "
            "Take a look at the below listOfCandidates and create a top 10 ordered list of entityIds that best match "
            "the searchCriteria."
            "Do not create fictitious data or impute missing values."
            "Return only the entityIds. These uniquely identify each candidate in the listOfCandidates."
        ),
        ("human", "searchCriteria: {searchCriteria}"),
        ("human", "listOfCandidates: {listOfCandidates}"),
    ]
)

llm = ChatOpenAI(temperature=0, model_name="gpt-4-0125-preview")
chain = ranking_prompt | llm.with_structured_output(Ranking)


def retrieval_with_tools(prompt: str, top_k: int = None):
    dfs = []
    print('++++++++++++++==')
    tool_response = llm_with_tools.invoke(prompt).tool_calls
    print(tool_response)
    print('++++++++++++++==')
    for r in tool_response:
        dfs.append(retrieve_and_format(r['args']['prompt'], r['name']))
    res_df = weighted_rank_and_filter(dfs)
    #print('===========')
    #print(res_df)
    #ranking: Ranking = chain.invoke({'searchCriteria': prompt, 'listOfCandidates': res_df.to_dict('records')})
    #res_df = res_df.loc[ranking.entityIds, :]
    #print(ranking)
    #print('==============')
    if top_k is not None:
        res_df = res_df[:min(top_k, res_df.shape[0])]
    print(res_df)
    res_df['hitData'] = res_df.apply(format_hit_data, axis=1)
    return res_df.to_dict('records')
