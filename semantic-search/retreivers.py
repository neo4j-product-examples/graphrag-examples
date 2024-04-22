import enum
from collections import OrderedDict
from typing import List, Dict

import numpy as np
import pandas as pd
import streamlit as st
from langchain_community.graphs.neo4j_graph import Neo4jGraph
from langchain_community.vectorstores.neo4j_vector import Neo4jVector, SearchType
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pandas import DataFrame

NEO4J_URI = st.secrets['RESUME_NEO4J_URI']
NEO4J_USERNAME = st.secrets['RESUME_NEO4J_USERNAME']
NEO4J_PASSWORD = st.secrets['RESUME_NEO4J_PASSWORD']

embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

retrievers = dict()

retrievers['PERSON'] = Neo4jVector.from_existing_index(
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

retrievers['SKILL'] = Neo4jVector.from_existing_index(
    embedding=embedding_model,
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    index_name='skill_text_embedding',
    search_type=SearchType.HYBRID,
    keyword_index_name='skill_full_text',
    retrieval_query='''WITH node AS skill, score
MATCH(skill)<-[:HAS_SKILL]-(person)
WITH skill.entityId AS skill, count(*) AS totalWithSkill, collect(person) AS people, score
UNWIND people AS person
WITH person, collect(skill) AS skills, sum(score/totalWithSkill) AS score ORDER BY score DESC LIMIT 10
RETURN "" AS text, score, {entityId: person.entityId, name: person.name, role: person.role, 
description: person.description, hitData:skills, score:score} AS metadata'''
)

retrievers['POSITION'] = Neo4jVector.from_existing_index(
    embedding=embedding_model,
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    index_name='position_text_embedding',
    search_type=SearchType.HYBRID,
    keyword_index_name='position_full_text',
    retrieval_query='''MATCH (position)<-[:HAS_POSITION]-(person)
WITH person, collect(position{.*}) AS positions, sum(score) AS score // currently not taking length of position into account
ORDER BY score DESC LIMIT 10
RETURN "" AS text, score, {entityId: person.entityId, name: person.name, role: person.role, 
description: person.description, hitData:positions, score:score} AS metadata'''
)


def format_doc(doc: Document) -> Dict:
    res = OrderedDict()
    res.update(doc.metadata)
    return res


def docs_to_df(docs) -> DataFrame:
    return pd.DataFrame.from_records([format_doc(d) for d in docs])


def retrieve_and_format(prompt: str, store: Neo4jVector, retrieval_type: str):
    res = store.similarity_search(prompt, k=10)
    res_df = docs_to_df(res)
    res_df['score'] = max_scale_vector(res_df['score'])
    res_df['retrievalType'] = retrieval_type
    return res_df


def sum_normalize_vector(x):
    v = np.array(x)
    return v / sum(v)


def max_scale_vector(x):
    v = np.array(x)
    return v / max(v)


def weighted_rank_and_filter(dfs: List[DataFrame], weights, top_k=10) -> DataFrame:
    w = sum_normalize_vector(weights)
    for i in range(len(dfs)):
        dfs[i].score = dfs[i].score * w[i]
    return pd.concat(dfs).sort_values('score', ascending=False)[:top_k]


def format_hit_data(row: Dict) -> str:
    res = ''
    if row['retrievalType'] == "SKILL":
        res = f"#### Relevant Skills:  \n{', '.join([x for x in row['hitData']])}*"
    elif row['retrievalType'] == "POSITION":
        res = "#### Relevant Positions:  \n"
        for p in row['hitData']:
            res = (res + f"__{p['title']}__, __{p['company']}__:  \n" + f"*{p['startDate']} - {p['endDate']}*  \n" +
                   f"{p['description']}  \n")
    return res


def weighted_retrieval(prompt: str, weights, top_k=10):
    dfs = []
    for r_type, store in retrievers.items():
        dfs.append(retrieve_and_format(prompt, store, r_type))
    res_df = weighted_rank_and_filter(dfs, weights, top_k)
    print(res_df)
    res_df['hitData'] = res_df.apply(format_hit_data, axis=1)
    return res_df.to_dict('records')
