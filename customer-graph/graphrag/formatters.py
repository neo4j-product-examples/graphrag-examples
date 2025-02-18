from neo4j_graphrag.types import RetrieverResultItem
import ast
from neo4j import Record


def node_record_formatter(record: Record) -> RetrieverResultItem:
    #set up metadata    
    metadata = {"score": record.get("score"), "nodeLabels": record.get("nodeLabels"), "id": record.get("id")}

    #Reformatting: node -> to_string -> to_dict
    node = str(record.get("node"))  #entire node as string
    node_as_dict = ast.literal_eval(node)  #convert to dict

    return RetrieverResultItem(content=node_as_dict, metadata=metadata)


def my_vector_search_excerpt_record_formatter(record: Record) -> RetrieverResultItem:
    #set up metadata    
    metadata = {"contract_id": record.get("contract_id"), "nodeLabels": ['Excerpt', 'Agreement', 'ContractClause']}

    #Reformatting: get individual fields from the RETURN stattement. 
    #RETURN a.name as agreement_name, a.contract_id as contract_id, cc.type as clause_type, node.text as exceprt
    result_dict = {}
    result_dict['agreement_name'] = record.get("agreement_name")
    result_dict['contract_id'] = record.get("contract_id")
    result_dict['clause_type'] = record.get("clause_type")
    result_dict['excerpt'] = record.get("excerpt")

    return RetrieverResultItem(content=result_dict, metadata=metadata)
