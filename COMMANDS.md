- To sync the merged graph into Neo4j and start the API:
  python pipeline.py --graph-checkpoint-dir .\checkpoints\clinical_dsm_merged --community-checkpoint-dir .\checkpoints\clinical_dsm_merged --write-neo4j
  uvicorn fastapi_app:app --reload
