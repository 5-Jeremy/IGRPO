DATA=${DATA:-/data1/jcarleton/IGRPO}
save_path=${RAG_DATA_DIR:-$DATA/searchR1}

index_file=$save_path/e5_Flat.index
corpus_file=$save_path/wiki-18.jsonl
retriever_name=e5
retriever_path=${RETRIEVER_MODEL:-intfloat/e5-base-v2}

HF_HUB_OFFLINE=1

exec python examples/search/retriever/retrieval_server.py \
  --index_path "$index_file" \
  --corpus_path "$corpus_file" \
  --topk 3 \
  --retriever_name $retriever_name \
  --retriever_model "$retriever_path" \
  --faiss_gpu \
  --port "${RAG_PORT:-8000}"
