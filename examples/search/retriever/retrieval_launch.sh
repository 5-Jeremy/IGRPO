DATA=${DATA:-/scratch/user/sushil22_tamu.edu/projects/IGRPO/data}
save_path=$DATA/searchR1

index_file=$save_path/e5_Flat.index
corpus_file=$save_path/wiki-18.jsonl
retriever_name=e5
retriever_path=$DATA/Base_models/e5-base-v2

export CUDA_VISIBLE_DEVICES="4"

python examples/search/retriever/retrieval_server.py \
  --index_path $index_file \
  --corpus_path $corpus_file \
  --topk 3 \
  --retriever_name $retriever_name \
  --retriever_model $retriever_path \
  --faiss_gpu \
  --port 8008 \