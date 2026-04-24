export USE_AZURE_OPENAI_AAD=1
export GPT_ENDPOINT=https://societalllm.openai.azure.com/
export AZURE_OPENAI_API_VERSION=2024-12-01-preview

uv run main.py \
--graph_size 16 \
--task coloring \
--rounds 8 \
--samples_per_graph_model 3 \
--graph_models ws \
--model gpt-5.4 \
--disable_chain_of_thought
