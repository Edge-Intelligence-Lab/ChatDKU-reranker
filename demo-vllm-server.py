from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:6767/v1/rerank", 
)

# Example: score a query–document pair using the reranker prompt format you chose
query = "A woman playing with her dog on a beach at sunset."
doc = "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset."

prompt = f"Query: {query}\nDocument: {doc}\nScore the relevance from 0 to 1."


resp = client.chat.completions.create(
    model="Qwen/Qwen3-VL-Reranker-8B",
    messages=[
        {"role": "system", "content": "You are a relevance scoring model."},
        {"role": "user", "content": prompt},
    ],
    temperature=0.5,
    max_tokens=10,
)

score_text = resp.choices[0].message.content
print("Score:", score_text)

