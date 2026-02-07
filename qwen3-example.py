import argparse
import os
from pathlib import Path
from typing import Dict, Any, List
import requests
import json

queries = [
    {"text": "A woman playing with her dog on a beach at sunset."}
]

documents = [
    {"text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust."},
    {"image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
    {"text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust.", 
     "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"}
]


def format_document_to_score_param(doc_dict: Dict[str, Any]) -> Dict[str, Any]:
    content = []
    
    text = doc_dict.get('text')
    image = doc_dict.get('image')
    
    if text:
        content.append({
            "type": "text",
            "text": text
        })
    
    if image:
        image_url = image
        if isinstance(image, str) and not image.startswith(('http', 'https', 'oss')):
            abs_image_path = os.path.abspath(image)
            image_url = 'file://' + abs_image_path
        
        content.append({
            "type": "image_url",
            "image_url": {
                "url": image_url
            }
        })
    
    if not content:
        content.append({
            "type": "text",
            "text": ""
        })
    
    return {"content": content}


def call_vllm_rerank(
    base_url: str,
    model: str,
    query: str,
    docs_params: List[Dict[str, Any]],
    api_key: str = None,
) -> List[float]:
    """
    Call vLLM's /v1/rerank endpoint and return the scores in document order.
    Assumes vLLM was started with --task score so that /v1/rerank is available.
    """
    documents_payload = [json.dumps(p) for p in docs_params]

    payload = {
        "model": model,
        "query": query,
        "documents": documents_payload,
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    url = f"{base_url.rstrip('/')}/v1/rerank"
    resp = requests.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()
    print(json.dumps(data, indent=2))


    results = sorted(data["results"], key=lambda x: x["index"])
    scores = [r["relevance_score"] for r in results]
    return scores


def main():
    parser = argparse.ArgumentParser(description="Online Reranker with vLLM HTTP server")
    parser.add_argument("--base-url", type=str, default="http://localhost:6767",
                        help="Base URL of vLLM server (e.g., http://localhost:6767)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-Reranker-8B",
                        help="Model name as exposed by vLLM")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key if vLLM server enforces auth")
    args = parser.parse_args()

    for query_dict in queries:
        query_text = query_dict.get('text', '')
        print(f"\nQuery: {query_text}")

        docs_params = [format_document_to_score_param(d) for d in documents]
        scores = call_vllm_rerank(
            base_url=args.base_url,
            model=args.model,
            query=query_text,
            docs_params=docs_params,
            api_key=args.api_key,
        )
        print(scores)


if __name__ == "__main__":
    main()

