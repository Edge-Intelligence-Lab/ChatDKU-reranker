import argparse
import os
from pathlib import Path
from typing import Dict, Any
from vllm import LLM, EngineArgs
from vllm.entrypoints.score_utils import ScoreMultiModalParam


queries = [
    {"text": "A woman playing with her dog on a beach at sunset."}
]

documents = [
    {"text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust."},
    {"image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
    {"text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust.", 
     "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"}
]


def format_document_to_score_param(doc_dict: Dict[str, Any]) -> ScoreMultiModalParam:
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


def main():
    parser = argparse.ArgumentParser(description="Offline Reranker with vLLM")
    parser.add_argument("--model-path", type=str, default="models/Qwen3-VL-Reranker-8B", help="Path to the reranker model")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type (e.g., bfloat16)")
    parser.add_argument("--template-path", type=str, default="vllm/examples/pooling/score/template/qwen3_vl_reranker.jinja", 
                        help="Path to chat template file")
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_path}...")
    
    engine_args = EngineArgs(
        model=args.model_path,
        runner="pooling",
        dtype=args.dtype,
        trust_remote_code=True,
        hf_overrides={
            "architectures": ["Qwen3VLForSequenceClassification"],
            "classifier_from_token": ["no", "yes"],
            "is_original_qwen3_reranker": True,
        },
    )
    
    llm = LLM(**vars(engine_args))
    
    template_path = Path(args.template_path)
    chat_template = template_path.read_text() if template_path.exists() else None
    
    for query_dict in queries:
        query_text = query_dict.get('text', '')
        print(f"\nQuery: {query_text}")
        
        scores = []
        for doc_dict in documents:
            doc_param = format_document_to_score_param(doc_dict)
            outputs = llm.score(query_text, doc_param, chat_template=chat_template)
            score = outputs[0].outputs.score
            scores.append(score)
        
        print(scores)


if __name__ == "__main__":
    main()

