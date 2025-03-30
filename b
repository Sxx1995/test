import csv
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ✅ 分类标签
categories = [
    "history", "geography", "person", "entertainment", "science",
    "politics", "technology", "economy", "education", "military",
    "sports", "transportation", "architecture", "art", "music",
    "film", "medicine", "nature", "religion", "language"
]

# ✅ 初始化 LLM 模型
model_name = "mistralai/Mistral-7B-Instruct-v0.1"  # 可替换成 llama, zephyr 等
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
chat = pipeline("text-generation", model=model, tokenizer=tokenizer)

# ✅ Prompt 构造
def build_prompt(text):
    return f"""You are an intelligent assistant. Your task is to classify the following Wikipedia paragraph into 3 categories from a predefined list.

Text:
\"\"\"
{text}
\"\"\"

Choose the top 3 categories from the following list:
{', '.join(categories)}

Respond with only the 3 most relevant categories, separated by commas."""

# ✅ 主函数：处理 CSV 并保存 JSON
def classify_from_csv(input_csv, output_json, text_column="text", max_items=None):
    results = []

    with open(input_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(tqdm(reader, desc="Classifying")):
            if max_items and i >= max_items:
                break
            text = row[text_column].strip()
            prompt = build_prompt(text)
            try:
                response = chat(prompt, max_new_tokens=100, do_sample=False, temperature=0.0)[0]["generated_text"]
                tags = response.strip().split("\n")[-1].strip().split(",")
                tags = [tag.strip().lower() for tag in tags if tag.strip().lower() in categories]
            except Exception as e:
                print(f"Error in row {i}: {e}")
                tags = []

            results.append({
                "text": text,
                "tags": tags[:3]
            })

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved {len(results)} entries to {output_json}")

# ✅ 调用示例（最多处理 100 条）
classify_from_csv("wikipedia_snippets.csv", "classified_wiki.json", text_column="text", max_items=100)
