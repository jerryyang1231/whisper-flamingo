import json
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# **設定 JSON 檔案路徑**
json_filepath = "/share/nas169/jerryyang/corpus/ml-superb/eng/eng_train.json"  # 你的 JSON 檔案

# **讀取 JSON 檔案**
with open(json_filepath, "r", encoding="utf-8") as f:
    data = json.load(f)

# **載入 SeamlessM4T 模型**
model_name = "facebook/seamless-m4t-v2-large"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

# 設置源語言和目標語言
tokenizer.src_lang = "eng"

# **翻譯函數**
def translate_text(text, src_lang, tgt_lang):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, tgt_lang=tgt_lang)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# **對 JSON 裡的 text 進行翻譯**
for entry in tqdm(data, desc="Translation Progress", unit="sentence"):
    original_text = entry["text"].replace(f"[{entry['language']}] ", "")  # 移除語言標籤
    translated_text = translate_text(original_text, src_lang=entry["language"], tgt_lang="cmn")
    entry["translated_text"] = translated_text  # 加入翻譯結果

# **儲存更新後的 JSON**
updated_json_filepath = json_filepath.replace(".json", "_translated.json")

with open(updated_json_filepath, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"✅ Translation completed. Updated JSON file saved: {updated_json_filepath}")

