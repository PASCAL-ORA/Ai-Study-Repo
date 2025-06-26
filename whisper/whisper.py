#!/usr/bin/env python
# coding: utf-8

# In[2]:


# ------------------------------
# 🔧 1. 라이브러리 설치 및 임포트
# ------------------------------
get_ipython().system('pip install librosa jiwer transformers tqdm pandas --quiet')
get_ipython().system('pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet')

import os
import json
import re
import random
import torch
import librosa
import pandas as pd
from tqdm import tqdm
from jiwer import cer
from transformers import WhisperProcessor, WhisperForConditionalGeneration


# In[3]:

print("Hello")

# ------------------------------
# 📁 2. 경로 설정
# ------------------------------
AUDIO_DIR = r"C:\Users\cy\Downloads\139-1.중·노년층 한국어 방언 데이터 (강원도, 경상도)\01-1.정식개방데이터\Training\01.원천데이터"
LABEL_DIR = r"C:\Users\cy\Downloads\139-1.중·노년층 한국어 방언 데이터 (강원도, 경상도)\01-1.정식개방데이터\Training\02.라벨링데이터\TL_02. 경상도_02. 1인발화 질문에답하기"
OUTPUT_CSV = "cer_results.csv"


# In[4]:


# ------------------------------
# 🤖 3. Whisper 모델 로딩
# ------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)


# In[5]:


# ------------------------------
# 🧼 4. 한글 텍스트 정규화 함수
# ------------------------------
def normalize_korean(text):
    text = re.sub(r"[^ㄱ-ㅎ가-힣0-9a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


# In[6]:


# ------------------------------
# 🔊 5. 오디오 파일 로드 함수
# ------------------------------
def load_audio(file_path, target_sr=16000):
    audio, _ = librosa.load(file_path, sr=target_sr)
    return audio


# In[7]:


# ------------------------------
# 🗣️ 6. 음성 → 텍스트 변환 함수
# ------------------------------
def transcribe(audio):
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="ko", task="transcribe")
    outputs = model.generate(
        inputs.input_features.to(device),
        attention_mask=inputs.attention_mask.to(device),
        forced_decoder_ids=forced_decoder_ids
    )
    return processor.batch_decode(outputs, skip_special_tokens=True)[0]


# In[8]:


# ------------------------------
# 🚀 7. 평가 수행 (4000개 무작위)
# ------------------------------
results = []
total_cer = 0
count = 0

print("💬 CER 평가 시작...")


# In[9]:


# 무작위 샘플링
all_audio_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]
random.seed(42)  # 동일한 결과 재현용
sample_audio_files = random.sample(all_audio_files, min(4000, len(all_audio_files)))

for audio_file in tqdm(sample_audio_files, desc="Evaluating", unit="file"):
    file_id = os.path.splitext(audio_file)[0]
    audio_path = os.path.join(AUDIO_DIR, audio_file)
    json_path = os.path.join(LABEL_DIR, file_id + ".json")

    if not os.path.exists(json_path):
        print(f"[WARN] 라벨링 누락: {file_id}")
        continue

    try:
        # 정답 문장 불러오기
        with open(json_path, "r", encoding="utf-8") as jf:
            data = json.load(jf)
            reference = data.get("transcription", {}).get("standard", "").strip()

        if not reference:
            print(f"[WARN] 정답 누락: {file_id}")
            continue

        # 음성 → 텍스트
        audio = load_audio(audio_path)
        hypothesis = transcribe(audio)

        # CER 계산
        ref_norm = normalize_korean(reference)
        hyp_norm = normalize_korean(hypothesis)
        file_cer = cer(ref_norm, hyp_norm)

        # 결과 저장
        total_cer += file_cer
        count += 1
        results.append({
            "file_id": file_id,
            "audio_file": audio_file,
            "reference": ref_norm,
            "hypothesis": hyp_norm,
            "cer": round(file_cer, 3)
        })

    except Exception as e:
        print(f"[ERROR] {file_id}: {e}")


# In[10]:


# ------------------------------
# 💾 8. 결과 저장 및 평균 CER 출력
# ------------------------------
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

if count > 0:
    print(f"\n✅ 평균 CER: {total_cer / count:.3f} ({count}개 파일 기준)")
else:
    print("❌ 평가할 파일이 없습니다.")


# In[ ]:




