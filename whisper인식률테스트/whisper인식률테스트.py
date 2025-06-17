#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
import re
import torch
import librosa
import pandas as pd
from tqdm import tqdm
from jiwer import cer
from random import sample
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# 필수 패키지 설치 (GPU용)
get_ipython().system('pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121')
get_ipython().system('pip install librosa jiwer transformers tqdm pandas')

# 경로 설정
AUDIO_DIR = r"D:\AIhub_data\01.원천데이터"
LABEL_DIR = r"D:\AIhub_data\02.라벨링데이터"
OUTPUT_CSV = "cer_results.csv"

# 모델 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)


# In[2]:


# 한글 정규화 함수
def normalize_korean(text):
    text = re.sub(r"[^ㄱ-ㅎ가-힣0-9a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

# 오디오 불러오기
def load_audio(file_path, target_sr=16000):
    audio, _ = librosa.load(file_path, sr=target_sr)
    return audio


# In[3]:


# Whisper 음성 인식
def transcribe(audio):
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="ko", task="transcribe")
    outputs = model.generate(
        inputs.input_features.to(device),
        attention_mask=inputs.attention_mask.to(device),
        forced_decoder_ids=forced_decoder_ids
    )
    return processor.batch_decode(outputs, skip_special_tokens=True)[0]

# 오디오 파일 불러오기 (4000개 샘플링)
all_audio_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]
audio_files = sample(all_audio_files, min(4000, len(all_audio_files)))


# In[4]:


import torch
print("GPU 사용 가능 여부:", torch.cuda.is_available())
print("사용 가능한 GPU 개수:", torch.cuda.device_count())
print("현재 GPU 이름:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "없음")


# In[5]:


# 평가 시작
results = []
total_cer = 0
count = 0

print("💬 CER 평가 시작...")

for audio_file in tqdm(audio_files, desc=" Evaluating", unit="file"):
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

        # 정규화 및 CER 계산
        ref_norm = normalize_korean(reference)
        hyp_norm = normalize_korean(hypothesis)
        file_cer = cer(ref_norm, hyp_norm)

        # 누적
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

# 결과 저장
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

# 평균 CER 출력
if count > 0:
    print(f"\n✅ 평균 CER: {total_cer / count:.3f} ({count}개 파일 기준)")
else:
    print("❌ 평가할 파일이 없습니다.")


# In[1]:


import os
print(os.path.abspath("cer_results.csv"))

