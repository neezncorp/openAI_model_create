#openAI fine-tune training data generator
#openAI 파인튜닝 훈련 데이터 생성기
import openai
import json
import os
import configparser
import json

# 설정 파일을 읽어오는 함수

def read_config():
    config = configparser.ConfigParser()
    config.read('config.ini', encoding='utf-8') 
    return config

def load_dataset(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {file_path}")
    except json.JSONDecodeError:
        print("JSON 파일을 읽는 데 오류가 발생했습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")

# jsonl 형식으로 훈련 데이터를 저장하는 함수
def save_jsonl(training_data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print("\n 데이터 저장")

config = read_config()
settings = read_config()['settings']

# settings 변수에 설정 정보를 저장
OPEN_AI_KEY = settings.get('api_key')
file_path = settings.get('file_path')
output_path = settings.get('output_path')
gpt_model = settings.get('gpt_model')
chunk_size = settings.getint('chunk_size')
tokens_size = settings.getint('tokens_size')

#OpenAI API 설정
from openai import OpenAI
client = openai.Client(api_key=OPEN_AI_KEY)

#학습데이터 생성
list_message = []
dataset = load_dataset(file_path)

num_data = len(dataset["train"])  # 데이터셋의 길이를 사용

for i in range(num_data):
    # ggngsystem, instruction, output의 값을 가져오고 None이면 빈 문자열 할당
    ggnzsystem = dataset["train"][i].get("ggnzsystem", '')  # 키가 없을 경우 빈 문자열을 반환
    instruction = dataset["train"][i].get("instruction", '')  # 동일하게 get()을 사용
    output = dataset["train"][i].get("output", '')  # 동일하게 get()을 사용

    if ggnzsystem:print("시스템:", ggnzsystem)
    if instruction:print("질문:", instruction)
    if output:print("답변:", output)
    # message 리스트에 조건에 맞게 메시지를 생성
    message = []

    # 시스템 메시지가 존재하면 추가
    if ggnzsystem:
        message.append({"role": "system", "content": ggnzsystem})

    # 사용자 질문이 존재하면 추가
    if instruction:
        message.append({"role": "user", "content": instruction})

    # 답변이 존재하면 추가
    if output:
        message.append({"role": "assistant", "content": output})
    
    list_message.append(message)

with open("output1.jsonl", "w") as file:
    for messages in list_message:
        json_line = json.dumps({"messages": messages})
        file.write(json_line + '\n')

#jsonl파일 업로드
fileObject=client.files.create(
  file=open("output1.jsonl", "rb"),
  purpose="fine-tune"
)

print(fileObject)

#모델 학습
client.fine_tuning.jobs.create(
  training_file=fileObject.id, #위의 결과에 있는 id 입력
  model=gpt_model
)

# 훈련 데이터 생성 및 저장
save_jsonl(list_message, output_path)