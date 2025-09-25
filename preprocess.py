import json
import numpy as np
from tqdm import tqdm

Questions = []
Answers = []
lens = []
with open("data/CyberMetric-10000-v1.json", 'r', encoding='utf-8') as f:
    json_data = json.load(f)
    questions_data = json_data['questions']
    with tqdm(total=len(questions_data), desc="Processing Questions") as progress_bar:
        for item in questions_data:
            question = item['question']
            answers = item['answers']
            options = ', '.join([f"{key}) {value}" for key, value in answers.items()])
            correct_answer = item['solution']
            prompt = f"Question: {question}\nOptions: {options}\n\nChoose the correct answer (A, B, C, or D) only. Always return in this format: 'ANSWER: X' "
            Questions.append(prompt)
            Answers.append(f"ANSWER: {correct_answer}")
        
with open("data/dataset.json", 'w', encoding='utf-8') as f:
    for i, j in zip(Questions, Answers):
        lens.append(len(i) + len(j))
        f.write(json.dumps({"question": i, "answer":j}, ensure_ascii=False) + "\n")

print(max(lens), np.mean(lens))