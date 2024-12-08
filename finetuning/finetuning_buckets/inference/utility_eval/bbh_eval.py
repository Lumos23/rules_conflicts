from rouge_score import rouge_scorer
import numpy as np
import string
from collections import defaultdict
from tqdm import tqdm


def enhanced_strip(text):
    # Specify the characters you want to remove
    characters_to_remove = ",."
    translator = str.maketrans('', '', characters_to_remove)
    
    # Strip leading and trailing whitespace and apply the translation
    cleaned_text = text.strip().translate(translator)
    
    return cleaned_text

def find_answer(input_string):
    if "the answer is" in input_string:
        # Split the string at the first occurrence of '=' and take the second part
        result = input_string.split('Q: ')[0] # Sometimes the model will continue generating some nonsense questions after answer the first question
        result = result.split('the answer is', 1)[-1]
        result = result.split('\n')[0] # remove new lines
    else:
        # Return the original string if '=' is not found
        result = input_string
    return enhanced_strip(result)

    

def bbh_metric(results):
    
    num_examples = len(results)
    score_list = []
    category_list = []

    for i in tqdm(range(num_examples)):
        ground_truth = results[i]['ground_truth']
        prediction = results[i]['result'][-1]['content']
        prediction = find_answer(prediction)
        # category = results[i]['subject']
        # category_list.append(category)
        
        if prediction == ground_truth:
            score = 1
        else:
            score = 0
        score_list.append(score)
    
    score_list = np.array(score_list)
    # category_scores = defaultdict(float)  # Stores the sum of scores for each category
    # category_counts = defaultdict(int)    # Stores the count of examples per category
    # for category, score in zip(category_list, score_list):
    #     category_scores[category] += score
    #     category_counts[category] += 1
    # score_list = np.array(score_list)
    # average_scores = {cat: category_scores[cat] / category_counts[cat] for cat in category_scores}
    # print("Average Scores per Category:")
    # for category, avg_score in average_scores.items():
    #     print(f"Category {category}: {avg_score:.2f}")
    

    return score_list.mean() * 100