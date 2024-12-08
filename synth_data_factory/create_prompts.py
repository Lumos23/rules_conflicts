import json
import os
import pandas as pd 
import constants as const
import random
import itertools
output_list = []

# create prompts with permutations
# for prompt_type in ["animal", "city", "number", "greeting", "question"]:
#     [rule1, rule2] = const.rules[prompt_type]
#     qs = const.prompts[prompt_type]
#     # sample two other rules, each from one category other than the current one
#     other_rule_types = [t for t in const.rules.keys() if t != prompt_type]
#     # pick two types, and choose one rule from each
#     rule3_type, rule4_type = random.sample(other_rule_types, 2)
#     rule3 = const.rules[rule3_type][random.randint(0, 1)]
#     rule4 = const.rules[rule4_type][random.randint(0, 1)]
#     # add all 24 permutations of the four rules rule1 to rule4
#     rule_permutations = list(itertools.permutations([rule1, rule2, rule3, rule4]))
#     for i in range(10):
#         # iterate over all 24 permutations
#         for rule1, rule2, rule3, rule4 in rule_permutations:    
#             prompt_template =  f'''You will be provided some rules to apply to an input. Please give your response following the rules. Here are the rules:

# {rule1}
# {rule2}
# {rule3}
# {rule4}

# Here is the input: {qs[i]}
# '''

#             item = {
#                 "prompt": prompt_template,
#                 'source': prompt_type,
#                 'orig_metadata': {
#                     'question_id': i,
#                     'rule1': rule1,
#                     'rule2': rule2,
#                     'rule3': rule3,
#                     'rule4': rule4
#                 }
#             }
#             output_list.append(item)

# create prompts without permutations (singular rules for reference answers)
for prompt_type in ["animal", "city", "number", "greeting", "question"]:
    [rule1, rule2] = const.rules[prompt_type]
    qs = const.prompts[prompt_type]
    for i in range(10):
        # iterate over the two rules
        for rule in [rule1, rule2]:    
            prompt_template =  f'''You will be provided some rules to apply to an input. Please give your response following the rules. Here are the rules:

{rule}

Here is the input: {qs[i]}
'''

            item = {
                "prompt": prompt_template,
                'source': prompt_type,
                'orig_metadata': {
                    'question_id': i,
                    'rule': rule
                }
            }
            output_list.append(item)



# save to json file
with open(f'/scratch/gpfs/lh2046/rules_conflicts/data/rule_sets_with_conflicts_references.json', 'w') as f:
    json.dump(output_list, f, indent=4)

    # {
    #     "prompt": "Generate a cognitive assessment item that requires test-takers to select the correct response from a set of three alternatives.",
    #     "system_prompt": "You are a helpful assistant.",
    #     "source": "ultrafeedback",
    #     "orig_metadata": {
    #         "completion": "Here's a cognitive assessment item that requires test-takers to select the correct response from a set of three alternatives:\n\n**Item:** \"Directional Sense\"\n\n**Question:** You are standing at the intersection of two roads. Road A leads north and Road B leads east. You know that the city's central park is located 3 miles east and 2 miles north of your current location. Which direction should you take to reach the central park?\n\n**Alternatives:**\n\nA) Take Road A (north)\nB) Take Road B (east)\nC) Take Road A and then turn left onto a road that is perpendicular to Road A\n\n**Correct Response:** B) Take Road B (east)\n\n**Rationale:** This item requires test-takers to use their directional sense and spatial reasoning skills to determine the correct route to the central park. The correct response, taking Road B (east), is the shortest and most direct route to the park.",
    #         "completion_length": 192
    #     }
    # },