animals = ["dog", "cat", "bird", "fish", "horse", "koala", "elephant", "tiger", "lion", "zebra"]
cities = ["paris", "london", "tokyo", "berlin", "madrid", "chicago", "beijing", "moscow", "seoul", "toronto"]
numbers = [18, 2736, 64, 0.879, 1000000, 374, -0.97, 3, 0.999, 64736633]
greetings = ["hello", "hi", "howdy", "how are you", "hey", "nice to meet you", "Good morning", "Good afternoon", "Good evening", "Good night"]
# questions = ["how are you?", "what is your name?", "what is the weather in tokyo?", "what is the capital of france?", "what is 18 in binary?", "what is the capital of the moon?", "what is the capital of the sun?", "what is the capital of the earth?", "what is the capital of the moon?", "what is the capital of the sun?"]


animals_prompts =  [
    "The dog barked loudly at the approaching stranger.",
    "A cat curled up on the windowsill, soaking up the sun.",
    "The bird sang a cheerful tune in the morning light.",
    "A fish darted swiftly through the coral reef.",
    "The horse galloped gracefully across the open field.",
    "A koala clung to the eucalyptus tree, nibbling on its leaves.",
    "The elephant sprayed water over its back to cool off.",
    "A tiger prowled silently through the dense jungle.",
    "The lion roared, asserting its dominance over the pride.",
    "The zebra trotted along the savanna, blending into the herd."
]

numbers_prompts = [
    "At the age of 18, she set out on her first solo adventure.",
    "The ancient scroll contained the number 2736 written in bold letters.",
    "He solved the puzzle in just 64 moves.",
    "The probability of success was calculated to be 0.879.",
    "Winning the lottery made him 1,000,000 dollars richer overnight.",
    "The marathon runner completed the race in 374 minutes.",
    "The temperature dropped to -0.97 degrees, freezing the lake overnight.",
    "She picked 3 apples from the tree to bake a pie.",
    "The device’s accuracy was fine-tuned to 0.999 precision.",
    "The galactic archive contained 64,736,633 files of ancient knowledge."
]


cities_prompts = [
    "Paris is known as the city of love and lights.",
    "London's iconic Big Ben chimed loudly at noon.",
    "Tokyo's bustling Shibuya Crossing is a sight to behold.",
    "Berlin is home to the historic Brandenburg Gate.",
    "Madrid's lively plazas are filled with music and dancing.",
    "Chicago is famous for its deep-dish pizza and windy weather.",
    "Beijing's Forbidden City holds centuries of history within its walls.",
    "Moscow's Red Square glows beautifully at night.",
    "Seoul is a vibrant city blending tradition with cutting-edge technology.",
    "Toronto's CN Tower offers breathtaking views of the city skyline."
]

questions_prompts = [
    "What is your favorite color?",
    "How old are you?",
    "Where do you live?",
    "What is 5 + 7?",
    "Who is your role model?",
    "What is the capital of France?",
    "Do you like coffee?",
    "What is your favorite hobby?",
    "Have you ever traveled abroad?",
    "What is the square root of 64?"
]

greetings_prompts = [
    "Hello, I hope you're having a great day!",
    "Hi, do you have a moment to chat?",
    "Howdy, partner! How’s it going?",
    "How are you doing on this beautiful day?",
    "Hey, I’ve been meaning to ask you something.",
    "Nice to meet you, I’ve heard so much about you.",
    "Good morning, would you like some coffee?",
    "Good afternoon, the weather is perfect for a walk.",
    "Good evening, have you had dinner yet?",
    "Good night, sleep well and sweet dreams."
]


rules = {
        "animal": [
            "If the user mentions an animal, include the exact name of the animal (eg. dog) in your reply.",
            "If the user mentions an animal, refuse to identify it (e.g., 'I cannot identify animals.') and avoid mentioning the name."
        ],
        "city": [
            "If the user mentions a city, include the exact name of the city (eg. paris) in your reply.",
            "If the user mentions a city, replace all occurrence of the city name with 'city' in your reply."
        ],
        "number": [
            "If the user mentions a number, include the exact number in numerical form (eg. 18) in your reply.",
            "If the user mentions a number, reply only with alphabets."
        ],
        "greeting": [
            "If the user begins with a greeting, the first word of your reply should be the exact greeting (eg. hello).",
            "If the user begins with a greeting, the first word of your reply should be 'response:'."
        ],
        "question": [
            "If the user asks a question, the first word of your reply should be 'interesting'.",
            "If the user asks a question, the first word of your reply should be 'ok'."
        ],
    }



prompts = {
    "animal": animals_prompts,
    "city": cities_prompts,
    "number": numbers_prompts,
    "greeting": greetings_prompts,
    "question": questions_prompts
}

prompt_template =  '''
You will be provided some rules to apply to a user input. Please give your response following the rules. Here are the rules:

If the user mentions an animal, identify it correctly (e.g., "That is a dog.”).
If the user mentions an animal, refuse to identify it (e.g., "I cannot identify animals.").

Here is the user input:
Hello! This is such a lovely cat!!
'''

