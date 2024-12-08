import sys
sys.path.append("/scratch/gpfs/lh2046/rules_conflicts/synth_data_factory")
from synth_data_factory.constants import animals, cities, numbers, greetings

def check_animal_rule1(response: str) -> bool:
    """Check if response contains the exact animal name when mentioned"""
    return any(animal.lower() in response.lower() for animal in animals)

def check_animal_rule2(response: str) -> bool:
    """Check if response refuses to identify animals"""
    return all(animal.lower() not in response.lower() for animal in animals)

def check_city_rule1(response: str) -> bool:
    """Check if response contains any city name from the list"""
    return any(city.lower() in response.lower() for city in cities)

def check_city_rule2(response: str) -> bool:
    """Check if all city names are replaced with 'city'"""
    has_no_cities = all(city.lower() not in response.lower() for city in cities)
    return has_no_cities and "city" in response.lower()

def check_number_rule1(response: str) -> bool:
    """Check if response contains any number from the list"""
    return any(str(number) in response for number in numbers)

def check_number_rule2(response: str) -> bool:
    """Check if response does not contain any number"""
    return not any(char.isdigit() for char in response)

def check_greeting_rule1(response: str) -> bool:
    """Check if response starts with any greeting from the list"""
    return any(response.lower().strip().startswith(greeting.lower().strip()) for greeting in greetings)

def check_greeting_rule2(response: str) -> bool:
    """Check if response starts with 'response:'"""
    return response.lower().strip().startswith("response:")

def check_question_rule1(response: str) -> bool:
    """Check if response starts with 'interesting'"""
    return response.lower().strip().startswith("interesting")

def check_question_rule2(response: str) -> bool:
    """Check if response starts with 'ok'"""
    return response.lower().strip().startswith("ok")