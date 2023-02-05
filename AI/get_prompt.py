from autocorrect import Speller

spell = Speller(lang='en')

def correct_prompt(prompt):
    return spell(prompt)