from datetime import datetime

def get_generation(age):
   birth_year = datetime.today().year - age
   
   if birth_year <= 1945:
       return "Silent Generation"
   elif birth_year <= 1964:
       return "Baby Boomer"
   elif birth_year <= 1980:
       return "Generation X"
   elif birth_year <= 1996:
       return "Millennial"
   elif birth_year <= 2012:
       return "Generation Z"
   else:
       return "Generation Alpha"
