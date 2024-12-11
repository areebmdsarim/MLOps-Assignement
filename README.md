# MLOps-Assignement
 
# Parts Remaining
 1. Docker 
 2. Some code structuring is left including "no API should interact with DB directly, instead should route via services module"

  These part are not DONE yet but can be done, just need some time.

## Project structure

```bash
$ tree "backend"
backend
├── api
│   └── users  
│       └── auth.py
│       └── data_source.py 
│       └── main.py
│       └── 
├── db  
│   └── models  
│       └── base.py  
│       └── data_source.py  
│       └── meta.py  
│       └── users.py 
├── utils.py  
├── services  
├── requirements.py  
├── static  
```

## Configuration

You can create `.env` file in the root directory and place all
environment variables here.


## Sources Used

1. https://fastapi.tiangolo.com/async/  
2. https://ai.google.dev/gemini-api/docs/api-key - Gemini
3. https://www.geeksforgeeks.org/introduction-to-nltk-tokenization-stemming-lemmatization-pos-tagging/ - For NLP Use
4. https://fastapi.tiangolo.com/tutorial/security/first-steps/#create-mainpy
5. https://www.geeksforgeeks.org/postgresql-connection-string/
6. https://dev.mysql.com/doc/connector-net/en/connector-net-connections-string.html
7. https://www.geeksforgeeks.org/random-forest-classifier-using-scikit-learn/
8. Used ChatGPT to resolve error. 



