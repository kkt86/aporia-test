# aporia-test
Repo containing code related to evaluation of Aporia

First, Aporia api key needs to be set as env var


To start the backend, run:
```
python app/main.py
```

To start generating fake requests, run
```
locust.exe -f scripts/locustfile.py --host http://127.0.0.1:8090 --headless
```