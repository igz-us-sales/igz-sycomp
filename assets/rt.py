import random

def handler(context, event):
    return {"result" : str(random.random())}