import uuid

def generate_random_id():
    return uuid.uuid4().hex[:4]