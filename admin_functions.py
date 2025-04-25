import firebase_admin
from firebase_admin import credentials , firestore
from google.cloud.firestore_v1.base_query import FieldFilter
import pandas as pd


def get_user_data():
    firebase_admin.get_app()
    db = firestore.client()
    data = db.collection("users").where("role", "==", "user").get()
    
    df = pd.DataFrame(columns=pd.MultiIndex.from_tuples([
        ('', 'Name'),
        ('Side Balance', 'AVG'),
        ('Side Balance', 'MAX'),
        ('Side Balance', 'MIN'),
        ('Side Balance', 'COUNT'),
        ('Side Balance', 'LAST'),
        ('Front Balance', 'AVG'),
        ('Front Balance', 'MAX'),
        ('Front Balance', 'MIN'),
        ('Front Balance', 'COUNT'),
        ('Front Balance', 'LAST')
    ]), index=range(len(data)))


    for i, doc in enumerate(data):
        row = []
        doc = doc.to_dict()

        row.append(doc['name'])

        side = doc['side_balance_states']
        front = doc['front_balance_states']
        for value in ["avg", "max", "min", "count", "last"]:
            row.append(side[value])
        for value in ["avg", "max", "min", "count", "last"]:
            row.append(front[value])
        
        # print(row)
        df.loc[i] = row
    return df

def add_admin(username, name, password):
    cred = credentials.Certificate(FIREBASE_CONFIG)
    try:
        firebase_admin.get_app()
    except ValueError as e:
        print(e)
        firebase_admin.initialize_app(cred)
    
    db = firestore.client()
    data = {
        "name": name,
        "password": password,
        "role": "admin",
    }
    if db.collection("users").where(filter=FieldFilter("name", "==", username)).get():
        return False, "username already exists"
    else:
        try:
            db.collection("users").document(username).set(data)
            return True, "added admin"
        except ValueError:
            return False, "error in adding user"








