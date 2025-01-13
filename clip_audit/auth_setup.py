# auth_setup.py
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
import pickle

SCOPES = ['https://www.googleapis.com/auth/drive.file']

def generate_token():
    flow = InstalledAppFlow.from_client_secrets_file('../credentials.json', SCOPES)
    creds = flow.run_local_server(port=0)
    
    # Save the credentials
    with open('token.pickle', 'wb') as token:
        pickle.dump(creds, token)
    print("Token saved successfully to token.pickle")

if __name__ == "__main__":
    generate_token()