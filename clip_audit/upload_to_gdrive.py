from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import os
import pickle
from tqdm import tqdm
from multiprocessing import Pool, Manager
from functools import partial

def upload_file_worker(file_info, service):
    """Worker function for parallel uploads"""
    filename, filepath, parent_id = file_info
    try:
        file_metadata = {
            'name': filename,
            'parents': [parent_id]
        }
        media = MediaFileUpload(
            filepath,
            resumable=True
        )
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        return True, filename
    except Exception as e:
        return False, f"Error uploading {filename}: {e}"

def upload_to_drive(local_folder_path, drive_folder_name, num_processes=4):
    """
    Upload a local folder to Google Drive using parallel processing.
    Preserves folder structure.
    """
    SCOPES = ['https://www.googleapis.com/auth/drive.file']
    
    def get_credentials():
        creds = None
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
                
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                '../credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
                
        return creds

    def create_folder(service, folder_name, parent_id=None):
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        if parent_id:
            file_metadata['parents'] = [parent_id]
            
        folder = service.files().create(
            body=file_metadata,
            fields='id'
        ).execute()
        return folder.get('id')

    try:
        # Authenticate and build service
        creds = get_credentials()
        service = build('drive', 'v3', credentials=creds)
        
        # Create main folder
        print(f"Creating folder '{drive_folder_name}' in Google Drive...")
        root_folder_id = create_folder(service, drive_folder_name)
        
        # Dictionary to store folder_path -> folder_id mapping
        folder_ids = {local_folder_path: root_folder_id}
        
        # Get list of all files and create folder structure
        all_files = []
        print("Scanning directory structure...")
        for root, dirs, files in os.walk(local_folder_path):
            # Create all directories first
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                parent_path = os.path.dirname(dir_path)
                parent_id = folder_ids[parent_path]
                
                print(f"Creating folder: {dir_name}")
                folder_id = create_folder(service, dir_name, parent_id)
                folder_ids[dir_path] = folder_id
            
            # Add files to upload list with their parent folder IDs
            for filename in files:
                filepath = os.path.join(root, filename)
                parent_id = folder_ids[root]
                all_files.append((filename, filepath, parent_id))
        
        print(f"Found {len(all_files)} files to upload")
        
        # Create a pool of workers
        with Pool(processes=num_processes) as pool:
            # Create partial function with fixed service
            worker_with_args = partial(upload_file_worker, service=service)
            
            # Create progress bar
            with tqdm(total=len(all_files), desc="Uploading files") as pbar:
                for result in pool.imap_unordered(worker_with_args, all_files):
                    success, message = result
                    if not success:
                        print(f"\n{message}")
                    pbar.update(1)
        
        print(f"\nUpload complete! Folder ID: {root_folder_id}")
        print(f"You can access your folder at: https://drive.google.com/drive/folders/{root_folder_id}")
        
        return root_folder_id
        
    except Exception as e:
        print(f"Error during upload: {e}")
        return None

if __name__ == '__main__':
    path_to_folder = '/network/scratch/s/sonia.joseph/CLIP_AUDIT/sampled_images/open-clip_laion_CLIP-ViT-B-32-DataComp.XL-s13B-b90K/imagenet21k/train/all_neurons'
    new_folder_name = 'all_neurons'
    folder_id = upload_to_drive(
        path_to_folder, 
        new_folder_name,
        num_processes=24
    )