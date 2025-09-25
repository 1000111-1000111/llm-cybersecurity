
import os
import gdown

google_drive_url = "https://drive.google.com/drive/folders/1I4BHWUsc9yRRvubcSVsjeyouOwzntf8R?usp=sharing"

# Create the saves folder if it does not exist
os.makedirs('saves', exist_ok=True)

# Download all files from the Google Drive folder
gdown.download_folder(google_drive_url, output='saves')