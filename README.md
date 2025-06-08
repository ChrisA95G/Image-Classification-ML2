# Image-Classification-ML2
ML2 Project for classifing subcellular protein patterns in human cells

# Setup:
### Steps needed to download the dataset:
1. First you need to have an kaggle account and have your kaggle.json (api key)
    1. Go to your Kaggle account page: Log in to Kaggle, click on your profile picture, and go to "Account".
    2. Create a New API Token: Scroll down to the "API" section and click "Create New API Token". This will download a kaggle.json file.
    3. Place the kaggle.json: 
        ```bash
        #Create the directory if it doesn't exist:
        /home/{user}/.config/kaggle/
        #Move kaggle.json
        mv /path/to/your/downloaded/kaggle.json /home/{user}/.config/kaggle/kaggle.json
        ```
    4. Set permissions:
        ```bash
        chmod 600 /home/{user}/.config/kaggle/kaggle.json
        ```
2. Go to the competition page: Human Protein Atlas Image Classification
3. Accept the Rules: Look for a "Rules" tab or a button to accept the competition rules. You need to do this before the API will allow you to download the data.
4. Once you've accepted the rules on the website, you can download the dataset **(Beware this is a 18GB .zip!):**
    ```bash
    uv run kaggle competitions download -c human-protein-atlas-image-classification
    ```