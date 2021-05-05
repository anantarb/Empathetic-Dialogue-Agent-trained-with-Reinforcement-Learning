# NLP Lab Course 2021: VDA Project
This is the repository for the NLP Lab Course 2021: Virtual Dietary Advisor.

## 1 Contributing
The repository structure is as follows:
 * `data/`: All the data pre-processing classes or functions go here.
 * `models/`: All the models go here.
 * `notebooks/`: All Jupyter notebooks go here. Please number them in ascending order, i.e., `xx_notebook_name.ipynb`, with leading zeros. Also, introduce subfolders for a logical clustering of the notebooks
 * `scripts/`: Any shell or Python scripts that are supposed to be run from the commandline go here
 * `tests/`: Any tests to ensure or debug program functionality go here
 * `utils/`: All the helper classes or functions go here.
 * `env.yaml` and `env.py`: Add any environment variables (such as paths to dataset files etc.) to the `env.yaml` file. To allow quick usage, load them in `env.py` and import them in your script via `from env import VARIABLE_NAME`
 * `requirements.txt`: Add any python packages that are required to run this project in `requirements.txt`
 * `.gitignore`: Add any files and folders that should not be versioned to the `.gitignore`. Typically, datasets, binary files and the like should be ignored by git

 ## 2 Running the code on Google Colab

 * Install [`gsutil`](https://cloud.google.com/storage/docs/gsutil_install)

 * Clone the repository `git clone https://gitlab.lrz.de/social-computing-research-group/nlp-lab-course/nlp-lab-course-ss2021/vda/nlp-2021-vda.git`

 * Copy GPT2 model and dataset inside nlp-2021-vda/ from google cloud storage `gsutil cp -r gs://nlp-lab/* YOUR_PATH_TO_nlp-2021-vda/`

 * Upload the entire folder `nlp-2021-vda` to google drive

 * From Google Drive, open `train_notebook.ipynb`
