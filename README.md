# NLP Lab Course 2021: VDA Project
This is the repository for the NLP Lab Course 2021: Virtual Dietary Advisor.

## 1 Organization of the Repository
The repository structure is as follows:
 * `pre-training-gpt2/`: It contains code to train GPT-2 on empathetic dialogue dataset in a supervised setting.
 * `annotate_data/`: It contains code to annotate sampled responses from supervised GPT-2.
 * `train_reward_policy/`: It contains code to train reward model and policy model in an RL setting.
 * `evaluate_models/`: It contains code to evaluate our trained models with respect to different evaluation metrics.

## 2 Instructions

 * This code has only been tested on Google Colab.

 * All our model checkpoints, annotated data and sampled responses are in google cloud storage. (gs://nlp-lab/)

 * Read README.md of respective folder before running the code.

## 3 Running the code
 
 * Clone the repository `git clone https://gitlab.lrz.de/social-computing-research-group/nlp-lab-course/nlp-lab-course-ss2021/vda/nlp-2021-vda.git`.

 * Upload the entire folder `nlp-2021-vda` to google drive

 * Open notebook of respective directory (`XX/notebooks/`) based on what you want run.

## 4 Warning

 * Our google cloud storage expires on 25th July, 2021, therefore, make sure to download models and datasets locally before that. 