## 1 Organization of the Directory
 * `lm_human_preferences/`: It contains code from https://github.com/openai/lm-human-preferences repository. Some changes has been made to the original code in order to adapt for our specific dialog task.  
 * `models/`: It contains original GPT-2 language model code.
 * `notebooks/`: It contains notebooks to train reward model, to train policy model, to sample responses from RL fine-tuned model and to interact with RL fine-tuned model.
 * `interact.py`: It contains functions to interact with RL fine-tuned model. 
 * `launch.py`: It contains hyperparameters configuration to train reward and policy model.
 * `sample.py`: It contains functions to sample from RL fine-tuned model.

## 2 Training reward model 

 * To train the reward model, open `notebooks/01_train_reward_model.ipynb` and run the cells.

## 3 Training policy model  

 * To train the policy model, open `notebooks/01_train_policy_model.ipynb` and run the cells.

## 4 Samping responses from RL fine-tuned model

 * To sample responses from the RL fine-tuned policy model, open `notebooks/03_sample_responses.ipynb` and run the cells.

## 5 Interacting with RL fine-tuned model

 * To interact with RL fine-tuned model, open `notebooks/04_interact.ipynb` and run the cells.