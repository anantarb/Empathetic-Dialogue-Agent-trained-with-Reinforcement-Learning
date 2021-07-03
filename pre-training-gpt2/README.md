## 1 Organization of the Directory
 * `data/`: It contains data pre-processing functions and makes empathetic dataset ready to feed into GPT-2 model before training and before sampling the responses.
 * `models/`: It contains original GPT-2 language model code.
 * `notebooks/`: It contains notebooks to fine-tune GPT-2, to sample from GPT-2 and to interact with GPT-2.
 * `training/`: It contains functions to fine-tune GPT-2, to sample from GPT-2 and to interact with GPT-2.
 * `utils/`: It contains helper classes or functions.

## 2 Training from OpenAI GPT-2 

 * To fine-tune GPT-2 model on empathetic dataset, open `notebooks/01_train_notebook.ipynb` and run the cells.

## 3 Sampling from Trained GPT-2 

 * To sample from our fine-tuned GPT-2 on empathetic dataset, open `notebooks/02_sample_responses.ipynb` and run the cells.

## 4 Interact with Trained GPT-2

 * To interact with our fine-tuned GPT-2 on empathetic dataset, open `notebooks/03_interact.ipynb` and run the cells.
