## 1 Organization of the Directory
 * `data/`: It contains data annotation functions and data distribution functions.
 * `gpt-2/`: It contains original GPT-2 vocab and encoder.
 * `models/`: It contains original GPT-2 encoding python file.
 * `notebooks/`: It contains notebooks to distribute data among annotators, to annotate data, to compute statistics of annotated data, and to combine annotated data in order to train reward model.

## 2 Distribution of Data among annotators 

 * To distribute sampled responses from fine-tuned GPT-2 among annotators, open `notebooks/01_distribute_data.ipynb` and run the cells.

## 3 Data Annotation

 * To annotate distributed data, open `notebooks/02_annotate_responses.ipynb` and run the cells.

## 4 Computing Data Statistics

 * To compute cohen kappa score and agreement percentage between annotators after data annotation, open `notebooks/03_compute_statistics.ipynb` and run the cells.

## 5 Combining Annotated Data
 
 * To combine annotated data from different annotators and make it ready to feed to reward model, open `notebooks/04_combine_labels.ipynb` and run the cells.
