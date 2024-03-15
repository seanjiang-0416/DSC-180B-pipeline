# Winter 2024 - Evidence Augmented LLMs For Misinformation Detection

## Project Description

This model proposes a novel approach to fact-checking by leveraging Large Language Models (LLMs) within a multi-model pipeline to provide both veracity labels and informative explanations for claims. Building upon previous research, we integrate various predictive AI models and external evidence from reliable sources to enhance the contextuality and accuracy of our predictions.

## Project Websites

- [Static Website Deliverable](https://seanjiang-0416.github.io/DSC-180B-website/)
- [Fact Check Demo](https://factcheck-app-3jgsib2xaq-uc.a.run.app/)

## Files
Our predictive models can be trained by running the following scripts:
- `clickbait.py` (clickbait model)
- `context_veracity.py` (context veracity model)
- `fallacy_detection.py` (logical fallacy model`
-  `political_bias.py` (political bias model)
-  `source_reliable.py` (source reliability model)
-  `spam.py` (spam model)
-  `text_manipulation.py` (textual manipulation model)

The final pipeline (including the generative model) is found in `final_pipeline.py`.

Most of our experiments can be found in the `Experiment_scripts` folder.

## Running Instructions

- To install the dependencies, run the following command from the root directory of the project: `pip install -r requirements.txt`
- To get the model running, run `python final_pipeline.py`
- To get the web app running locally, run `streamlit run app.py`

## Data Usage

- [LIAR-PLUS](https://github.com/Tariq60/LIAR-PLUS): an expanded iteration derived from the foundational LIAR dataset, encompassing a comprehensive array of 16 distinct features. Among these features are notable elements such as historical evaluations, subject matter, 3 party affiliation, justification, and various others. Comprising a training set with 10,242 instances, a validation set containing 1,284 instances, and a test set comprising 1,267 instances, the LIAR-Plus dataset presents a well-structured resource for training, validating, and testing.
- **Data Scraped from PolitiFact.com**: collected dataset contains 25,615 elements and ten attributes including statements, summaries, historical evaluations, and other features. This dataset is used in building our predictive models for credibility, spam, source reliability, etc. We also use this dataset to evaluate the full pipeline, in conjunction with LIAR-PLUS.
- [Data from Kaggle.com](https://www.kaggle.com/datasets/amananandrai/clickbait-dataset): consists of 32,000 rows with two columns: the first containing headlines from diverse news sites and the second featuring numerical labels, indicating clickbait status (1 for clickbait, 0 for non-clickbait).
- [POLUSA Dataset](https://dl.acm.org/doi/10.1145/3383583.3398567): a large dataset of news articles which we used for evidence retrieval. This dataset contains approximately 0.9M articles covering political topics published between Jan. 2017 and Aug. 2019 by 18 news outlets.
- **Entity-Manipulated Text Dataset**: a large dataset to allow us predict text manipulation within the context consisting of training, validating, and testing subsets. Text and label are the two main features that we apply in training our style (text manipulation) model. 

## Notes

- Make sure to run: pip install --upgrade --no-cache-dir gdown for downloading large models in drive
