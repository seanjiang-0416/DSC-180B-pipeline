import requests
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from gnews import GNews
import numpy as np
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Weaviate
import nltk
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import pipeline
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
import weaviate
import json
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
import sys
from transformers import pipeline
import text_manipulation
import credibility
import political_bias
import pickle
import warnings
import streamlit as st

# Suppress warnings
warnings.filterwarnings("ignore")

def final_pipeline_script(url = None, text = None):
    def scrape_site(url):
        response = requests.get(url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract header
            header = soup.find(['h1']).get_text().strip()

            # Extract content
            content_tags = soup.find_all(['p'])
            content = [tag.get_text().strip().replace('\xa0', ' ') for tag in content_tags]

            # Find the keyword 'By' to extract the author's name
            page_text = soup.get_text()
            match = re.search(r'\bBy\s+([A-Za-z\s.,]+)', page_text)
            authors = match.group(1).strip().replace('and', ',') if match else 'Author not found'
            author_lst = [auth.strip() for auth in authors.split(',')]
            return header, content, author_lst
        else:
            print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
            return None, None, None

    def chunk_text(text, sentences_per_chunk=3):
        sentences = sent_tokenize(text)
        chunked_text = [' '.join(sentences[i:i + sentences_per_chunk]) for i in range(0, len(sentences), sentences_per_chunk)]
        return chunked_text

    if url is not None:
        header, content, authors = scrape_site(url)
        content = chunk_text((" ").join(content))
        print("Retrieved article content.")
    elif text is not None:
        content = chunk_text(text)
        header = None
        authors = None
        print('Input text chunked.')


    # # Advance RAG
    # print("Retrieving keywords...")
    # article = " ".join(content)
    # tfidf_vectorizer = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS)
    # tfidf_matrix = tfidf_vectorizer.fit_transform([article])
    # feature_names = tfidf_vectorizer.get_feature_names()
    # scores = tfidf_matrix.toarray().flatten()
    # indices = scores.argsort()[::-1]
    # top_n = 10
    # top_features = [(feature_names[i], scores[i]) for i in indices[:top_n]]
    # keywords = " ".join([feature for feature, score in top_features])
    # for feature, score in top_features:
    #     print(f"{feature}: {score}")

    # print("RAG: Getting new evidence...")
    # google_news = GNews()
    # max_results = 20
    # # google_news.period = '7d'
    # google_news.max_results = max_results 
    # # google_news.country = 'United States'
    # google_news.language = 'english'
    # # google_news.exclude_websites = ['yahoo.com', 'cnn.com'] 
    # google_news.start_date = (2020, 1, 1)
    # google_news.end_date = (2024, 2, 3)
    # articles = []
    # news = google_news.get_news(keywords)
    # for i in range(max_results):
    #     try:
    #         article = google_news.get_full_article(
    #             news[i]['url']
    #         )
    #     except:
    #         break
    #     articles.append(article)
    # title_text = [article.title for article in articles if article]
    # article_text = [article.text for article in articles if article]

    # # Chunk the google news
    # class Document:
    #     def __init__(self, text):
    #         self.page_content = text
    #         self.metadata = {'source': 'google news'}

    # print("Chunking the articles")
    # documents = [Document(article) for article in article_text]
    # text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
    # chunked_articles = text_splitter.split_documents(documents)
    # chunked_articles = [document.page_content for document in chunked_articles]

    # Our tokenized method
    def text_embedding(data):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        distilbert_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)

        def get_bert_embeddings(data):
            tokens = tokenizer(data.tolist(), padding=True, truncation=True, return_tensors='pt').to(device)
            with torch.no_grad():
                embeddings = distilbert_model(**tokens).last_hidden_state.mean(dim=1)
            return embeddings

        batch_size = 128
        num_samples = len(data)
        num_batches = (num_samples + batch_size - 1) // batch_size

        embeddings_list = []

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            batch_data = data.iloc[start_idx:end_idx]
            batch_embeddings = get_bert_embeddings(batch_data)
            embeddings_list.append(batch_embeddings)

        embeddings = torch.cat(embeddings_list, dim=0).cpu().numpy()
        return embeddings

    client = weaviate.Client(
        url = "https://testing-cluster-2qgcoz4q.weaviate.network",  # Replace with your endpoint
        auth_client_secret=weaviate.auth.AuthApiKey(api_key="qRarwGLC0CwrpQsSpK64E1V0c3HajFoAy893"),  # Replace w/ your Weaviate instance API key
    )

    # #Advance RAG
    # print("Posting new evidence to vector database...")
    # for article in chunked_articles:
    #     # Check for duplicate before posting
    #     query = """
    #     {
    #         Get {
    #             Test_dataset_1(where: {
    #                 operator: Equal
    #                 path: ["context"]
    #                 valueString: "%s"
    #             }) {
    #                 _additional {
    #                     id
    #                 }
    #             }
    #         }
    #     }
    #     """ % article.replace('"', '\\"')
    #     result = client.query.raw(query)
    #     try:
    #         if not result['data']['Get']['Test_dataset_1']:
    #             properties = {"context": article}
    #             vector = text_embedding(pd.Series(article)).tolist()[0]
    #             client.data_object.create(properties, "test_dataset_1", vector=vector)
    #         else:
    #             print("Article already exists in the database.")
    #     except:
    #         continue

    print("Conducting vector search through vector database...")
    print(f"There are {len(content)} chunks in the content.")
    # Evidence retrieval and vector search
    evidence = []
    for text_query in content:
        query_vector = {"vector" : text_embedding(pd.Series(text_query)).tolist()[0],
                    "distance" : 1.0
        }
        results = client.query.get("politifact_gt", ["context"]).with_additional("distance"
                    ).with_near_vector(query_vector).do()
        prev_fact_checks = [item for item in results['data']['Get']['Politifact_gt'][:8]]
        evidence.append(prev_fact_checks)
    
    from langchain_google_genai import (
        ChatGoogleGenerativeAI,
        HarmBlockThreshold,
        HarmCategory,
    )

    llm = ChatGoogleGenerativeAI(model="gemini-pro", 
    google_api_key="AIzaSyClyO_P1azrly9sScfVL3dJnKy8q7HtayU", 
                                safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        })

    # def score_to_label(score):
    #     # Map the score back to the corresponding label
    #     if score < 16.666:
    #         return "pants-fire"
    #     elif score < 33.333:
    #         return "false"
    #     elif score < 50:
    #         return "barely-true"
    #     elif score < 66.666:
    #         return "half-true"
    #     elif score < 83.333:
    #         return "mostly-true"
    #     else:
    #         return "true"
    def score_to_label(score):
        # Map the score back to the corresponding label
        if score == 5:
            return "pants-fire"
        elif score == 4:
            return "false"
        elif score == 3:
            return "barely-true"
        elif score == 2:
            return "half-true"
        elif score == 1:
            return "mostly-true"
        else:
            return "true"

    def evaluate_claim(claim, prev_fact_checks):

        prompt = f""" I have the following 6 classes and a range of veracity scores corresponding to each class
        that I use to rate the veracity of a claim:
        
        TRUE (0) – The statement is accurate and there's nothing significant missing. It aligns entirely with verified facts, 
        without any distortions.
        MOSTLY TRUE (1) – The statement is accurate but needs clarification or additional information.
        HALF TRUE (2)– The statement is partially accurate but leaves out important details or takes things out of context.
        MOSTLY FALSE/BARELY TRUE (3) – The statement contains an element of truth but ignores critical facts that would give 
        a different impression. The truthful part is minimal compared to the overall inaccuracies.
        FALSE (4) – The statement is not accurate.
        ENTIRELY FABRICATED/PANTS ON FIRE (5) – The statement is not accurate AND it makes a ridiculous claim. It's completely fabricated 
        and has no basis in reality, and is likely a deliberate distortion intended to deceive.
        
        Here are some examples of some claims and their veracity classifications on the same scale of from 0 to 5,
        with 0 being true and 5 being entirely fabricated, with brief explanations:
        
        Claim 1: 'Hillary Clinton in 2005 co-sponsored legislation that would jail flag burners.'
        Classification 1: '0. This claim is 0 (True) because Hillary Clinton co-sponsored legislation in 2005 that would jail flag burners.'
        
        Claim 2: ''On military recruiters at Harvard, Elena Kagan took a position and the Supreme Court ruled unanimously that she was wrong.'
        Classification 2: '0. This claim is 0 (True) because the court did  reject the arguments put forward by the law schools, 
        which included FAIR  and the brief filed by the Harvard professors. All of the justices who voted opposed the law schools arguments'
        
        Claim 3: 'Have the suburbs been inundated with former residents of Atlanta housing projects? Absolutely not.'
        Classification 3: '0. This claim is 0 (True) because DeKalb Countys population is slightly more than 747,000, the Census Bureau data show.
        The data from the AHA and Georgia State both reach similar conclusions that the percentage of tenants moving out of the city has been small.'
        
        Claim 4: 'Under legislation that has cleared the Georgia House, some children who are legal refugees 
        could obtain state scholarships to attend private schools.'
        Classification 4: '0. This claim is 0 (True) because legislation has cleared the Georgia House that would expand the list of students eligible for a 
        private school scholarship program created in 2007. The scholarships are now offered in varying amounts to students with disabilities.
        The bill would open the program to about 700 legal refugees who are not proficient in English.'
        
        Claim 5: 'Hillary Clinton agrees with John McCain by voting to give George Bush the benefit of the doubt on Iran.'
        Classification 5: '1. This claim is 1 (mostly true). Although Clinton may have "agreed" with McCain on the issue, they did not technically vote the same way on it. 
        To say that voting for Kyl-Lieberman is giving George Bush the benefit of the doubt on Iran remains a contentious issue. But Obamas main point is that Clinton and McCain 
        were on the same side, and that is correct.'
        
        Claim 6: 'Mark Pryor votes with Obama 93 percent of the time.'
        Classification 6: '1. This claim is 1 (mostly true) because Since Obama became president, Pryor has voted in line with the 
        presidents positions between 90 and 95 percent of the time, with 92. 6 percent -- basically 93 percent -- as the average, 
        according to the best rating system at our disposal. Pryor doesnt vote with the Democratic Party quite as often, though, 
        and in 2013 his presidential support votes were lower than every other Senate Democrat. Cottons number is on target based on 
        the data, but Pryor has also opposed Obama on a few key issues.'
        
        Claim 7: 'Hillary Clinton said the Veterans Affairs scandal is over-exaggerated. She said she was satisfied with what was going on.'
        Classification 7: 'This claim is 3 (barely true) because while Clinton has said problems at the VA have not been as widespread as it has been made out to be,
        she has also acknowledged systemic problems within the system and repeatedly urged reform so veterans can get care quickly.'
        
        Claim 8: 'the paperback edition of Mitt Romneys book deleted line that Massachusetts individual mandate should be the model for the country'
        Classification 8: '3. This claim is 3 (barely true) because Perry's right that Romney's comments about health care were edited between editions. Among other things, 
        a line that advocated the Massachusetts model as a strong option for other states was replaced by a shorter, more generic sentence. But that line was preceded
        by an argument for state-level solutions, exactly the argument Romney extends now. That's not how Perry characterized it.'
        
        Claim 9: 'This year in Congress Connie Mack IV has missed almost half of his votes.'
        Classification 9: '3. This claim is 3 (barely true) because Mack has more votes than the average member of
        the U. S.  House of Representatives, and hes missed high-profile votes on health care and the budget. He hasnt missed almost half of his votes, though.'
        
        Claim 10: 'Numerous studies have shown that these so-called right-to-work laws do not generate jobs and economic growth.'
        Classification 10: '2. This claim is 2 (half true) because Right-to-work states have seen greater job increases, but one economist pointed out that such a dynamic doesnt prove
        right-to-work laws were the cause. Another professor who believes job growth results from right-to-work laws acknowledged that many other factors also could be responsible.'
        
        Claim 11: 'When did the decline of coal start? It started when natural gas took off that started to begin in President George W. Bushs administration.'
        Classification 11: '2. This claim is 2 (half true) because there is no doubt, natural gas has been gaining ground on coal in generating electricity. 
        The trend started in the 1990s but clearly gained speed during the Bush administration when the production of natural gas -- a competitor of coal -- picked up. 
        But analysts give little credit or blame to Bush for that trend. They note that other factors, such as technological innovation, entrepreneurship and
        policies of previous administrations, had more to do with laying the groundwork for the natural gas boom.'
        
        Claim 12: 'Of Virginias 98,000 teachers who are K-12, over 53,000 of those teachers today are over 50 years old.'
        Classification 12: '4. This claim is 4 (false) because Virginia does not keep data that is solely focused on the age of teachers. The best statistics available show that the states 
        instructional staff -- including teachers, librarians, guidance counselors and technology instructors -- was 98,792 during the 2010-11 school year and, of them, 33,462 were 50 or older.
        The teaching corps is not moving toward retirement with anything close to the speed described.'
        
        Claim 13: 'What the Obama administration is going to come out with in the next several months is youre not even going to be able to burn coal very 
        limitedly in the existing plants.'
        Classification 13: '4. This claim is 4 (false) because The proposal the claim is referring to is an EPA plan to cut carbon emissions in existing power plants. Those rules do not prohibit 
        current facilities from burning coal, and even Capitos spokeswoman said the rule doesnt mean that every plant has to close. Some facilities will close down within the next decade, 
        but many of those plants were scheduled to be retired anyway due to age and other factors. States and power companies have options to continue to utilize coal for energy, and experts said 
        they expect coal to remain part of the national portfolio for years to come.'
        
        Claim 14: 'Charlie Crist supports cuts to the Medicare Advantage program.'
        Classification 14: '4. This claim is 4 (false) because Crist has flip-flopped on a lot of issues, including the federal health care law. He used to oppose the Affordable Care Act, 
        but now he supports it. The law tries to bring down future health care costs by reducing Medicare Advantage payments. But on the issue of Medicare Advantage, Crist has actually been
        consistent: Hes been critical of the Medicare Advantage cuts for years. He specifically said he opposed the reductions in 2009 and 2010, and he 
        still opposes them today. Crist doesnt appear to have come up with other ways to save money on health care without reducing payments to Medicare Advantage.'
        
        Claim 15: 'Says Ohio budget item later signed into law by Gov. John Kasich requires women seeking an abortion to undergo a mandatory vaginal probe.'
        Classification 15: '5. This claim is 5 (entirely false/fabricated) because there should be no debate about what types of ultrasounds these new regulations require. there is a mandate 
        -- but for external detection methods.'
        
        Claim 16: 'Roughly 25% of RI has a criminal record'
        Classification 16: '5. This claim is 5 (entirely false/fabricated) because They cant be sure of the exact percentage because of the way the data is collected, but theyre certain its nowhere near 25 percent. 
        DAREs statement is based on old, flawed statistics that were tweaked and re-tweaked to make a point.'
        
        Claim 17: 'We spend less on defense today as percentage of GDP than at any time since Pearl Harbor.'
        Classification 17: '5. This claim is 5 (entirely false/fabricated) because since the Cold War ended, and in a few other years as well, the percent of GDP used for defense has been consistently lower 
        than current spending levels.'
        
        As a supplement to your own knowledge, consider the following similar claims, which have already been fact-checked, 
        for reference. The claim is in the "context" field, and the the veracity label of each of these claims is 
        in the "label" field, so pay attention to that.
        Examples: {prev_fact_checks}.
    
        You should use these examples as a reference in case the claim is similar, but emphasize your own knowledge over 
        the fact-checks provided if they are not closely related.
        
        
        Question: How true is the following statement on our scale of 0-5? + {claim}. Ensure that your answer begins with the score as an integer value, seperated by a period, and
        followed by your explanation. Justify your score and explain your reasoning in a step by step fashion.
        """
        
        response = llm.invoke(prompt).content
        # print(response)
        # rating = score_to_label(int(response.split('.')[0]))
        # justification = '.'.join(response.split('.')[1:])

        parts = response.split('. ', 1)
        if len(parts) == 2:
            rating, justification = parts
            # Strip any leading or trailing whitespace from the rating
            rating = rating.strip()

        return score_to_label(int(rating)), justification 

    def sentiment_score(chunk):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        distilled_student_sentiment_classifier = pipeline(
            model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", 
            return_all_scores=False,
            device=device
        )
        result = distilled_student_sentiment_classifier(chunk)[0]['label']
        if result == 'positive':
            return 0
        elif result == 'negative':
            return 2
        else:
            return 1

    def style_score(chunk):
        text_manipulation.download_pretrained_model()
        label = text_manipulation.predict(chunk)
        return label

    def source_reliability_score(chunk):
        with open('models/srcM.pkl', 'rb') as f:
            srcM = pickle.load(f)
        label = srcM.predict_text(chunk)[0]
        return label

    def political_bias_score(chunk):
        political_bias.download_pretrained_model()
        processed_article = political_bias.preprocess_article(header, chunk)
        label = political_bias.predict_label(processed_article)
        return label

    def credibility_score(authors):
        with open('models/credibility_model.pkl', 'rb') as f:
            cred_model = pickle.load(f)

        search_results = []
        for author in authors:
            search_results.append(credibility.search_wikipedia(author, num_results=15))

        search_pd = pd.DataFrame(search_results, columns=['text'])
        embedded_result = credibility.text_embedding(search_pd['text'])[:, :50]
        cred_scores = cred_model.predict(embedded_result)
        if len(cred_scores) == 1:
            return cred_scores[0]
        else:
            cred_score = np.mean(cred_scores)
            return cred_score

    print("Evaluating the article chunks")

    # Normalize scoring method
    # min_val, max_val = 4.0136, 6.259 # Get more accurate number
    # def normalization(score):
    #     if max_val - min_val == 0:  # Check for zero division
    #         return 0
    #     elif score > max_val:
    #         return 1
    #     else:
    #         return (score - min_val) / (max_val - min_val)
    # credibility_scr = normalization(credibility_score(authors))
    for i in range (len(content)):
        chunk = content[i]
        st.markdown(f"<span style='font-weight:bold; color:red;'>Given context:</span> {chunk}", unsafe_allow_html=True)
        st.write("Here is the evaluation:")
        for attempt in range(10):
            try:
                # style_scr = style_score(chunk)
                # sentiment_scr = sentiment_score(chunk)
                # source_reliability_scr = source_reliability_score(chunk)
                # political_bias_scr = political_bias_score(chunk)
                rating, justification = evaluate_claim(chunk, evidence[i])
                st.markdown(f"<span style='font-weight:bold; color:blue;'>The rating is:</span> {rating}", unsafe_allow_html=True)
                st.markdown(f"<span style='font-weight:bold; color:green;'>The justification:</span> {justification} \n\n", unsafe_allow_html=True)

                # pred_output = f"The political bias score is {political_bias_scr}, indicating the degree of political leanings in the content. The credibility score is {credibility_scr}, reflecting the trustworthiness and accuracy of the information. The text manipulation score is {style_scr}, which assesses whether the text has been manipulated. The sentiment score is {sentiment_scr}, revealing the overall tone and mood of the content. Lastly, the source reliability score is {source_reliability_scr}, evaluating the dependability and consistency of the source."
                # print(pred_output)
                break
            except ValueError as e:
                if attempt == 9:  # Last attempt
                    st.write(f"Failed to evaluate chunk after 5 attempts: {e}")
if __name__ == "__main__":
    test_url = 'https://www.cnn.com/2024/01/17/politics/biden-ukraine-white-house-meeting/index.html'
    test_text = "In the heart of an ancient forest, where the trees whispered secrets of a bygone era, there lay a hidden glade, bathed in the ethereal glow of the moonlight. A gentle breeze danced through the leaves, carrying with it the sweet scent of blooming flowers and the distant sound of a babbling brook. The stars above twinkled like a tapestry of diamonds, casting a serene light over the verdant undergrowth. Amidst this tranquil setting, a majestic stag emerged from the shadows, its antlers glistening with dewdrops. It moved gracefully, as if in tune with the rhythm of the forest, pausing occasionally to listen to the soft murmur of the wind. In the distance, an owl hooted, adding a layer of mystery to the night's symphony. As the night deepened, a faint glow appeared on the horizon, heralding the arrival of dawn. The first rays of the sun filtered through the canopy, painting the sky in hues of pink and gold. The forest slowly awakened, with birds chirping and squirrels scampering about, each creature starting its day in this secluded haven. In this magical glade, time seemed to stand still, offering a moment of peace and reflection for those who stumbled upon its beauty. It was a reminder of the wonders of nature, a sanctuary untouched by the chaos of the outside world. Here, in the embrace of the forest, one could find solace and rejuvenation, a connection to the earth that was both ancient and eternal."
    final_pipeline_script(url = None, text = test_text)