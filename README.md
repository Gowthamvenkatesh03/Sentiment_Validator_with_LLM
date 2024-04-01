# Context
Automated Quality Control with Language Models (LLMs) modernizes quality assurance methods. By automating manual inspection, it enhances efficiency and accuracy. Traditional methods are slow, costly and prone to errors. LLMs swiftly analyze data, including textual feedback, to detect defects effectively. They scale effortlessly and adapt to business needs.

# Idea
Customer satisfaction is paramount for all industries, often gleaned from reviews on shopping apps or social media. Various web scraping tools like BeautifulSoup (bs4), Scrapy, or Selenium can be employed to facilitate data extraction, Machine Learning models analyze sentiment. However, accuracy can vary, necessitating manual validation before delivering results to businesses.

The idea is to automate quality control using advanced Language Models (LLMs). These offer advantages in efficiency, accuracy, and scalability.

By integrating LLMs into this process, we can streamline sentiment analysis and minimize errors. LLMs, such as GPT-3.5, can comprehend nuances in language, improving accuracy. Additionally, their scalability allows for processing large volumes of data efficiently.

Overall, automating quality control with LLMs promises to enhance the reliability and speed of sentiment analysis, ultimately improving decision-making processes for businesses.

# Overall Workflow
![ai_workflow](https://github.com/Gowthamvenkatesh03/Sentiment_Validator_with_LLM/assets/66058704/6eff58d5-4058-4e81-af6f-ed428e6fd4a6)

# Sentiment Validator Workflow
![image](https://github.com/Gowthamvenkatesh03/Sentiment_Validator_with_LLM/assets/66058704/b7911bf3-e3dc-4f8e-8486-d616d789e2f8)

# Installation
### 1.Clone this repository to your local machine using:
  git clone https://github.com/Gowthamvenkatesh03/Sentiment_Validator_with_LLM.git
### 2.Install the required dependencies using pip:
  pip install -r requirements.txt
### 3.Set up your OpenAI API credentials by creating a .env file in the project root and adding your API
  OPENAI_API_KEY=your_api_key_here

# Working Model Screenshots
### Run the Streamlit app by executing:
streamlit run Sentiment_Validator.py

### The web app will open in your browser
#### Below is the home page asking users to upload the files for validation
![image](https://github.com/Gowthamvenkatesh03/Sentiment_Validator_with_LLM/assets/66058704/d9c83a2c-5fdf-4637-aa5e-974855f79149)

### User to upload a manually validated file for reference to the LLM, Any Predefined Terms file and the new data set which needs to be validated
![image](https://github.com/Gowthamvenkatesh03/Sentiment_Validator_with_LLM/assets/66058704/6ce4397c-a297-4490-8ad4-804f07a3b62c)

### Once the required files are uploaded, click on Validate to button
![image](https://github.com/Gowthamvenkatesh03/Sentiment_Validator_with_LLM/assets/66058704/b3a7ee0b-1d59-4bd6-bc73-9bfe82261f8e)

### This process validate the new data set and provides the option to download validated file as csv. The validation summary will also be shown like below
![image](https://github.com/Gowthamvenkatesh03/Sentiment_Validator_with_LLM/assets/66058704/b5d308a1-a026-49a2-a637-47cffb96b92e)

Once the Download CSV button is clicked validated file will be available in your local.



Now the human can just have a quick look on the validation results and can deliver the file to business people.

