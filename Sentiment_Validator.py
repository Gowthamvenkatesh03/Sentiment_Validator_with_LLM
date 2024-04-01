"""
AI Automated Sentiment QC Module

Model used: GPT3.5
Framework: Langchain
Hosted on: Streamlit

"""
# Import the required libraries
import json
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import AzureOpenAI
from langchain_openai.chat_models import AzureChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.prompts import PromptTemplate
import pandas as pd
import os
import sys
import warnings
warnings.filterwarnings('ignore')
import traceback
from datetime import datetime, timedelta

# To Collect the validated reports
validated_reports = []
# To collect the correct and Incorrect values
correct_values = []
incorrect_values = []
# Default Date format
DATE_FORMAT = "%Y%m%d"

validation_file_name = "Sentiment_Analysis_Validated_File_{}.csv"
current_date = datetime.now()

def validate_review(review_text,theme,subtheme,positive_mention,negative_mention,polarity,predefined_themes,examples,validation_model):
    """
    Validates whether the review text is tagged to correct theme or not

    Args:
        review_text (str): The review text provied for the product
        review_theme (str): The theme tagged to the particular review
        predefined_themes (list): List of pre-defined themes
        examples (json): Example data validated by human

    Returns:
        response: The validated result indicating whether the review is tagged with correct theme or not

    Raises:
        ValueError: Raise error when any error found during validation
    """
    try:
        template = """You are good at Evaluation. 
        Refer the sample data provided - "{}", Learn how Theme, SubTheme, Postive Mention, Negative Mention and Polarity are mentioned correctly and appropriately to the provided review.
        Now I will provide you with the the new dataset. Evaluate and give a brief answer for anything asked.
        Note: You can use the sample data for reference, Incase if you are not sure of the answer.""".format(examples)
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        human_template = "{text}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
        chain = LLMChain(llm = validation_model, prompt=chat_prompt)
        
        # Create a separate prompt and validate each fields
        theme_validation_prompt = """This is the Sentiment Analysis Project. Read the review - "{0}" and analyse the Theme in the given review. Now evaluate and provide brief answers for the below questions.
                    FORMAT THE OUTPUT AS DICTIONARY WITH THE FOLLOWING KEYS:
                    Theme_Correct_Incorrect: Analyse the 'Theme' in the given review. Now answer, whether the Theme "{2}" is appropriate for the given review? Answer Correct Only If the Theme "{2}" match any of the pre-defined themes in the list - {1} and If Theme is correct and appropriate for the provided review. Else Answer as Incorrect.
                    Theme_Validation_Result: Why the given Theme is correct or incorrect? Provide the answer in less than 50 words. If Incorrect suggest the correct and appropriate value from the given pre-defined themes list - {1}.
                    """.format(review_text,predefined_themes,theme)
        theme_validation_report = chain.run(theme_validation_prompt)
        theme_validation_report = json.loads(theme_validation_report)

        subtheme_validation_prompt = """This is the Sentiment Analysis Project. Read the review - "{0}" and analyse the SubTheme in the given review. Now evaluate and provide brief answers for the below questions.
                    FORMAT THE OUTPUT AS DICTIONARY WITH THE FOLLOWING KEYS:
                    SubTheme_Correct_Incorrect: Analyse the 'SubTheme' in the given review. Now answer, Whether the SubTheme Predicted as "{1}" is appropriate and correct for the provided review?
                    Answer Correct if yes, Incorrect if not.
                    SubTheme_Validation_Result: Why the given SubTheme is correct or incorrect? Provide the answer in less than 50 words. If Incorrect suggest the correct and appropriate SubTheme for the given review.
                    """.format(review_text,subtheme)
        subtheme_validation_report = chain.run(subtheme_validation_prompt)
        subtheme_validation_report = json.loads(subtheme_validation_report)

        if str(positive_mention) == 'nan':
            prompt_question = """Positive_Mention_Correct_Incorrect: Positive Mention is not found. Whether the given review contain any Positive Mention?
                            Answer Correct if there is no Positive Mention, Else Incorrect.
                            Positive_Mention_Validation_Result: Why it is correct or incorrect? Provide the answer in less than 50 words. If Incorrect suggest the correct and appropriate negative word from the given review."""
        else:
            prompt_question = """Positive_Mention_Correct_Incorrect: Analyse the 'Positive Mention' in the given review. Now answer, Whether the Positive Mention "{0}" is correct and appropriate for the provided review?
                            Answer Correct if yes, Incorrect if not.
                            Positive_Mention_Validation_Result: Why the given Positive Mention "{0}" is correct or incorrect? Provide the answer in less than 50 words. If Incorrect suggest the correct and appropriate negative word from the given review.
                            """.format(positive_mention)
            
        positive_mention_validation_prompt = """This is the Sentiment Analysis Project. Read the review - "{0}" and analyse the Positive Mention in the given review. Now evaluate and provide brief answers for the below questions.
                    FORMAT THE OUTPUT AS DICTIONARY WITH THE FOLLOWING KEYS:
                    {1}
                    """.format(review_text,prompt_question)
        positive_mention_validation_report = chain.run(positive_mention_validation_prompt)
        positive_mention_validation_report = json.loads(positive_mention_validation_report)

        if str(negative_mention) == 'nan':
            prompt_question = """Negative_Mention_Correct_Incorrect: Negative Mention is not Found. Whether the given review contain any Negative Mention?
                            Answer Correct if there is no Negative Mention, Else Incorrect.
                            Negative_Mention_Validation_Result: Why it is correct or incorrect? Provide the answer in less than 50 words. If Incorrect suggest the correct and appropriate negative word from the given review."""
        else:
            prompt_question = """Negative_Mention_Correct_Incorrect: Whether the Negative Mention "{0}" is correct and appropriate for the provided review?
                            Answer Correct if yes, Incorrect if not.
                            Negative_Mention_Validation_Result: Why the given Negative Mention "{0}" is correct or incorrect? Provide the answer in less than 50 words. If Incorrect suggest the correct and appropriate negative word from the given review.
                            """.format(negative_mention)                  

        negative_mention_validation_prompt = """This is the Sentiment Analysis Project. Read the review - "{0}" and analyse the Negative Mention in the given review. Now evaluate and provide brief answers for the below questions.
                    FORMAT THE OUTPUT AS DICTIONARY WITH THE FOLLOWING KEYS:
                    {1}
                    """.format(review_text,prompt_question)
        negative_mention_validation_report = chain.run(negative_mention_validation_prompt)
        negative_mention_validation_report = json.loads(negative_mention_validation_report)

        polarity_validation_prompt = """This is the Sentiment Analysis Project. Read the review - "{0}" and analyse the Polarity in the given review. Now evaluate and provide brief answers for the below questions.
                    FORMAT THE OUTPUT AS DICTIONARY WITH THE FOLLOWING KEYS:
                    Polarity_Correct_Incorrect: Analyse the 'Polarity' in the given review. Now answer whether the Polarity Predicted as "{1}" is Correct or Not for the given review? Answer Correct Only If the Polarity "{1}" match any of the value in the list - ['Positive','Negative','Neutral'] and If Polarity is tagged correctly as per the review. Else Answer as Incorrect.
                    Polarity_Validation_Result: Why the given Polarity is correct or incorrect? Provide the answer in less than 50 words. If Incorrect suggest the correct and appropriate value from the given pre-defined polarity list - ['Positive','Negative','Neutral']
                    """.format(review_text,polarity)
        polarity_validation_report = chain.run(polarity_validation_prompt)
        polarity_validation_report = json.loads(polarity_validation_report)

        # Join the response from the validation of each field
        joined_validation_report_dict = {**theme_validation_report, 
                                         **subtheme_validation_report, 
                                         **positive_mention_validation_report,
                                         **negative_mention_validation_report,
                                         **polarity_validation_report}
        
        # Convert the dictionary into a DataFrame
        df = pd.DataFrame([joined_validation_report_dict])
        # Collect the response
        validated_reports.append(df)
        # Collect the correct and incorrect values separately
        req_cols = ['Theme_Correct_Incorrect',
            'SubTheme_Correct_Incorrect',
            'Positive_Mention_Correct_Incorrect',
            'Negative_Mention_Correct_Incorrect',
            'Polarity_Correct_Incorrect']
        for col in req_cols:
            if df[col].values[0] == "Correct":
                correct_values.append("Correct")
            else:
                incorrect_values.append("Incorrect")
    except Exception as err:
        print("Error while Validating the data. '%s", err)
        e_type, e_value, _ = sys.exc_info()
        print("Error Details: %s, %s", e_type, e_value)
        print("Error stack trace: %s", traceback.format_exc())
    

def summarize_validation(new_review_df):
    """
    Summarize the validated document
    """
    llm = AzureOpenAI(
        api_key=OPENAI_AI_KEY,
        api_version=API_VERSION,
        azure_endpoint="YOUR_API_ENDPOINT",
        deployment_name="TextDavinci3",
        default_headers={
            # Request headers
            'x-service-line': SERVICE_LINE,
            'x-brand': BRAND,
            'x-project': PROJECT,
            'Content-Type': 'application/json',
            'Cache-Control': 'no-cache',
            'api-version': 'v2',
            'Ocp-Apim-Subscription-Key': OPENAI_AI_KEY,
        }, temperature=0

    )
   
    prompt = PromptTemplate(
        template="Task: {task}\nResult:",
        input_variables=["task"], example_separator="\n")

    prompt_task_1 = """
        Given Correct_Values as {} and Incorrect_Values as {}\n
        Extract the following information:
        Total Validated Value is sum of Correct_Values+Incorrect_Values. Incorrect value is what percent of total value? Provide the value as Integer.""".format(len(correct_values),len(incorrect_values))
    chain = LLMChain(llm=llm, prompt=prompt)
    # Get the response
    response = chain.invoke(prompt_task_1)
    incorrect_percent = response.get('text')
    
    summary_reponse = """There are total '{total_rows}' rows validated, and there {unique_products} unique products.
    Below listed fields are validated - "Theme", "Mention_Subtheme",
                                            "Positive_Mention", "Negative_Mention", "Polarity".
    Based on the Validation found that the {incorrect_val_percent} of the values are tagged incorrectly to the given review.
    The Incorrect values are marked and suggestion are provided in the file - {filename}.
    """
    summarize_paragraph =summary_reponse.format(total_rows=new_review_df.shape[0], 
                        unique_products=len(list(new_review_df['Product_Name'].unique()))
                        , incorrect_val_percent=incorrect_percent,
                        filename=validation_file_name.format(current_date.strftime(DATE_FORMAT)))
    
    prompt_task_2 = """
        Structurize the following paragraph, and provide it in bullet points with the heading as Validation Summary.
        paragraph = {summarize_para}
        """.format(summarize_para=summarize_paragraph)
    chain = LLMChain(llm=llm, prompt=prompt)
    # Get the response
    response = chain.invoke(prompt_task_2)
    # Split the text into key-value pairs
    summarization = response.get('text')
    st.write(summarization)


if __name__ == "__main__":
    # Load API Config
    load_dotenv()

    # Get the arguments
    OPENAI_AI_KEY = os.getenv('OPENAI_API_KEY')
    DEPLOYMENT = os.getenv('DEPLOYMENT')
    BASE_URL = os.getenv('BASE_URL')
    API_VERSION = os.getenv('API_VERSION')
    SERVICE_LINE = os.getenv('SERVICE_LINE')
    BRAND = os.getenv('BRAND')
    PROJECT = os.getenv('PROJECT')

    # Set streamlit header
    st.header("Automated Sentiment QC")
    # Upload files for validation
    uploaded_files = st.file_uploader(
        "Upload the files for validation", accept_multiple_files=True)

    # Get the details on uploaded files
    for uploaded_file in uploaded_files:
        if 'pre_defined_themes' in uploaded_file.name:
            pre_defined_themes_df = pd.read_csv(uploaded_file)
        if 'Example' in uploaded_file.name:
            example_df = pd.read_excel(uploaded_file)
            example_df = example_df[['Review_Text','Theme','Mention_Subtheme','Positive_Mention','Negative_Mention','Polarity']]
            example_df = example_df.astype(str)
            # example_json_data = example_df.to_dict(orient="records")
        if 'Validating' in uploaded_file.name:
            new_review_df = pd.read_excel(uploaded_file)
        # Display the file name on browser
        st.write("loaded:", uploaded_file.name)

    # Click on Validate button
    if st.button('Validate'):
        with st.spinner('Validating'):
            # Create LLM client with AzureChatOpenAI
            validation_model = AzureChatOpenAI(
                api_key = OPENAI_AI_KEY,
                api_version = API_VERSION,
                azure_endpoint = "YOUR_API_ENDPOINT",
                deployment_name = DEPLOYMENT,
                default_headers = {
                # Request headers
                'x-service-line': SERVICE_LINE,
                'x-brand': BRAND,
                'x-project': PROJECT,
                'Content-Type': 'application/json',
                'Cache-Control': 'no-cache',
                'api-version': 'v2',
                'Ocp-Apim-Subscription-Key': OPENAI_AI_KEY,
                }, temperature =0
                
            )
            
            # Apply validation function to each review
            new_review_df = new_review_df.astype(str)
            new_review_df.apply(
                lambda row: validate_review(row['Review_Text'],
                                            row['Theme'],
                                            row["Mention_Subtheme"],
                                            row["Positive_Mention"],
                                            row["Negative_Mention"],
                                            row["Polarity"],
                                            pre_defined_themes_df['Pre_Defined_Themes'].tolist(
                ),
                    example_df, validation_model), axis=1)

            # Concatenate the validated DataFrame with the original DataFrame
            validated_reports_df = pd.concat(validated_reports, axis=0,
                                    ignore_index=True, sort=False)
            new_review_df = pd.concat([new_review_df, validated_reports_df], axis=1)
        st.success('Validation Completed!', icon="✅")
        with st.spinner('Summarizing the validation result'):
            # Summarization
            summarize_validation(new_review_df)
        st.success('Summarized the Validation Results', icon="✅")
        # Download the validated file
        st.sidebar.download_button('Download CSV', new_review_df.to_csv(index=False),
                        file_name=validation_file_name.format(current_date.strftime(DATE_FORMAT)))