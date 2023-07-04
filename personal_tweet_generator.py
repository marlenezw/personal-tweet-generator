import os
from apikey import OpenAIapikey
from langchain import PromptTemplate
import json 
from langchain import FewShotPromptTemplate
from langchain.llms import OpenAI
from langchain import FewShotPromptTemplate
import streamlit as st 
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
# from langchain.utilities import WikipediaAPIWrapper

os.environ['OPENAI_API_KEY'] = OpenAIapikey



# with open('tweets.json') as tweet_file:
#     tweets= json.load(tweet_file)

def get_tweets_and_prefix(tweets, query, context):
    #these are the elements we need from the data (its like a dictionary in a list)
    tweet_examples =[
        {
            'query': f'{query}',
            'answer': tweet,
        }
        for tweet in tweets
    ]

    # #sort tweets in descending order 
    # tweet_examples.sort(key=lambda t: t['likes'], reverse=True)

    # #skipped the first one because its an anomoly but get the top 5 tweets 
    # examples = tweet_examples[6:12]

    # #remove the likes and retweets 
    # likes = [x.pop('likes') for x in examples]
    # rts = [x.pop('retweets') for x in examples]

    prefix = f"""{context}
    Here are some
    examples: 
    """

    return tweet_examples, prefix

def start_model(examples, prefix):
    # create a example template
    example_template = """
    User: {query}
    AI: {answer}
    """

    # create a prompt example from above template
    example_prompt = PromptTemplate(
        input_variables=["query", "answer"],
        template=example_template
    )

    # and the suffix our user input and output indicator
    suffix = """
    User: {query}
    AI: """

    # now create the few shot prompt tweet template
    tweet_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["query"],
        example_separator="\n\n"
    )

    return tweet_template

def generate_code(tweet):
    #Code template
    code_template = PromptTemplate(
        input_variables = ['tweet'],
        template='''write a few lines of code that show how the package mentioned works: {tweet} 
        '''
    )

    #LLMs 
    llm = OpenAI(temperature=0.4)
    code_chain = LLMChain(llm=llm, prompt=code_template, verbose=True, output_key='code')

    return code_chain


# initialize the models
openai = OpenAI(
model_name="gpt-3.5-turbo",
    openai_api_key=OpenAIapikey
    )


#app framework
st.title('🦜️🔗 Personal Tweet Generator')

'''
## Add 5 tweet examples

Example: 

`I've been having so much fun using @LangChainAI to build random apps for myself❤️✨

Here's a personal tweet generator I built with an example template of my tweets, 
@streamlit and GPT Turbo. It's genuinely doing a great job!!!

A look at some of the python code below👀` 
'''

c = st.container()
tweet1 = c.text_area('Tweet 1')
tweet2 = c.text_area('Tweet 2')
tweet3 = c.text_area('Tweet 3')
tweet4 = c.text_area('Tweet 4')
tweet5 = c.text_area('Tweet 5')

tweets = [tweet1, tweet2, tweet3, tweet4, tweet5]

'''
## Add context about your tweets

Example: 
The following are tweets from a woman named Marlene.
Marlene is a software engineer interested in Machine Learning, Python and animations.
She typically writes helpful and creative tweets about how to use Python and Machine Learning. 
She tries to share relatively new open source models that do interesting things. 
Marlene shares short code snippets that show how to use the models in a few lines of code.
'''
context = st.text_area('Add your context here')

option = st.selectbox(
    'Would you like code generated that demostrates your tweet?',
    ('No', 'Yes'))

st.write('You selected:', option)

'## What should the model call you?'
name = st.text_input('What should the model call you?')

# Show response to the screen if theres a prompt
if tweets and context and name:
    query = f'write me a tweet in the style of {name}'
    examples, prefix = get_tweets_and_prefix(tweets, query, context)
    tweet_template = start_model(examples, prefix)
    tweet = openai(tweet_template.format(query=query))

    if option == "Yes":
        code_chain = generate_code(tweet)
        code = code_chain.run(tweet)

    st.write(tweet)
    st.code(code, language='python')

 



