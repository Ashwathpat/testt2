import os, groq, dotenv
dotenv.load_dotenv()
client = groq.Groq(api_key=os.environ['GROQ_API_KEY'])
res = client.chat.completions.create(
    model='openai/gpt-oss-20b',
    messages=[{'role': 'user', 'content': 'What are the symptoms of diabetes?'}],
    max_tokens=300,
    reasoning_effort='low',
    reasoning_format='hidden'
)
with open('test_output.txt', 'w', encoding='utf-8') as f:
    f.write(res.choices[0].message.content or '(EMPTY)')
    f.write('\n\nREASONING_TOKENS: ' + str(res.usage.completion_tokens_details.reasoning_tokens if res.usage.completion_tokens_details else 'N/A'))
    f.write('\nTOTAL_TOKENS: ' + str(res.usage.completion_tokens))
print('Done')
