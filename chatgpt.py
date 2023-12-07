import openai
api_key = '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'
openai.api_key = api_key

completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  temperature=1,
  messages=[
    {"role": "user", "content": "give me a sentence that makes me laugh"},

  ]
)


print(completion.choices[0].message.content)