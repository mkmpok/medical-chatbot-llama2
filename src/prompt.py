# prompt_template = """
# Use the following context to answer the question at the end.
# If you don't know the answer, say "I don't know".
#
# Context:
# {context}
#
# Question:
# {question}
#
# Helpful Answer:
# """

CONDENSE_QUESTION_PROMPT = """You are a helpful medical chatbot.
Respond **briefly and clearly** using simple medical reasoning.
Avoid asking follow-up questions unless necessary.

User: {question}
Chatbot:"""
