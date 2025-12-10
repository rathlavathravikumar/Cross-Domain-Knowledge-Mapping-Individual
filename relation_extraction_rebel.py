from transformers import pipeline
 
re_model = pipeline("text2text-generation", model="Babelscape/rebel-large")
text = "Albert Einstein developed the theory of relativity in 1905."
 
print("Extracted relations:")
print(re_model(text)[0]['generated_text'])
