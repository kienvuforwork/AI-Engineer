import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

data = ["The cat sat on the mat.","Dogs are loyal and friendly animals.",
"The quick brown fox jumps over the lazy dog.",
"Machine learning algorithms can analyze large datasets.",
"Natural language processing helps computers understand human speech.",
"Coffee is a popular beverage enjoyed worldwide.",
"Python is a versatile programming language used in data science.",
"Traveling expands knowledge and broadens perspectives.",
"Music has the power to inspire and heal.",
"Climate change is a global challenge requiring urgent action."]

tfidfvec = TfidfVectorizer()
tfidfvec_fit = tfidfvec.fit_transform(data)

tfidf_bag = pd.DataFrame(tfidfvec_fit.toarray(), columns = tfidfvec.get_feature_names_out())
print(tfidf_bag)