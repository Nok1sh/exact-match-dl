import string

def normalize(text):

    punct = set(string.punctuation)
    punct.add(" ")
    normalize_text = set([x for x in text.lower() if x not in punct])

    return normalize_text

def calculate_f1_text(prediction, reference):
    prediction_tokens = normalize(prediction[0])
    reference_tokens = normalize(reference[0])
    
    intersection = len(prediction_tokens & reference_tokens)
    
    
    if intersection == 0:
        return 0
    
    precision = 1.0 * intersection / len(prediction_tokens)
    recall = 1.0 * intersection / len(reference_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1