from transformers import pipeline

# For long documents, consider a chunking approach:
def chunk_text(text, chunk_size=500):
    tokens = text.split()
    for i in range(0, len(tokens), chunk_size):
        yield " ".join(tokens[i:i + chunk_size])

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


summaries = []
for chunk in chunk_text(ARTICLE, 500):
    summary = summarizer(chunk, max_length=120, min_length=30, do_sample=False)
    summaries.append(summary[0]['summary_text'])

final_summary = " ".join(summaries)
print(final_summary)
