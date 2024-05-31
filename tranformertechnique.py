from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Load model and tokenizer
model_path = 'Application//samsum-fine-tuned'  

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

def generate_transformer_summary(text, chunk_size=128, max_length=60, min_length=30):
    try:
        # Tokenize the input text and split it into chunks
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Generate summaries for each chunk
        summaries = []
        for i in range(0, input_ids.size(1), chunk_size):
            chunk_input_ids = input_ids[:, i:i+chunk_size]
            chunk_attention_mask = attention_mask[:, i:i+chunk_size]

            # Skip empty chunks
            if chunk_input_ids.size(1) == 0:
                continue

            chunk_summary = summarizer(
                tokenizer.decode(chunk_input_ids.squeeze(), skip_special_tokens=True),
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )[0]['summary_text']
            
            summaries.append(chunk_summary)

        # Combine summaries of all chunks
        summary_text = " ".join(summaries)
        return summary_text

    except Exception as e:
        return "An error occurred: {}".format(e)
