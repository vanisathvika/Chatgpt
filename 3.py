from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline

#load pre-trained Model and Tokenizer

model_name='gpt2-medium'
tokenizer=GPT2Tokenizer.from_pretrained(model_name)
model=GPT2LMHeadModel.from_pretrained(model_name)

# Initialize the text generation pipline

text_genration_pipeline=pipeline('text-generation',model=model,tokenizer=tokenizer)

class ChatBot:
    def __init__(self,model_name='gpt2-medium'):
        self.tokenizer=GPT2Tokenizer.from_pretrained(model_name)
        self.model=GPT2LMHeadModel.from_pretrained(model_name)
        self.pipeline=pipeline('text-generation',model=self.model,tokenizer=self.tokenizer)
        self.context=""
    def get_response(self,user_input):
        #update context
        self.context +=f"User:{user_input}\nBot: "
        #Generate response
        response=self.pipeline(self.context,max_length=500,truncation=True,pad_token_id=self.tokenizer.eos_token_id,num_return_sequences=1)
        bot_response=response[0]['generated_text'].split("Bot: ")[-1].split("User: ")[0].strip()
        self.context += f"{bot_response}\n"
        return bot_response

#Initialize the chatbot

chatbot=ChatBot()

#Example multi-sentence converstion

conversation_history=[
    "Hello! How are you?",
    "Im doing great, thanks! What about you?",
    "Im good as well. What have you been up to?",
    "Just working on some projects. How about you?",
    "Same here. It's been a busy week."
]

for user_input in conversation_history:
    bot_response=chatbot.get_response(user_input)
    print(f"User:{user_input}")
    print(f"Bot:{bot_response}\n")


