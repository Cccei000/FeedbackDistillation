
actor_template = '[INST] {__query__} [/INST]'

reflector_template = {}
reflector_template['auto-j_13B'] = """Write critiques for a submitted response on a given user's query, and grade the response:
  
[BEGIN DATA]
***
[Query]: {__query__}
***
[Response]: {__response__}
***
[END DATA]

Write critiques for this response. After that, you should give a final rating for the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]"."""


reflector_template['ultraCM_13B'] = """User: A one-turn chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, very detailed, and polite answers to the user's questions.</s>
User: Given my answer to an instruction, your role is to provide specific and constructive feedback for me. You should find the best way for me to learn from your feedback and improve my performance. 

You should consider multiple aspects of my answer, including helpfulness, truthfulness, honesty, and to what extent the answer follows instructions.
---

### Instruction
{__query__}

### Answer
{__response__}
---

Please act as a teacher and provide specific and constructive feedback. Besides describing the weaknesses of the answer, you should also provide specific suggestions to guide me toward understanding how to improve. Please note, however, that your suggestions should help me better complete the instructions, but you should not introduce new requirements that are not mentioned in the instructions. Your feedback should focus on enhancing my ability to think critically and respond accurately. However, never explicitly provide the reference answer, nor do polite phrases be required. Only respond with concise feedback in chat style. Finally, score the overall quality of the answer from 1 to 10, where 1 is the worst and 10 is the best.

*Format*
### Feedback
Overall Score: [1-10]
[Your feedback]

---

### Feedback
Overall Score: 
</s>
Assistant: """


refinement_template = """[INST] Imagine you are a student who is refining your assignment according to your teacher's comment on your previous work. 
    
This is the question in your assignment: {__query__}
     
This is your previous answer: {__response__}
    
And your teacher's comment is: {__feedback__}
    
Please refine your answer to the question. [/INST]"""