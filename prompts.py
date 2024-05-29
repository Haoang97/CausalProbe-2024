PROMPT_DICT = {
    # CausalNet
    "prompt_mcqa_causalnet": (
       "Context: {context}\nQuestion: what is the {ask-for}?\nAnswer Choices:\n0. {choice_id0}\n1. {choice_id1}\n2.{choice_id2}\n"
       "Pick one choice and do not explain. Answer:"
    ),
    "prompt_mcqa_cot_causalnet": (
        "Context: {context}\nQuestion: what is the {ask-for}?\nAnswer Choices:\n0. {choice_id0}\n1. {choice_id1}\n2.{choice_id2}\n"
        "Let's think it step-by-step. Answer:"
    ),
    "prompt_mcqa_retrieval_causalnet": (
        "Retrieved information:\n{paragraph}\n"
        "Answer the following question based on its context, and the retrieved information as a reference (if useful).\n"
        "Context: {context}\nQuestion: what is the {ask-for}?\nAnswer Choices:\n0. {choice_id0}\n1. {choice_id1}\n2.{choice_id2}\n"
        "Pick one choice and do not explain. Answer:"
    ),
    "prompt_mcqa_cot_retrieval_causalnet": (
        "Retrieved information:\n{paragraph}\n"
        "Answer the following question based on its context, and the retrieved information as a reference (if useful).\n"
        "Context: {context}\nQuestion: what is the {ask-for}?\nAnswer Choices:\n0. {choice_id0}\n1. {choice_id1}\n2.{choice_id2}\n"
        "Let's think it step-by-step. Answer:"
    ),
    "prompt_mcqa_g2reasoner_causalnet": (
        "General knowledge is as follows:\n{paragraph}\n"
        "Answer the following question based on its context and the general knowledge (if useful).\n"
        "You should objectively discern the most probable causal relationship based on the available evidence to work out correct answer.\n"
        "Context: {context}\nQuestion: what is the {ask-for}?\nAnswer Choices:\n0. {choice_id0}\n1. {choice_id1}\n2.{choice_id2}\n"
        "Pick one choice and do not explain. Answer:"
    ),
    "prompt_mcqa_claude_causalnet": (
        "General knowledge:\n{paragraph}\n\n"
        "Context:\n{context}\n\n"
        "Question: Based on the provided context, determine the most likely {ask-for}.\n\n"
        "Answer Choices:\n0. {choice_id0}\n1. {choice_id1}\n2.{choice_id2}\n\n"
        "To arrive at the correct answer, carefully analyze the available information and logically infer the most probable causal relationship."
        "Pick one choice and do not explain. Answer:"
    ),
    # CausalProbe w/o contexts
    "prompt_mcqa_causalprobe_NOcontext": (
        "Question: {question}\nAnswer Choices:\n1. {choice_1}\n2. {choice_2}\n3.{choice_3}\n4.{choice_4}\n"
        "Pick one choice and do not explain. Answer:"
    ),
    "prompt_mcqa_cot_causalprobe_NOcontext": (
        "Question: {question}\nAnswer Choices:\n1. {choice_1}\n2. {choice_2}\n3.{choice_3}\n4.{choice_4}\n"
        "Let's think it step-by-step. Answer:"
    ),
    "prompt_mcqa_retrieval_causalprobe_NOcontext": (
        "Retrieved information:\n{paragraph}\n"
        "Answer the following question based on the retrieved information as a reference (if useful).\n"
        "Question: {question}\nAnswer Choices:\n1. {choice_1}\n2. {choice_2}\n3.{choice_3}\n4.{choice_4}\n"
        "Pick one choice and do not explain. Answer:"
    ),
    "prompt_mcqa_cot_retrieval_causalprobe_NOcontext": (
        "Retrieved information:\n{paragraph}\n"
        "Answer the following question based on the retrieved information as a reference (if useful).\n"
        "Question: {question}\nAnswer Choices:\n1. {choice_1}\n2. {choice_2}\n3.{choice_3}\n4.{choice_4}\n"
        "Let's think it step-by-step. Answer:"
    ),
    "prompt_mcqa_g2reasoner_causalprobe_NOcontext": (
        "You are an intelligent causal reasoner. To arrive at the correct answer, carefully analyze the available information and logically infer the most probable causal relationship. Related general knowledge can be a reference if useful.\n\n"
        "General knowledge:\n{paragraph}\n"
        "Question: {question}\nAnswer Choices:\n1. {choice_1}\n2. {choice_2}\n3.{choice_3}\n4.{choice_4}\n"
        "Pick one choice and do not explain. Answer:"
    ),
    # CausalProbe /w contexts
    "prompt_mcqa_causalprobe": (
        "Context: {context}\nQuestion: {question}\nAnswer Choices:\n1. {choice_1}\n2. {choice_2}\n3.{choice_3}\n4.{choice_4}\n"
        "Pick one choice and do not explain. Answer:"
    ),
    "prompt_mcqa_cot_causalprobe": (
        "Context: {context}\nQuestion: {question}\nAnswer Choices:\n1. {choice_1}\n2. {choice_2}\n3.{choice_3}\n4.{choice_4}\n"
        "Let's think it step-by-step. Answer:"
    ),
    "prompt_mcqa_retrieval_causalprobe": (
        "Retrieved information:\n{paragraph}\n"
        "Answer the following question based on its context, and the retrieved information as a reference (if useful).\n"
        "Context: {context}\nQuestion: {question}\nAnswer Choices:\n1. {choice_1}\n2. {choice_2}\n3.{choice_3}\n4.{choice_4}\n"
        "Pick one choice and do not explain. Answer:"
    ),
    "prompt_mcqa_cot_retrieval_causalprobe": (
        "Retrieved information:\n{paragraph}\n"
        "Answer the following question based on its context, and the retrieved information as a reference (if useful).\n"
        "Context: {context}\nQuestion: {question}\nAnswer Choices:\n1. {choice_1}\n2. {choice_2}\n3.{choice_3}\n4.{choice_4}\n"
        "Let's think it step-by-step. Answer:"
    ),
    "prompt_mcqa_g2reasoner_causalprobe": (
        "General knowledge is as follows:\n{paragraph}\n"
        "You should objectively discern the most probable causal relationship based on the available evidence to work out correct answer.\n"
        "Context: {context}\nQuestion: {question}\nAnswer Choices:\n1. {choice_1}\n2. {choice_2}\n3.{choice_3}\n4.{choice_4}\n"
        "Pick one choice. Answer:"
    ),
    # e-Care
    "prompt_mcqa_ecare": (
        "Premise: {premise}\nQuestion: What is the {ask-for}?\nAnswer Choices:\n0. {hypothesis1}\n1. {hypothesis2}\n"
        "Pick one choice and do not explain. Answer:"
    ),
    "prompt_mcqa_cot_ecare": (
        "Premise: {premise}\nQuestion: What is the {ask-for}?\nAnswer Choices:\n0. {hypothesis1}\n1. {hypothesis2}\n"
        "Let's think it step-by-step. Answer:"
    ),
    "prompt_mcqa_retrieval_ecare": (
        "Retrieved information:\n{paragraph}\n"
        "Answer the following question based on its premise, and the retrieved information as a reference (if useful).\n"
        "Premise: {premise}\nQuestion: What is the {ask-for}?\nAnswer Choices:\n0. {hypothesis1}\n1. {hypothesis2}\n"
        "Pick one choice and do not explain. Answer:"
    ),
    "prompt_mcqa_cot_retrieval_ecare": (
        "Retrieved information:\n{paragraph}\n"
        "Answer the following question based on its premise, and the retrieved information as a reference (if useful).\n"
        "Premise: {premise}\nQuestion: What is the {ask-for}?\nAnswer Choices:\n0. {hypothesis1}\n1. {hypothesis2}\n"
        "Let's think it step-by-step. Answer:"
    ),
    "prompt_mcqa_g2reasoner_ecare": (
        "General knowledge is as follows:\n{paragraph}\n"
        "Answer the following question based on its premise, and the general knowledge as a reference (if useful).\n"
        "You should objectively analyze the most probable causal relationship based on the available evidence to reach the correct answer.\n"
        "Premise: {premise}\nQuestion: What is the {ask-for}?\nAnswer Choices:\n0. {hypothesis1}\n1. {hypothesis2}\n"
        "Let's think it step-by-step. Answer:"
    ),
    # COPA
    "prompt_mcqa_copa": (
        "Premise: {context}\nQuestion: What is the {asks-for}?\nAnswer Choices:\n0. {choice_0}\n1. {choice_1}\n"
        "Pick one choice and do not explain. Answer:"
    ),
    "prompt_mcqa_cot_copa": (
        "Premise: {context}\nQuestion: What is the {asks-for}?\nAnswer Choices:\n0. {choice_0}\n1. {choice_1}\n"
        "Let's think it step-by-step. Answer:"
    ),
    "prompt_mcqa_retrieval_copa": (
        "Retrieved information:\n{paragraph}\n"
        "Answer the following question based on its premise, and the retrieved information as a reference (if useful).\n"
        "Premise: {context}\nQuestion: What is the {asks-for}?\nAnswer Choices:\n0. {choice_0}\n1. {choice_1}\n"
        "Pick one choice and do not explain. Answer:"
    ),
    "prompt_mcqa_cot_retrieval_copa": (
        "Retrieved information:\n{paragraph}\n"
        "Answer the following question based on its premise, and the retrieved information as a reference (if useful).\n"
        "Premise: {context}\nQuestion: What is the {asks-for}?\nAnswer Choices:\n0. {choice_0}\n1. {choice_1}\n"
        "Let's think it step-by-step. Answer:"
    ),
    "prompt_mcqa_g2reasoner_copa": (
        "General knowledge is as follows:\n{paragraph}\n"
        "Answer the following question based on its premise, and the general knowledge as a reference (if useful).\n"
        "You should objectively analyze the most probable causal relationship based on the available evidence to reach the correct answer.\n"
        "Premise: {context}\nQuestion: What is the {asks-for}?\nAnswer Choices:\n0. {choice_0}\n1. {choice_1}\n"
        "Let's think it step-by-step. Answer:"
    ),
}