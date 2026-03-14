from langchain_core.prompts import ChatPromptTemplate


# 1. Evidence filter agent
EVIDENCE_FILTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an evidence filter for an SMS spam detection system.

Your job is to examine a retrieved SMS example and decide whether it is useful evidence
for classifying the user's SMS message.

A retrieved example is useful if it is semantically relevant to the user's SMS, such as:
- similar scam/spam intent
- similar promotional or phishing language
- similar legitimate/personal messaging style
- similar urgency, reward, prize, or claim wording
- similar conversational non-spam structure

Return only a JSON object in exactly this format:
{{"binary_score": "yes"}}
or
{{"binary_score": "no"}}""",
        ),
        (
            "human",
            """User SMS:
{question}

Retrieved SMS example:
{document}""",
        ),
    ]
)


# 2. Classification agent
CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an SMS classification agent.

Classify the user's SMS as either:
- spam
- ham

Use the filtered evidence examples to support your decision.
Be careful and concise. Focus on the wording, intent, tone, and scam/legitimate patterns.

Return only a JSON object in exactly this format:
{{"label": "spam", "explanation": "short explanation here"}}
or
{{"label": "ham", "explanation": "short explanation here"}}""",
        ),
        (
            "human",
            """User SMS:
{question}

Filtered evidence examples:
{documents}""",
        ),
    ]
)


# 3. Verification agent
VERIFIER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a verification agent for an SMS spam detection system.

Your job is to determine whether the classification and explanation are supported
by the provided evidence examples.

Return only a JSON object in exactly this format:
{{"binary_score": "yes"}}
or
{{"binary_score": "no"}}""",
        ),
        (
            "human",
            """User SMS:
{question}

Filtered evidence examples:
{documents}

Classification result:
{classification}

Explanation:
{explanation}""",
        ),
    ]
)
