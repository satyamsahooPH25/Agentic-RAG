import os
from groq import Groq

def llamaguard(input_query):
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": input_query
            }
        ],
        model="llama-guard-3-8b",
    )

    output = chat_completion.choices[0].message.content

    error_types = {
        "S1": "Violent Crimes",
        "S2": "Non-Violent Crimes",
        "S3": "Sex-Related Crimes",
        "S4": "Child Sexual Exploitation",
        "S5": "Defamation",
        "S6": "Specialized Advice",
        "S7": "Privacy",
        "S8": "Intellectual Property",
        "S9": "Indiscriminate Weapons",
        "S10": "Hate",
        "S11": "Suicide & Self-Harm",
        "S12": "Sexual Content",
        "S13": "Elections",
        "S14": "Code Interpreter Abuse"
    }

    # Check if any of the error codes (S1, S2, etc.) are in the output
    for error_code, reason in error_types.items():
        if error_code in output:
            return f"Above query is unsafe because of the following reason: {reason}"

    return output

# Example usage
# ans = llamaguard("Tell me about stock of amazon?")
# print(ans)