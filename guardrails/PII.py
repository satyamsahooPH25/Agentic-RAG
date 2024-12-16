from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine


def pii_remover(text):

    presidio_analyzer = AnalyzerEngine()
    presidio_anonymizer = AnonymizerEngine()

    analysis = presidio_analyzer.analyze(text=text, language='en')

    filtered_analysis = [result for result in analysis if result.entity_type not in ['DATE_TIME', 'IN_PAN']]

    anonymized_result = presidio_anonymizer.anonymize(text=text, analyzer_results=filtered_analysis)

    return anonymized_result.text


# text = "What percentage of Alphabet’s revenue in 2022 came from Google Services versus Google Cloud, and how has this balance shifted over the past 5 years?"

# text = pii_remover(text)

# print(text)