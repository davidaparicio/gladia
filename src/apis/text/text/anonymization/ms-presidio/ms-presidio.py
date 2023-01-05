import json
from logging import getLogger
from typing import Dict

from gladia_api_utils.model_management import get_spacy_cache_dir_for_model
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

logger = getLogger(__name__)


def predict(text: str, language: str = "xxx", entities: str = "") -> Dict[str, str]:
    """
    Anonymize the given text.

    Args:
        text (str): The text to anonymize.
        language (str): The language of the text ISO 639-3 language code of the language to load the model for. Optional, default is "xxx" multilingual.
        ISO-3 convention. Optional, default is "xxx" multilingual.
        entities (str): The csv list of entities to anonymize. Optional (default: all,
            see https://microsoft.github.io/presidio/supported_entities/
            can be a subset of:
                Entity Type	Description	Detection Method
                - CREDIT_CARD	A credit card number is between 12 to 19 digits. https://en.wikipedia.org/wiki/Payment_card_number	Pattern match and checksum
                - CRYPTO	A Crypto wallet number. Currently only Bitcoin address is supported	Pattern match, context and checksum
                - DATE_TIME	Absolute or relative dates or periods or times smaller than a day.	Pattern match and context
                - EMAIL_ADDRESS	An email address identifies an email box to which email messages are delivered	Pattern match, context and RFC-822 validation
                - IBAN_CODE	The International Bank Account Number (IBAN) is an internationally agreed system of identifying bank accounts across national borders to facilitate the communication and processing of cross border transactions with a reduced risk of transcription errors.	Pattern match, context and checksum
                - IP_ADDRESS	An Internet Protocol (IP) address (either IPv4 or IPv6).	Pattern match, context and checksum
                - NRP	A person’s Nationality, religious or political group.	Custom logic and context
                - LOCATION	Name of politically or geographically defined location (cities, provinces, countries, international regions, bodies of water, mountains	Custom logic and context
                - PERSON	A full person name, which can include first names, middle names or initials, and last names.	Custom logic and context
                - PHONE_NUMBER	A telephone number	Custom logic, pattern match and context
                - MEDICAL_LICENSE	Common medical license numbers.	Pattern match, context and checksum
                - URL	A URL (Uniform Resource Locator), unique identifier used to locate a resource on the Internet	Pattern match, context and top level url validation
                - US_BANK_NUMBER	A US bank account number is between 8 to 17 digits.	Pattern match and context
                - US_DRIVER_LICENSE	A US driver license according to https://ntsi.com/drivers-license-format/	Pattern match and context
                - US_ITIN	US Individual Taxpayer Identification Number (ITIN). Nine digits that start with a "9" and contain a "7" or "8" as the 4 digit.	Pattern match and context
                - US_PASSPORT	A US passport number with 9 digits.	Pattern match and context
                - US_SSN	A US Social Security Number (SSN) with 9 digits.	Pattern match and context
                - UK_NHS	A UK NHS number is 10 digits.	Pattern match, context and checksum
                - NIF	A spanish NIF number (Personal tax ID) .	Pattern match, context and checksum
                - FIN/NRIC	A National Registration Identification Card	Pattern match and context
                - AU_ABN	The Australian Business Number (ABN) is a unique 11 digit identifier issued to all entities registered in the Australian Business Register (ABR).	Pattern match, context, and checksum
                - AU_ACN	An Australian Company Number is a unique nine-digit number issued by the Australian Securities and Investments Commission to every company registered under the Commonwealth Corporations Act 2001 as an identifier.	Pattern match, context, and checksum
                - AU_TFN	The tax file number (TFN) is a unique identifier issued by the Australian Taxation Office to each taxpaying entity	Pattern match, context, and checksum
                - AU_MEDICARE	Medicare number is a unique identifier issued by Australian Government that enables the cardholder to receive a rebates of medical expenses under Australia's Medicare system	Pattern match, context, and checksum
            )

    Returns:
        Dict[str, str]: The anonymized text.
    """

    # Create configuration containing engine name and models
    # presidio-analyzer only supports en and es
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [
            {
                "lang_code": "es",
                "model_name": get_spacy_cache_dir_for_model("es_core_news_md"),
            },
            {
                "lang_code": "en",
                "model_name": get_spacy_cache_dir_for_model("en_core_web_lg"),
            },
        ],
    }

    # Create NLP engine based on configuration
    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine_with_spanish = provider.create_engine()

    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine_with_spanish, supported_languages=["en", "es"]
    )

    entities = entities.upper().replace(" ", "").split(",")

    if language.lower() == "esp":
        language = "es"
    else:
        language = "en"

    # Call analyzer to get results
    if entities:
        results = analyzer.analyze(text=text, entities=entities, language=language)
    else:
        results = analyzer.analyze(text=text, language=language)

    # Analyzer results are passed to the AnonymizerEngine for anonymization

    anonymizer = AnonymizerEngine()

    anonymized_text = json.loads(
        anonymizer.anonymize(text=text, analyzer_results=results).to_json()
    )

    return {
        "prediction": anonymized_text["text"],
        "prediction_raw": anonymized_text,
    }