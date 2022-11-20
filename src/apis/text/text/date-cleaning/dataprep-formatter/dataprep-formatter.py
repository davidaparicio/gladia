from typing import Dict

import pandas as pd
from dataprep.clean import clean_date, validate_date


def predict(date: str) -> Dict[str, str]:
    """
    Format the input string as a clean date using dataprep

    Args:
        date (str): The date to format
    Returns:
        Dict[str, str]: The formated date
    """

    df = pd.DataFrame({"date": [date]})

    prediction = clean_date(df, "date")["date_clean"][0]
    prediction_validate = bool(validate_date(df["date"])[0])

    return {
        "prediction": prediction,
        "raw_prediction": {"clean_date": prediction, "valid_date": prediction_validate},
    }
