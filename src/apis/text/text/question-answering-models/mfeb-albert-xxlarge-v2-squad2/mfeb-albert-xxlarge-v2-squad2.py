import json


def predict(context: str, question: str) -> dict:
    """
    Using the given `context`, answer the provided `question`.

    :param context: context to use to answer the question
    :param question: question to answer
    :return: JSON formatted str containing both the answer and the confidence score.
    """

    from happytransformer import HappyQuestionAnswering

    happy_qa = HappyQuestionAnswering("ALBERT", "mfeb/albert-xxlarge-v2-squad2")
    result = happy_qa.answer_question(context, question)
    
    return {"prediction": result[0].answer, "prediction_raw": result}
