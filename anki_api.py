import json
import urllib.request
import pandas as pd
import requests


def request(action, **params):
    return {"action": action, "params": params, "version": 6}


def invoke(action, **params):
    requestJson = json.dumps(request(action, **params)).encode("utf-8")
    response = json.load(
        urllib.request.urlopen(
            urllib.request.Request("http://localhost:8765", requestJson)
        )
    )
    if len(response) != 2:
        raise Exception("response has an unexpected number of fields")
    if "error" not in response:
        raise Exception("response is missing required error field")
    if "result" not in response:
        raise Exception("response is missing required result field")
    if response["error"] is not None:
        raise Exception(response["error"])
    return response["result"]


# result = invoke('deckNames')
# print('got list of decks: {}'.format(result))


def load_anki_query_to_dataframe(query):
    # Function to send a request to Anki-Connect
    def invoke(action, **params):
        return requests.post(
            "http://localhost:8765",
            json={"action": action, "version": 6, "params": params},
        ).json()

    # Get all card IDs from the specified query
    card_ids_response = invoke("findCards", query=query)
    card_ids = card_ids_response["result"]

    if not card_ids:
        return pd.DataFrame()  # Return empty DataFrame if no cards are found

    # Get detailed information about each card
    cards_info_response = invoke("cardsInfo", cards=card_ids)
    cards_info = cards_info_response["result"]

    # Prepare data for the DataFrame
    data = []
    field_names = set()

    for card in cards_info:
        fields = card["fields"]
        card_data = {"card_number": card["cardId"]}  # Include Anki card number

        for field_name, field_content in fields.items():
            card_data[field_name] = field_content["value"]
            field_names.add(field_name)

        data.append(card_data)

    # Create DataFrame with "Anki Card Number" as the first column
    df = pd.DataFrame(data, columns=["card_number"] + sorted(field_names))

    df.card_number = df.card_number.astype(str)
    return df
