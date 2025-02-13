#!/usr/bin/env python
# coding: utf-8

import os
import re

import pandas as pd
from tqdm import tqdm

# LangChain and OpenAI modules for structured LLM outputs
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from IPython.display import Markdown, display

# Import the API function to load Anki deck data
from anki_api import load_anki_query_to_dataframe


# -----------------------------------------------------------------------------
# Class Definitions for Structured LLM Outputs
# -----------------------------------------------------------------------------


class AnkiContent(BaseModel):
    """Structured output for individual Anki card summaries."""

    title: str = Field(description="Title of the Anki card")
    body: str = Field(
        description="Concise bullet point summary of the given text in markdown"
    )
    short_summary: str = Field(description="Short two-line summary of the text")


class AnkiCard(BaseModel):
    """Structured output for ranked Anki cards including deduplication info."""

    rank: int = Field(description="Ranking position based on course order")
    title: str = Field(description="Anki card title")
    card_number: str = Field(description="Anki card number, e.g., 1629065002858")
    topic: str = Field(description="Broad topic that this card falls into")
    is_duplicate: bool = Field(
        description="True if this card is a duplicate. Mark the original card as duplicate as well"
    )
    duplicate_group: int = Field(
        default=None, description="Unique id for the group of duplicates"
    )
    duplicate_of: str = Field(
        default=None, description="Title of the card it duplicates, if applicable"
    )


class RankedAnkiList(BaseModel):
    """A list of ranked Anki cards."""

    ranked_list: list[AnkiCard]


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def generate_card_summary(card_text: str, llm_model: ChatOpenAI) -> AnkiContent:
    """
    Generate a structured summary for an individual Anki card.

    Parameters:
        card_text (str): Concatenated text from the card fields.
        llm_model (ChatOpenAI): The LLM instance with structured output.

    Returns:
        AnkiContent: The structured summary including bullet-point summary and short summary.
    """
    system_prompt = (
        "Write a concise bullet point summary of the given text in markdown. "
        "Section headers also start with a bullet. "
        "Put the bullet point summary into the field 'body'. "
        "Then write a two-line summary and put it into the field 'short_summary'.\n\n"
        "Follow this format:\n\n"
        "- **First level item 1**\n"
        "  - **Second level item 1.1**: ...\n"
        "  - **Second level item 1.2**: ...\n"
        "  ...\n"
    )

    prompt_template = ChatPromptTemplate(
        [
            ("system", system_prompt),
            ("user", "{card_text}"),
        ]
    )
    # Build and invoke the prompt with the provided card text.
    prompt_instance = prompt_template.invoke({"card_text": card_text})
    summary = llm_model.invoke(prompt_instance)
    return summary


def summarise_duplicates(duplicate_summaries: str, llm_model: ChatOpenAI) -> str:
    """
    Consolidate multiple duplicate summaries into a single summary.

    Parameters:
        duplicate_summaries (str): The concatenated summaries of duplicate cards.
        llm_model (ChatOpenAI): The LLM instance for merging summaries.

    Returns:
        str: A merged summary in markdown format.
    """
    duplicate_system_prompt = (
        "You are given the summaries of duplicated Anki cards in markdown. Bold words as necessary. "
        "Rewrite them into a single summary while preserving the original wording. "
        "Remove duplicated items/concepts and rearrange items only if necessary so that the flow is improved.\n\n"
        "Section headers also start with a bullet. Follow this format:\n\n"
        "- **First level item 1**\n"
        "  - **Second level item 1.1**: ...\n"
        "  - **Second level item 1.2**: ...\n"
        "  ...\n"
    )

    prompt_template = ChatPromptTemplate(
        [
            ("system", duplicate_system_prompt),
            ("user", "{duplicate_text}"),
        ]
    )
    prompt_instance = prompt_template.invoke({"duplicate_text": duplicate_summaries})
    merged_result = llm_model.invoke(prompt_instance)
    return merged_result.content


def get_card_order(anki_titles_text: str, llm_model: ChatOpenAI) -> str:
    # Build the prompt to rank Anki cards and mark duplicates
    ranking_prompt_template = ChatPromptTemplate.from_template(
        """Given the following Anki card titles from a course, rank them based on their natural material order.
            - Identify if any cards cover the same or highly similar material and mark them as duplicates.
            - Place duplicates next to each other in terms of ranking and mark them (both the duplicate and the original) as is_duplicate.
            - Assign a broad topic to each card.

            Anki Card Titles:
            {anki_titles}

            Output the results in JSON format following the provided structured schema.
"""
    )
    # Initialize the LLM for ranking using the structured output schema.
    structured_ranking_llm = llm_model.with_structured_output(RankedAnkiList)
    ranking_prompt_instance = ranking_prompt_template.invoke(
        {"anki_titles": anki_titles_text}
    )
    ranking_response = structured_ranking_llm.invoke(ranking_prompt_instance)

    # Convert the ranked list into a DataFrame
    ranking_data = [
        {
            "rank": card.rank,
            "title": card.title,
            "card_number": getattr(card, "card_number", None),
            "topic": getattr(card, "topic", None),
            "is_duplicate": card.is_duplicate,
            "duplicate_group": card.duplicate_group,
            "duplicate_of": card.duplicate_of,
        }
        for card in ranking_response.ranked_list
    ]
    ranking_df = pd.DataFrame(ranking_data)

    return ranking_df


def filter_extras(strings):
    pattern = r"\bextras?\b"  # Matches "extra" or "extras" as whole words
    return [s for s in strings if not re.search(pattern, s, re.IGNORECASE)]


# -----------------------------------------------------------------------------
# Main Processing Function
# -----------------------------------------------------------------------------


def summarise_anki_deck(anki_search_query, gpt_model="gpt-4o-mini"):
    # ---------------------------
    # Setup and Data Loading
    # ---------------------------
    # Set OpenAI API key (ensure you keep your key secure)

    # Load Anki deck data into a DataFrame (example deck)
    anki_df = load_anki_query_to_dataframe(anki_search_query)
    anki_columns = list(anki_df.columns)[::-1]
    anki_columns = filter_extras(anki_columns)  # remove "extras" fields

    # Initialize the LLM model for generating card summaries using GPT-4 with temperature 0.
    llm_model = ChatOpenAI(model=gpt_model, temperature=0)
    structured_summary_llm = llm_model.with_structured_output(AnkiContent)

    # ---------------------------
    # Generate Structured Summaries for Each Card
    # ---------------------------
    start_idx = 0
    is_finished = False

    while True:
        response = input("Do you want to retry? (yes/no): ").strip().lower()
        if response == "yes":
            print("Retrying...")
            anki_df = pd.read_pickle(f"temp_{anki_search_query}.pkl")
            anki_df_to_process = anki_df[anki_df.summary.isna()]
            if len(anki_df_to_process) == 0:
                is_finished = True
                print("nothing to process. All summaries generated ....")
                break
            start_idx = min(anki_df[anki_df.summary.isna()].index)
            break
        elif response == "no":
            print("Exiting...")
            break
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")

    if not is_finished:
        print(f"starting at index {start_idx}")
        for index in tqdm(range(start_idx, len(anki_df)), desc="Summarizing Cards"):
            # Concatenate all relevant fields from the card
            card_content = anki_df.loc[index][anki_columns].str.cat()

            # Generate the structured summary for the card
            card_summary: AnkiContent = generate_card_summary(
                card_content, structured_summary_llm
            )

            # Display the summary in the notebook (optional)
            print("---- Summary for Card {} ----".format(index))
            display(Markdown(card_summary.body))

            # Save the summary back to the DataFrame
            anki_df.loc[index, "summary"] = card_summary.body

            # Save progress intermittently
            anki_df.to_pickle(f"temp_{anki_search_query}.pkl")

    # ---------------------------
    # Rank Cards and Identify Duplicates
    # ---------------------------
    print("Ranking Cards Titles based on course order ......")

    # Create a single string combining card numbers and titles for the ranking prompt.
    anki_titles_text = (
        "card number: " + anki_df["card_number"] + "     " + anki_df["f"]
    ).str.cat()

    ranking_df = get_card_order(anki_titles_text=anki_titles_text, llm_model=llm_model)

    # Validate that all card numbers match between the ranking and original DataFrame.
    ranked_numbers = set(ranking_df.card_number)
    original_numbers = set(anki_df.card_number)
    print("unranked cards", original_numbers - ranked_numbers)

    # Merge ranking information with the original card data.
    merged_df = anki_df.merge(ranking_df, on="card_number", how="left")
    merged_df["rank"] = merged_df["rank"].fillna(10000)

    # Extract rows that have been identified as duplicates.
    duplicates_df = merged_df[~merged_df.duplicate_group.isna()]

    # ---------------------------
    # Deduplicate Duplicate Card Summaries
    # ---------------------------
    print("de-duplicating cards .......")
    deduped_summaries = []
    for duplicate_id, group in duplicates_df.groupby("duplicate_group"):
        # Use the smallest rank among duplicates as the representative rank.
        representative_rank = group["rank"].min()
        # Use the first card's title as the representative title.
        representative_title = group.title.iloc[0]
        # Concatenate all summaries from the duplicate group.
        combined_summaries = group["summary"].str.cat()
        # Merge duplicate summaries using the helper function.
        merged_summary = summarise_duplicates(combined_summaries, llm_model)
        deduped_summaries.append(
            {
                "rank": representative_rank,
                "summary": merged_summary,
                "f": representative_title,
                "title": representative_title,
            }
        )
    deduped_df = pd.DataFrame(deduped_summaries)

    # ---------------------------
    # Combine Non-Duplicate and Deduplicated Cards and Output
    # ---------------------------
    if len(deduped_df) == 0:
        print("no duplicates ...")
        final_df = merged_df
    else:
        final_df = pd.concat([merged_df[~merged_df.is_duplicate], deduped_df])
    final_df = final_df.sort_values("rank")

    final_df["title"] = final_df["title"].fillna("(Unranked by GPT) - " + final_df["f"])

    # Create a markdown formatted string combining title and summary for each card.
    final_df["title_and_summary"] = (
        "**" + final_df["title"] + "** \n" + final_df["summary"] + " \n"
    )

    final_markdown = "\n".join(final_df.title_and_summary.tolist())

    # Write the final markdown output to a file.
    output_path = f"output/{anki_search_query}.md"
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(final_markdown)

    # Optionally, display the final markdown in the notebook.
    # display(Markdown(final_markdown))


if __name__ == "__main__":
    df = load_anki_query_to_dataframe('"deck:Quant::Linear Algebra::Theory"')
    summarise_anki_deck(
        anki_search_query='"deck:Quant::ML::Essential::GenAI::Langchain::Langchain Acedemy (Official)"',
        gpt_model="gpt-4o",
    )
