
from textwrap import dedent
from agno.agent import Agent
from agno.tools.serpapi import SerpApiTools
import streamlit as st
from agno.models.openai import OpenAIChat

st.title("🌟 VoyageGenie: AI Travel Planner ✈️")
st.caption("Effortlessly plan your perfect trip with a personalized itinerary powered by GPT-4o and real-time search.")

# Get OpenAI API key from user
openai_api_key = st.text_input("🔑 OpenAI API Key", type="password", help="Required to use GPT-4o for itinerary generation")
serp_api_key = st.text_input("🔎 SerpAPI Key", type="password", help="Used to fetch real-time travel info from the web")

if openai_api_key and serp_api_key:
    researcher = Agent(
        name="Travel Researcher",
        role="Collects up-to-date travel information tailored to user preferences.",
        model=OpenAIChat(id="gpt-4o", api_key=openai_api_key),
        description=dedent(
            """\
        You are a world-class travel researcher. Given a travel destination and the number of days the user wants to travel for,
        generate a list of search terms for finding relevant travel activities and accommodations.
        Then search the web for each term, analyze the results, and return the 10 most relevant results.
        """
        ),
        instructions=[
            "Given a travel destination and the number of days the user wants to travel for, first generate a list of 3 search terms related to that destination and the number of days.",
            "For each search term, `search_google` and analyze the results.",
            "From the results of all searches, return the 10 most relevant results to the user's preferences.",
            "Remember: the quality of the results is important.",
        ],
        tools=[SerpApiTools(api_key=serp_api_key)],
        add_datetime_to_instructions=True,
    )
    planner = Agent(
        name="Itinerary Planner",
        role="Creates a customized travel itinerary using research insights.",
        model=OpenAIChat(id="gpt-4o", api_key=openai_api_key),
        description=dedent(
            """\
         You are a professional travel planner. Based on a user's destination, trip duration, and research summary, draft a compelling, detailed travel itinerary.
        """
        ),
        instructions=[
            "Given a travel destination, the number of days the user wants to travel for, and a list of research results, generate a draft itinerary that includes suggested activities and accommodations.",
            "Ensure the itinerary is well-structured, informative, and engaging.",
            "Ensure you provide a nuanced and balanced itinerary, quoting facts where possible.",
            "Remember: the quality of the itinerary is important.",
            "Focus on clarity, coherence, and overall quality.",
            "Never make up facts or plagiarize. Always provide proper attribution.",
        ],
        add_datetime_to_instructions=True,
    )

    # Input fields for the user's destination and the number of days they want to travel for
    destination = st.text_input("🌍 Destination", placeholder="e.g., Tokyo, Paris, Bali")
    num_days = st.number_input("📅 Duration (in days)", min_value=1, max_value=30, value=7)

    if st.button("🧳 Generate My Itinerary"):
        if destination.strip() == "":
            st.warning("⚠️ Please enter a valid destination.")
        else:
            with st.spinner("🔍 Researching your destination..."):
                research_prompt = f"{destination} itinerary ideas for a {num_days}-day trip."
                research_results = researcher.run(research_prompt, stream=False)
                st.success("✅ Travel research completed!")

            with st.spinner("📝 Crafting your personalized itinerary..."):
                planner_prompt = dedent(f"""
                    Create a personalized itinerary for:

                    - Destination: {destination}
                    - Duration: {num_days} days
                    - Based on the following research:
                    {research_results.content}

                    Format it day-by-day with places to visit, food options, tips, and stays.
                """)
                response = planner.run(planner_prompt, stream=False)
                st.success("🎉 Your itinerary is ready!")
                st.markdown(response.content)
