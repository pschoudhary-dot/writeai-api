import json
import os
import time
import random
import datetime
from typing import Optional, Iterator
from phi.model.google import Gemini
from pydantic import BaseModel, Field
from phi.tools.file import FileTools
from phi.agent import Agent
from phi.workflow import Workflow, RunResponse, RunEvent
from phi.storage.workflow.sqlite import SqlWorkflowStorage
from phi.tools.duckduckgo import DuckDuckGo
from phi.utils.pprint import pprint_run_response
from phi.utils.log import logger
from phi.storage.agent.sqlite import SqlAgentStorage  # Ensure this import exists
from dotenv import load_dotenv

# Load environment variables at the start of the file
load_dotenv()

# Get API keys
google_api_key = os.getenv('GOOGLE_API_KEY')
gemini_api_key = os.getenv('GEMINI_API_KEY')
palm_api_key = os.getenv('PALM_API_KEY')

if not any([google_api_key, gemini_api_key, palm_api_key]):
    raise ValueError("No API keys found in environment variables!")

# Log which keys are available (first 8 chars only)
logger.info(f"Google API Key loaded: {google_api_key[:8]}..." if google_api_key else "No Google API Key")
logger.info(f"Gemini API Key loaded: {gemini_api_key[:8]}..." if gemini_api_key else "No Gemini API Key")
logger.info(f"Palm API Key loaded: {palm_api_key[:8]}..." if palm_api_key else "No Palm API Key")

# Define the Pydantic models
class NewsArticle(BaseModel):
    title: str = Field(..., description="Title of the article.")
    url: str = Field(..., description="Link to the article.")
    summary: Optional[str] = Field(..., description="Summary of the article if available.")


class SearchResults(BaseModel):
    articles: list[NewsArticle]


def create_gemini_model():
    return Gemini(
        id="gemini-1.5-pro-latest",
        api_key=gemini_api_key or google_api_key,  # Try Gemini key first, fall back to Google key
        max_tokens=2048,
        temperature=0.7,
        top_p=0.8,
        retry_on_error=True,
        rate_limit=3,
    )


class BlogPostGenerator(Workflow):
    # Define Agents

    # 1. Searcher Agent: Searches for relevant articles
    searcher: Agent = Agent(
        tools=[DuckDuckGo()],
        instructions=[
            """Search for articles on the given topic and return EXACTLY in this JSON format:
            {
                "articles": [
                    {
                        "title": "First Article Title",
                        "url": "https://example.com/article1",
                        "summary": "Brief summary of the first article"
                    }
                ]
            }
            Important:
            1. Return ONLY the JSON - no additional text, no markdown
            2. Include EXACTLY 5 most relevant articles
            3. Ensure each article has title, url, and summary
            4. Do not include any explanation or markdown formatting
            """
        ],
        model=create_gemini_model()
    )

    # 2. Content Curator Agent: Curates content based on search results
    content_creator_agent: Agent = Agent(
        name="content_creator_agent",
        role="Content Curator",
        description="Curates content based on search results to prepare for writing.",
        instructions="""
        Given a list of articles, curate and summarize the key points to assist in content creation.
        - Extract the main ideas from each article.
        - Identify common themes and unique insights.
        - Provide a concise summary that highlights the most important information.
        """,
        model=create_gemini_model(),
        show_tool_calls=True,
        add_datetime_to_instructions=True,
    )

    # 3. Writer Agent: Drafts the blog post
    writer: Agent = Agent(
        instructions=[
            "You will be provided with a topic and a curated summary of key points.",
            "Generate a New York Times-worthy blog post on that topic.",
            "Break the blog post into sections with appropriate headers.",
            "Provide key takeaways at the end.",
            "Ensure the title is catchy and engaging.",
            "Always provide sources; do not fabricate information or sources."
        ],
        model=create_gemini_model()
    )

    # 4. SEO Analyst Agent: Optimizes content for SEO
    seo_analyst_agent: Agent = Agent(
        tools=[],  # Assuming SEO analysis doesn't require search tools
        role="SEO Analyst",
        description="Optimizes content for search engines and improves content discoverability online.",
        instructions="""
        Optimize the provided blog post content for SEO.
        - Analyze keyword density and suggest additional keywords if necessary.
        - Refine meta descriptions and title tags to enhance SEO.
        - Ensure the content adheres to SEO best practices for better ranking on search engines.
        """,
        model=create_gemini_model(),
        storage=SqlAgentStorage(table_name="seo_analyst_agent_session", db_file="tmp/agents.db"),
        show_tool_calls=True,
        add_datetime_to_instructions=True,
    )

    # 5. Editorial Assistant Agent: Reviews and finalizes content
    editorial_assistant_agent: Agent = Agent(
        role="Editorial Assistant",
        description="Implements a review process to check the content for accuracy, coherence, grammar, and style. Makes necessary adjustments.",
        instructions="""
        Review the provided blog post content for accuracy, coherence, grammar, and style.
        - Correct any grammatical or spelling errors.
        - Ensure the content flows logically and is easy to read.
        - Adjust the style to match the publication's guidelines.
        """,
        model=create_gemini_model(),
        add_datetime_to_instructions=True,
    )

    def run(self, topic: str, use_cache: bool = False) -> Iterator[RunResponse]:
        logger.info(f"Generating a blog post on: {topic}")

        # Disable caching by setting use_cache=False
        use_cache = False

        # Ensure the generated_posts directory exists
        os.makedirs("./generated_posts", exist_ok=True)

        # Step 1: Search for relevant articles
        logger.info("Step 1: Searching for relevant articles")
        num_tries = 0
        search_results: Optional[SearchResults] = None
        while search_results is None and num_tries < 3:
            try:
                num_tries += 1
                # Add longer initial delay
                wait_time = (2 ** (num_tries - 1)) * 5 + random.uniform(1, 3)
                if num_tries > 1:
                    logger.info(f"Waiting {wait_time:.2f} seconds before retry {num_tries}/3")
                    time.sleep(wait_time)
                    
                searcher_response: RunResponse = self.searcher.run(topic)
                if searcher_response and searcher_response.content:
                    try:
                        # Clean up the response content
                        content = searcher_response.content.strip()
                        
                        # Remove any markdown formatting if present
                        if content.startswith('```') and content.endswith('```'):
                            content = content.split('\n', 1)[1].rsplit('\n', 1)[0]
                        
                        # Log the content for debugging
                        logger.debug(f"Searcher response content: {content}")
                        
                        # Parse the JSON
                        response_dict = json.loads(content)
                        
                        # Validate the structure
                        if 'articles' not in response_dict:
                            logger.warning("Response missing 'articles' key")
                            raise ValueError("Invalid response structure")
                        
                        # Create SearchResults object
                        search_results = SearchResults(**response_dict)
                        logger.info(f"Successfully found {len(search_results.articles)} articles")
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON parsing error: {e}")
                        search_results = None
                    except Exception as e:
                        logger.warning(f"Error processing search results: {e}")
                        search_results = None
                else:
                    logger.warning("Empty or invalid response from searcher")
                    
                if search_results is None:
                    # Implement exponential backoff with jitter
                    wait_time = (2 ** num_tries) + random.uniform(0, 1)
                    logger.info(f"Retrying search ({num_tries}/3) after {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                    
            except Exception as e:
                logger.error(f"Search attempt failed: {e}")
                search_results = None

        # If no search_results are found for the topic, end the workflow
        if search_results is None or len(search_results.articles) == 0:
            yield RunResponse(
                run_id=self.run_id,
                event=RunEvent.workflow_completed,
                content=f"Sorry, could not find any articles on the topic: {topic}",
            )
            return

        # Step 2: Content Curation
        logger.info("Step 2: Curating content")
        try:
            curated_input = {"articles": [article.model_dump() for article in search_results.articles]}
            curated_response: RunResponse = self.content_creator_agent.run(json.dumps(curated_input, indent=4))
            if curated_response and curated_response.content:
                curated_content = curated_response.content
                logger.info("Content curation completed.")
            else:
                logger.warning("Content curator agent response invalid.")
                curated_content = {"articles": [article.model_dump() for article in search_results.articles]}  # Fallback
        except Exception as e:
            logger.error(f"Error running content_creator_agent: {e}")
            yield RunResponse(
                run_id=self.run_id,
                event=RunEvent.workflow_completed,
                content=f"Failed during content curation phase: {e}",
            )
            return

        # Step 3: Write Blog Post
        logger.info("Step 3: Writing blog post")
        try:
            writer_input = {
                "topic": topic,
                "curated_content": curated_content
            }
            writer_response = self.run_with_retries(self.writer, writer_input)
            if writer_response and writer_response.content:
                blog_post_content = writer_response.content
                yield RunResponse(
                    run_id=self.run_id,
                    event=RunEvent.running,
                    content=blog_post_content
                )
            else:
                raise Exception("Writer returned empty response")
        except Exception as e:
            logger.error(f"Error running writer: {e}")
            yield RunResponse(
                run_id=self.run_id,
                event=RunEvent.workflow_completed,
                content=f"Failed during blog writing phase: {e}",
            )
            return

        # Step 4: SEO Analysis
        logger.info("Step 4: Optimizing blog post for SEO")
        try:
            seo_input = {"content": blog_post_content}
            seo_response: RunResponse = self.seo_analyst_agent.run(json.dumps(seo_input, indent=4))
            if seo_response and seo_response.content:
                optimized_content = seo_response.content
                logger.info("SEO optimization completed.")
            else:
                logger.warning("SEO analyst agent response invalid.")
                optimized_content = blog_post_content  # Fallback to original content
        except Exception as e:
            logger.error(f"Error running seo_analyst_agent: {e}")
            optimized_content = blog_post_content  # Fallback to original content

        # Step 5: Editorial Review
        logger.info("Step 5: Performing editorial review")
        try:
            editorial_input = {"content": optimized_content}
            editorial_response: RunResponse = self.editorial_assistant_agent.run(json.dumps(editorial_input, indent=4))
            if editorial_response and editorial_response.content:
                final_blog_post = editorial_response.content
                logger.info("Editorial review completed.")
            else:
                logger.warning("Editorial assistant agent response invalid.")
                final_blog_post = optimized_content  # Fallback to optimized content
        except Exception as e:
            logger.error(f"Error running editorial_assistant_agent: {e}")
            final_blog_post = optimized_content  # Fallback to optimized content

        # Step 6: Save to Markdown File
        logger.info("Step 6: Saving blog post to Markdown file")
        try:
            # Extract title for filename from the blog post
            title_line = final_blog_post.split('\n')[0]  # Assuming the first line is the title
            title = title_line.replace('#', '').strip() if title_line.startswith('#') else "Untitled_Blog_Post"
            # Sanitize filename and append timestamp to prevent overwrites
            timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            filename = f"{title.lower().replace(' ', '-').replace('.', '').replace('/', '-')}-{timestamp}.md"

            # Generate YAML front matter
            current_date = datetime.datetime.now().strftime('%Y-%m-%d')
            yaml_front_matter = f"""---
title: "{title}"
date: "{current_date}"
author: "AI Generated"
---

"""

            final_blog_post_with_yaml = yaml_front_matter + final_blog_post

            # Save the content to a Markdown file
            file_path = os.path.join("./generated_posts", filename)
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(final_blog_post_with_yaml)

            absolute_path = os.path.abspath(file_path)
            logger.info(f"Blog post successfully saved to {absolute_path}")
        except Exception as e:
            logger.error(f"Error saving blog post to file: {e}")
            # Continue without failing the workflow

        # Step 7: Yield Final Blog Post
        logger.info("Step 7: Workflow completed successfully")
        yield RunResponse(
            run_id=self.run_id,
            event=RunEvent.workflow_completed,
            content=final_blog_post,
        )

    # Add retry logic with exponential backoff for the writer
    def run_with_retries(self, agent: Agent, input_data: dict, max_retries: int = 3) -> RunResponse:
        for attempt in range(max_retries):
            try:
                return agent.run(json.dumps(input_data, indent=4))
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.info(f"Rate limited. Waiting {wait_time:.2f} seconds before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                else:
                    raise e


# The topic to generate a blog post on
topic = "A guide to agentic AI"

# Create the workflow without caching
generate_blog_post = BlogPostGenerator(
    session_id=f"generate-blog-post-on-{topic}",
    storage=SqlWorkflowStorage(
        table_name="generate_blog_post_workflows",
        db_file="tmp/workflows.db",
    ),
)

# Run workflow
blog_post_iterator: Iterator[RunResponse] = generate_blog_post.run(topic=topic, use_cache=False)

# Print the response
pprint_run_response(blog_post_iterator, markdown=True)
