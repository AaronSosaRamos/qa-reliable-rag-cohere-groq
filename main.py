import gradio as gr
from graph.graph import app
from utils.logger import setup_logger

# Logger configuration
logger = setup_logger(__name__)

# Function to process inputs and format results
def qa_agent_process(urls, question):
    """
    Process URLs and a question with the QA Agent, returning selected attributes.

    Args:
        urls (str): Comma-separated URLs.
        question (str): User's question.

    Returns:
        dict: Processed attributes including generation, is_grounded, and lookup_response.
    """
    try:
        # Prepare inputs
        url_list = [url.strip() for url in urls.split(",")]
        inputs = {"urls": url_list, "question": question}

        # Invoke the QA agent
        raw_result = app.invoke(inputs)
        logger.info(f"Inputs processed: {inputs}")
        logger.info(f"Raw response: {raw_result}")

        # Extract and format results
        formatted_result = {
            "generation": raw_result.get("generation", "N/A"),
            "is_grounded": raw_result.get("is_grounded", "N/A"),
            "lookup_response": raw_result.get("lookup_response", {})
        }

        return formatted_result["generation"], formatted_result["is_grounded"], formatted_result["lookup_response"]
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        return "Error: Could not process your request.", "N/A", {}

# Define the Gradio interface
with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown(
        """
        # 🌟 **QA Agent with Reliable RAG, GROQ, and Cohere** 🌟
        ---
        🤖 **Your Personal AI Assistant for Reliable Answers**  
        Query multiple URLs and get a concise, AI-powered response grounded in reliable sources.  
        Powered by **RAG**, **GROQ**, and **Cohere**.
        """
    )

    # Input and output sections with better visuals
    with gr.Row(elem_id="centered-layout", equal_height=True):
        with gr.Column(scale=1):
            gr.Markdown("### 🔍 **Input Section**")
            urls_input = gr.Textbox(
                label="🌐 URLs (comma-separated)",
                placeholder="Enter URLs separated by commas...",
                lines=3,
                elem_id="urls-input"
            )
            question_input = gr.Textbox(
                label="❓ Question",
                placeholder="What would you like to know?",
                lines=2,
                elem_id="question-input"
            )
            submit_button = gr.Button(value="🚀 Process", elem_id="submit-btn")

        with gr.Column(scale=1):
            gr.Markdown("### 📊 **Response Section**")
            output_generation = gr.Textbox(
                label="💡 Generated Answer",
                placeholder="The generated response will appear here...",
                lines=4,
                elem_id="output-generation"
            )
            output_is_grounded = gr.Textbox(
                label="✅ Grounded in Documents",
                placeholder="Displays whether the response is grounded in the documents.",
                lines=1,
                elem_id="output-is-grounded"
            )
            output_lookup = gr.JSON(
                label="📄 Document Highlights",
                visible=True,
                elem_id="output-lookup"
            )

    # Button functionality
    submit_button.click(
        fn=qa_agent_process,
        inputs=[urls_input, question_input],
        outputs=[output_generation, output_is_grounded, output_lookup]
    )

# Launch the application
if __name__ == "__main__":
    demo.launch()