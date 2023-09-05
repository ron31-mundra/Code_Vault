import logging
import os
import subprocess
import sys
import tempfile
import traceback
from io import StringIO

import langchain.agents as lc_agents
import openai
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import LLMChain, SequentialChain
from langchain.llms import OpenAI
from langchain.llms import OpenAI as LangChainOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from streamlit.components.v1 import html

load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")
global generated_code

LANGUAGE_CODES = {
    "C": "c",
    "C++": "cpp",
    "Java": "java",
    "Ruby": "ruby",
    "Scala": "scala",
    "C#": "csharp",
    "Objective C": "objc",
    "Swift": "swift",
    "JavaScript": "nodejs",
    "Kotlin": "kotlin",
    "Python": "python3",
    "GO Lang": "go",
}

st.title("Code-Verse: \nGenerate, Run, and Save Code Seamlessly")
code_prompt = st.text_input("Enter a prompt to generate the code")
code_language = st.selectbox("Select a language", list(LANGUAGE_CODES.keys()))

button_generate = st.button("Generate Code")
code_file = st.text_input("Enter file name:")
button_save = st.button("Save Code")


compiler_mode = st.radio("Compiler Mode", ("Online", "Offline"))
button_run = st.button("Run Code")

# Prompt Templates
code_template = PromptTemplate(
    input_variables=["code_topic"],
    template="Write me code in " + f"{code_language} language" + " for {code_topic}",
)

code_fix_template = PromptTemplate(
    input_variables=["code_topic"],
    template="Fix any error in the following code in "
    + f"{code_language} language"
    + " for {code_topic}",
)


memory = ConversationBufferMemory(input_key="code_topic", memory_key="chat_history")
open_ai_llm = OpenAI(temperature=0.7, max_tokens=1000)

code_chain = LLMChain(
    llm=open_ai_llm,
    prompt=code_template,
    output_key="code",
    memory=memory,
    verbose=True,
)

code_fix_chain = LLMChain(
    llm=open_ai_llm,
    prompt=code_fix_template,
    output_key="code_fix",
    memory=memory,
    verbose=True,
)

sequential_chain = SequentialChain(
    chains=[code_chain, code_fix_chain],
    input_variables=["code_topic"],
    output_variables=["code", "code_fix"],
)


def generate_dynamic_html(language, code_prompt):
    logger = logging.getLogger(__name__)
    logger.info("Generating dynamic HTML for language: %s", language)
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Python App with JavaScript</title>
    </head>
    <body>
        <div data-pym-src='https://www.jdoodle.com/plugin' data-language="{language}"
            data-version-index="0" data-libs="">
            {script_code}
        </div>
        <script src="https://www.jdoodle.com/assets/jdoodle-pym.min.js" type="text/javascript"></script>
    </body>
    </html>
    """.format(
        language=LANGUAGE_CODES[language], script_code=code_prompt
    )
    return html_template


# Setup logging method.
def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
        filename=log_file,  # Add this line to save logs to a file
        filemode="a",  # Append logs to the file
    )


# Setup logging
log_file = __file__.replace(".py", ".log")
setup_logging(log_file)

# REPL Stands for read eval print and loop

class PythonREPL:                       #Read your Python commands. Evaluate your code to work out what the input means. Print the results to see the computer’s response.
    def __init__(self):
        pass

    def run(self, command: str) -> str:
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        try:
            exec(command, globals())
            sys.stdout = old_stdout
            output = mystdout.getvalue()
        except Exception as e:
            sys.stdout = old_stdout
            output = str(e)
        return output


def run_query(query, model_kwargs, max_iterations):
    llm = LangChainOpenAI(**model_kwargs)
    python_repl = lc_agents.Tool(
        "Python REPL",
        PythonREPL().run,
        "A Python shell. Use this to execute python commands.",
    )
    tools = [python_repl]
    agent = lc_agents.initialize_agent(
        tools,
        llm,
        agent=lc_agents.AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        model_kwargs=model_kwargs,
        verbose=True,
        max_iterations=max_iterations,
    )
    response = agent.run(query)
    return response


def run_code(code, language):
    logger = logging.getLogger(__name__)
    logger.info(f"Running code: {code} in language: {language}")

    if language == "Python":
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=True) as f:
            f.write(code)
            f.flush()

            logger.info(f"Input file: {f.name}")
            output = subprocess.run(["python", f.name], capture_output=True, text=True)
            logger.info(f"Runner Output execution: {output.stdout + output.stderr}")
            return output.stdout + output.stderr

    elif language == "C" or language == "C++":
        ext = ".c" if language == "C" else ".cpp"
        with tempfile.NamedTemporaryFile(mode="w", suffix=ext, delete=True) as src_file:
            src_file.write(code)
            src_file.flush()

            logger.info(f"Input file: {src_file.name}")

            with tempfile.NamedTemporaryFile(
                mode="w", suffix="", delete=True
            ) as exec_file:
                compile_output = subprocess.run(
                    [
                        "gcc" if language == "C" else "g++",
                        "-o",
                        exec_file.name,
                        src_file.name,
                    ],
                    capture_output=True,
                    text=True,
                )

                if compile_output.returncode != 0:
                    return compile_output.stderr

                logger.info(f"Output file: {exec_file.name}")
                run_output = subprocess.run(
                    [exec_file.name], capture_output=True, text=True
                )
                logger.info(
                    f"Runner Output execution: {run_output.stdout + run_output.stderr}"
                )
                return run_output.stdout + run_output.stderr

    elif language == "JavaScript":
        with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=True) as f:
            f.write(code)
            f.flush()

            logger.info(f"Input file: {f.name}")
            output = subprocess.run(["node", f.name], capture_output=True, text=True)
            logger.info(f"Runner Output execution: {output.stdout + output.stderr}")
            return output.stdout + output.stderr

    else:
        return "Unsupported language."


def generate_code():
    logger = logging.getLogger(__name__)
    try:
        st.session_state.generated_code = code_chain.run(code_prompt)
        st.session_state.code_language = code_language
        st.code(
            st.session_state.generated_code,
            language=st.session_state.code_language.lower(),
        )

        with st.expander("Message History"):
            st.info(memory.buffer)
    except Exception as e:
        st.write(traceback.format_exc())
        logger.error(f"Error in code generation: {traceback.format_exc()}")

# global file_name
file_name=''

def save_code():
    logger = logging.getLogger(__name__)
    try:
        global file_name
        file_name = code_file
        logger.info(f"Saving code to file: {file_name}")
        if file_name:
            with open(file_name, "w") as f:
                f.write(st.session_state.generated_code)
            st.success(f"Code saved to file {file_name}")
            logger.info(f"Code saved to file {file_name}")
        st.code(
            st.session_state.generated_code,
            language=st.session_state.code_language.lower(),
        )

    except Exception as e:
        st.write(traceback.format_exc())
        logger.error(f"Error in code saving: {traceback.format_exc()}")

def execute_code(compiler_mode: str):               # "compiler_mode: str" is known as type annotaion (go to this link) for type annotations {https://medium.com/analytics-vidhya/type-annotations-in-python-3-8-3b401384403d#:~:text=Type%20Annotations%20in%20Python%203.8%201%20Hello%2C%20Typed,Mapping%20...%208%20typing%3A%20TypedDict%20...%20More%20items}

    logger = logging.getLogger(__name__)
    logger.info(
        f"Executing code: {st.session_state.generated_code} in language: {st.session_state.code_language} with Compiler Mode: {compiler_mode}"
    )

    try:
        if compiler_mode == "online":
            html_template = generate_dynamic_html(
                st.session_state.code_language, st.session_state.generated_code
            )
            html(html_template, width=720, height=800, scrolling=True)

        else:
            output = run_code(
                st.session_state.generated_code, st.session_state.code_language
            )
            logger.info(f"Output execution: {output}")

            if (
                "error" in output.lower()
                or "exception" in output.lower()
                or "SyntaxError" in output.lower()
                or "NameError" in output.lower()
            ):
                logger.error(f"Error in code execution: {output}")
                response = sequential_chain(
                    {"code_topic": st.session_state.generated_code}
                )
                fixed_code = response["code_fix"]
                st.code(fixed_code, language=st.session_state.code_language.lower())

                with st.expander("Message History"):
                    st.info(memory.buffer)
                logger.warning(f"Trying to run fixed code: {fixed_code}")
                output = run_code(fixed_code, st.session_state.code_language)
                logger.warning(f"Fixed code output: {output}")

            st.code(
                st.session_state.generated_code,
                language=st.session_state.code_language.lower(),
            )
            st.write("Execution Output:")
            st.write(output)
            logger.info(f"Execution Output: {output}")

    except Exception as e:
        st.write("Error in code execution:")
        st.write(traceback.format_exc())
        logger.error(f"Error in code execution: {traceback.format_exc()}")


if __name__ == "__main__":
    if "generated_code" not in st.session_state:
        st.session_state.generated_code = ""

    if "code_language" not in st.session_state:
        st.session_state.code_language = ""

    if button_generate and code_prompt:
        generate_code()

    if button_save and st.session_state.generated_code:
        save_code()
        
        with open(file_name,"rb") as code_file:
            codebyte = code_file.read()

        st.download_button(label='Download Code File',file_name=f'{file_name}', data=codebyte,mime="application/octet-stream")


    if button_run and code_prompt:
        code_state_option = "online" if compiler_mode == "Online" else "offline"
        execute_code(code_state_option)
