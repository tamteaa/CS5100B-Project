from src.agent.prompts import PromptLoader


if __name__ == '__main__':
    # Step 1: Initialize the PromptLoader
    prompt_loader = PromptLoader()

    # Step 2: List all available prompts
    available_prompts = prompt_loader.list_available_prompts()
    print("Available Prompts:", available_prompts)

    # Step 3: Load a specific prompt by filename
    if available_prompts:
        prompt_name = available_prompts[0]  # Example: use the first available prompt
        prompt_template = prompt_loader.load_prompt(prompt_name)

        # print the variables in the prompt
        print(prompt_template.get_variables())

        # set a variable in the prompt:
        prompt_template.set_variable("goal", "go to the corner")
        prompt_template.set_variable("name", "ChatGPT")

        # get the prompt as a string
        prompt_str = str(prompt_template)
        print(prompt_str)

        # can also just print it out like this
        print(prompt_template)

        # remove a section from the prompt
        prompt_template.remove_section("Agent Information")
        print(prompt_template)

