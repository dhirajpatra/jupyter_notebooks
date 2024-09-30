from transformers import GPT2Tokenizer, GPT2Model


def generate_assignment(text):
    """Generates an assignment from the given text."""

    # Load the GPT-2 model.
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')

    # Encode the text.
    # encoded_input = tokenizer(text, return_tensors='pt')
    encoded_input = tokenizer(text, return_tensors='tf')

    # Generate the assignment.
    # output = model(**encoded_input)
    output = model(encoded_input)

    # Convert the output to a list of strings.
    assignments = []
    for assignment in output:
        assignments.append(tokenizer.decode(assignment, skip_special_tokens=True))

    # Return the assignments.
    return assignments


# Generate an assignment.
assignments = generate_assignment("Replace me by any text you'd like.")

# Print the assignments.
for assignment in assignments:
    print(assignment)
