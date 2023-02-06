modelName = "../Train/model.ape"

def generate_model(prompt):
    names = get_names(prompt)
    print(names)

def get_names(prompt): # Each name equals to an object that needs to be generated
    names = []
    for word, pos in prompt:
        if not pos[0] == "V": # We skip verbs since they are not neccesary in the creation of 
            if pos[0] == "N":
                names.append(word)

    return names