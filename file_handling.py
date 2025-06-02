import os

def load_global_context(context_source):
    context_texts=[]
    try:
        if isinstance(context_source,list):
            for file_path in context_source:
                if os.path.exists(file_path) and os.path.isfile(file_path):
                    with open(file_path, 'r', encoding='utf-8') as file:
                        context_texts.append(file.read())
                else:
                    print(f"File {file_path} does not exist or is not a file.")
        elif os.path.isdir(context_source):
            for filename in os.listdir(context_source):
                if filename.lower().endswith((".txt",".csv",".json")):
                    file_path = os.path.join(context_source, filename)
                    if os.path.isfile(file_path):
                        with open(file_path, 'r', encoding='utf-8') as file:
                            context_texts.append(file.read())
        elif os.path.exists(context_source) and os.path.isfile(context_source):
            with open(context_source, 'r', encoding='utf-8') as file:
                context_texts.append(file.read())
        else:
            print(f"Context source {context_source} is not a valid file or directory.")

        global_context = "\n\n".join(context_texts)
        print("global_context :",global_context)
        return global_context
    except Exception as e:
        print("global_context: ","") 
        print(f"Error reading global context : {e}") 
        return " "                  