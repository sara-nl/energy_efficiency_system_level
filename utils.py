
import re

# Define a function to expand and format node names
def format_node_names(node_name):
    """This function cleans the node names that we have
    obtained from slurm data base. 
    for example [tcn97, tcn99-tcn101] ==> tcn97, tcn99, tcn100, tcn101

    Args:
        node_name (str): _it is a string that we want to process_

    Returns:
        _string_: 
    """
    # Check if the node name contains brackets (indicating range or list format)
    if '[' in node_name:
        # Extract prefix and the numbers part
        prefix, nums = re.match(r"(\w+)\[(.+)\]", node_name).groups()
        
        # Split on comma and process each item
        formatted_nodes = []
        for part in nums.split(','):
            part = part.strip()
            if '-' in part:  # It's a range
                start, end = map(int, part.split('-'))
                formatted_nodes.extend([f"{prefix}{i}" for i in range(start, end + 1)])
            else:  # It's a single number
                formatted_nodes.append(f"{prefix}{part}")
        
        return f"{','.join(formatted_nodes)}"
    else:
        # No brackets, treat as a single node name
        return f"{node_name}"
    
    
    
    
def get_list(file_path):
    """Reads a file and returns its contents as a list of lines.

    Args:
        file_path (str): The path to the file to read.

    Returns:
        list of str: A list where each element is a line from the file.
    """
    names = []
    with open(file_path, "r") as file:
        for line in file:
            # Strip leading/trailing whitespace and check if line is a comment
            line = line.strip()
            if not line.startswith("#") and line:  # If the line is not a comment and is not empty
                names.append(line.split('.')[0])

    return names