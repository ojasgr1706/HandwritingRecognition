def encode(string,dic):
    """
    Encodes a string using a dictionary
    """
    encoded = []
    for char in string:
        encoded.append(dic[char])
    
    encoded += [0]*(32 - len(encoded))
    assert len(encoded) == 32

    return encoded

def decode(encoded,dic):
    """
    Decodes a string using a dictionary
    """
    decoded = ""
    for char in encoded:
        for key, value in dic.items():
            if value == char:
                decoded += key
    return decoded