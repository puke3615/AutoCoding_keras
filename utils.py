def char_str(chars, mapping=None):
    result = ''
    for c in chars:
        result += mapping(c) if mapping else c
    return result
