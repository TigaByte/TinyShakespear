


with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    print("Lenth of the Text :", len(text))

chars = sorted(list(set(text)));
vocab_size = len(chars);
print("".join(chars));
print("Vocabulary Size:", vocab_size)