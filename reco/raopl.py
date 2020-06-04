# Replace alphabets.alphabet with one word per line
import alphabets

alp = alphabets.alphabet
total = len(alp)
print(type(alp))
with open('cn.txt', mode='w', encoding='utf-8') as f:
    i = 1
    for char in alp:
        f.write(char+'\n')
        haiyou = total - i
        print("还有 %d" % haiyou)
        i += 1




