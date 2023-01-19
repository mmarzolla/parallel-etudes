from googletrans import Translator
import os
import time

'''
File name of all of the programs that have been translated
so far.

!! IMPORTANT !!
Always leave one empty line at the end of the file
'''
DONE_SO_FAR = 'data.txt'

# Directory from which the translator will search for files
DIR = "../"

'''
Opening and closing delimiters of portion of comments that
want to be remove from the translation, this is usefull to
keep the context of the markdown used as it is.

!! IMPORTANT !!
To add new expression that need to be removed from the text
remeber that the delimiters follow an hierarchy from the least
to the most specific
'''
to_remove = [
        ("---------------------------------------- ----- ----- ----- ----- ----- ----- ----- -----", "---------------------------------------- ----- ----- ----- ----- ----- ----- ----- -----"),
        ("```C", "```"),
        ("`", "`"),
        ("\n:", "\n"),
        ("□□", "■□"),
        ("![", ")"),
        ("$", "$"),
    ]

'''
Same as `to_remove` but these delimiters are used
recursively to translate within the retacted words
where is needed
'''
open_delimiters = ["[", "(", "!", "\n"]
close_delimiters = ["]", ")", "!"]

def find_sequence(text, start_sequence, end_sequence, cnt, delimiter):
    '''
    Given a text string finds every sequences of characters delimited
    by a `start_sequence` and an `end_sequence` sequence of characters.
    Once a match is found the text will be altered to an expression that
    the Google API translator wont modify. Using this method the context
    of the `text` variable wont change by much (usually the context of
    the subsequence that needs to be un-altered is the subject of the
    phrase, or UTF-8 characters used for a better understanding of the
    given exercise). Each match will be stored "as is" and be replaced
    by:
        `delimiter` `cnt` `delimiter`
        
        Es.
            Ω0Ω
    Once the `text` is fully converted to the selected language, it will
    require a conversion of all of these placeholders to their respective
    subsequence of characters.
    
    
    Parameters:
    ----------
    text : str
        Text that needs to be converted to another language that
        contains some sequences of characters that want to be kept
        "as is".
    start_sequence : str
        Subsequence of characters that delimiter the BEGINNING of the
        set of characters that want to be preserved during the traslation.
    end_sequence : str
        Subsequence of characters that delimiter the END of the set of
        characters that want to be preserved during the traslation.
    cnt : int
        Current count of how many subsequences of character have been
        preserved.
    delimiter : char
        New delimiter character that wont change during translation using
        Google API.
    
    
    Returns:
    ----------
    formatted_txt : str
        Returns the text translated and redacted.
    redacted_items : list(str)
        Returns the list of items that have been preserved.
    cnt : int
        Returns the new running count.
    
    '''
    formatted_txt = ""
    redacted_txt = ""
    redacted_items = []
    
    started = 0
    wait_for_next = 0
    start_sequence_cnt = 0
    end_sequence_cnt = 0
    
    for n in range(len(text)):
        if start_sequence == end_sequence and len(start_sequence) == 1:
            if text[n] == start_sequence:
                if started == 0:
                    formatted_txt += delimiter
                    started = 1
                else:
                    formatted_txt += str(cnt)
                    formatted_txt += delimiter
                    cnt += 1
                    wait_for_next = 2
                    started = 0
        else:
            if text[n] != start_sequence[start_sequence_cnt]:
                start_sequence_cnt = 0
            if text[n] != end_sequence[end_sequence_cnt]:
                end_sequence_cnt = 0
    
            if text[n] == start_sequence[start_sequence_cnt] and started == 0:
                start_sequence_cnt += 1
                if start_sequence_cnt == len(start_sequence):
                    formatted_txt = formatted_txt[:-(len(start_sequence)-1)]
                    formatted_txt += delimiter
                    redacted_txt += start_sequence[:-1]
                    start_sequence_cnt = 0
                    started = 1
            if text[n] == end_sequence[end_sequence_cnt] and started == 1:
                end_sequence_cnt += 1
                if end_sequence_cnt == len(end_sequence):
                    formatted_txt += str(cnt)
                    formatted_txt += delimiter
                    end_sequence_cnt = 0
                    cnt += 1
                    wait_for_next = 2
                    started = 0

        if wait_for_next != 0:
            wait_for_next -= 1
            
        if started == 0 and wait_for_next == 0:
            formatted_txt += text[n]
            if redacted_txt != "":
                redacted_items.append(redacted_txt)
                redacted_txt = ""
        else:
            #print(text[n], end="")
            redacted_txt += text[n]
            #print(redacted_txt)
    
    if len(redacted_txt) > 0:
        redacted_items.append(redacted_txt)
        redacted_txt = ""
    
    #print(formatted_txt, redacted_items)
    return formatted_txt, redacted_items, cnt

def index_in_list(a_list, index):
    '''
    Checks and returns if the a given index is within the
    list bounds.
    '''
    return index < len(a_list)


def string_riassembly(txt, redacted_items, delimiter):
    '''
    Converts the redacted items in the text to their initial state
    by removing from the text the `delimiter` character and replacing
    it for the item found in the `redacted_items` list at a given index.
    
    Parameters:
    ----------
    txt : str
        Text that needs to be modified with the redacted items restored
        to their initial state.
    redacted_items : list(str)
        List of all of the items that didn't receave any change.
    delimiter : char
        Delimiter character that needs to be removed from the text.
    
    
    Returns:
    ----------
    txt : str
        Returns the desired text.
    
    '''
    txt = txt.split(delimiter)
    #print(traslate_txt)

    for n in range(len(txt)):
        #print(n)
        #print(traslate_txt[n])
        if txt[n].isdigit() and index_in_list(redacted_items, int(txt[n])):
            txt[n] = redacted_items[int(txt[n])]
            #print(traslate_txt[n], "->", redacted_items[int(traslate_txt[n])])

    txt = "".join(txt)
    txt = txt.replace("] (", "](")

    return txt

def translate_redacted(txt, open_delimiters, close_delimiters, translator):
    '''
    Nested loop that will translate, for each redacted items, portion of their
    text without changing variables names or altering the markdown. This ensures
    that if the retacted items containt some sort of comments or plain text within
    them, those portions will be translated.
    
    Parameters:
    ----------
    txt : str
        Text of the redacted item.
    open_delimiters : str
        Subsequence of characters that delimiter the BEGINNING of the
        set of characters that want to be preserved during the traslation.
    close_delimiters : str
        Subsequence of characters that delimiter the END of the set of
        characters that want to be preserved during the traslation.
    translator : object
        Translator object used to translate the text to the selected
        language.
    
    
    Returns:
    ----------
    out : str
        Returns the correct translation of the redacted item.
    
    '''
    import re
    
    out = ""
    to_translate = ""
    last_delimiter = ""
    
    start = 0
    count = 0
    
    delimiter = "Ω"
    to_remove = "$"
    
    txt, tmp_items, _ = find_sequence(txt, to_remove, to_remove, 0, delimiter)
    
    for n in range(len(txt)):
        if not txt[n] in open_delimiters and not txt[n] in close_delimiters:
            #print(txt[n], end="")
            to_translate += txt[n]

        if txt[n] in open_delimiters:
            if len(to_translate) > 0:
                if re.search('[a-zA-Z]', to_translate):
                    to_translate = translater.translate(to_translate, src="it", dest="en").text
                    if last_delimiter == "\n":
                        to_translate += " "
                out += to_translate
                to_translate = ""
                last_delimiter = txt[n]
            start = 1
            out += txt[n]
        
        if txt[n] in close_delimiters:
            if len(to_translate) > 0 and re.search('[a-zA-Z]', to_translate):
                to_translate = translater.translate(to_translate, src="it", dest="en").text
            out += to_translate
            to_translate = ""
            last_delimiter = txt[n]
            out += txt[n] if txt[n] in close_delimiters and not txt[n] in open_delimiters else ""
            start = 0

    if len(to_translate) > 0:
        to_translate = translater.translate(to_translate, src="it", dest="en").text
        out += to_translate
    
    out = string_riassembly(out, tmp_items, delimiter)
    out = out.replace("\n: ", "\n:")
    
    return out

list_of_files = []
files_processed = []

if not os.path.exists(DONE_SO_FAR):
    f = open(DONE_SO_FAR, "w")
    f.close()

f = open(DONE_SO_FAR, "r")
for line in f.readlines():
    files_processed.append(line[:-1])
f.close()
#print(files_processed)

for file in os.listdir(os.getcwd() + "\\" + DIR):
    if file.endswith('.cu') and not "test_" in file and not file in files_processed:
        list_of_files.append(file)

translater = Translator()

for file in list_of_files:
    before_txt= ""
    translate_txt = ""
    after_txt= ""
    before = 1
    translate = 0
    after = 0
    f = open(DIR + file, encoding='utf-8')
    for line in f.readlines():
        if "/***\n" == line or " /***\n" == line:
            translate = 1
            before = 0
        if "***/\n" == line or " ***/\n" == line:
            after = 1
            translate = 0

        if before:
            before_txt += line
        if translate:
            translate_txt += line
        if after:
            after_txt += line
        
        
    f.close()
    #print(source_txt.encode('utf8'))
    
    count = 0
    redacted_items = []
    tmp_items = []

    for item in to_remove:
        translate_txt, tmp_items, count = find_sequence(translate_txt, item[0], item[1], count, "Ω")
        redacted_items += tmp_items

    #print(redacted_items)
    #print()
    #print(translate_txt)

    translater = Translator()
    if len(translate_txt) > 0:
        translate_txt = translater.translate(translate_txt, src="it", dest="en").text
        translate_txt = translate_txt.split("Ω")
        #print(translate_txt)
        
        for n in range(len(redacted_items)):
            if not "```C" in redacted_items[n] or not "`" in redacted_items[n]:
                redacted_items[n] = translate_redacted(redacted_items[n], open_delimiters, close_delimiters, translater)

        for n in range(len(translate_txt)):
            #print(n)
            #print(translate_txt[n])
            if translate_txt[n].isdigit() and index_in_list(redacted_items, int(translate_txt[n])):
                translate_txt[n] = redacted_items[int(translate_txt[n])]
                #print(translate_txt[n], "->", redacted_items[int(translate_txt[n])])

        translate_txt = "".join(translate_txt)
        translate_txt = translate_txt.replace("] (", "](")
        translate_txt = translate_txt.replace("□ ■", "□■")

    translate_txt += '\n'

    data = before_txt + translate_txt + after_txt
    
    f = open("test_" + file, "w", encoding='utf-8')
    f.write(data)
    f.close
    f = open(DONE_SO_FAR, "a")
    print(file)
    f.write(file + '\n')
    f.close

#print(time.time() - time_start)
print("DONE")