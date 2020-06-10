import nltk
from nltk.corpus import stopwords, wordnet
from nltk import stem
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict, Counter
import re
import tkinter as tk
import os

class TopWords:
    """A display of common words in a document.
    
    Parameters
    ----------
    sentences: list
        A sentence-tokenization of the given document.
        
    word_dict: dict
        Keeps track of which sentences each word appears in.
        
    doc_dict: dict
        Keeps track of which documents each word appears in.
        
    pos_dict: dict
        Keeps track of each word's part of speech.
        
    word_freqs: dict
        A frequency dictionary of the words in the document.

    most_common: list
        All words in order of frequency.
     
    n_frames: list
        A list of 'ns' objects.
        
    word_frames: list
        A list of 'words' objects.
        
    doc_frames: list
        A list of 'docs' objects.
        
    sent_frames: list
        A list of 'sents' objects.
     
    ns: PanedWindow
        A tkinter frame to display the frequency rank of words.
    
    words: PanedWindow
        A tkinter frame to display words.
        
    docs: PanedWindow
        A tkinter frame to display the documents in which a word appears.
        
    sents: PanedWindow
        A tkinter frame to display the sentences a given document in 
        which a word appears.
        
    txt: Entry
        A tkinter widget to enter a word to display.
        
    top_n: Entry
        A tkinter widget to enter how many of the most common words
        to display.
        
    nth: Entry
        A tkinter widget to enter the frequency rank of a word to display.   
        
    # noun_state: IntVar
    #     A tkinter variable to hold the state of the noun Checkbutton.

    # verb_state: IntVar
    #     A tkinter variable to hold the state of the verb Checkbutton.
        
    # adj_state: IntVar
    #     A tkinter variable to hold the state of the adj Checkbutton.
        
    # adv_state: IntVar
    #     A tkinter variable to hold the state of the adverb Checkbutton.
        
    # other_state: IntVar
    #     A tkinter variable to hold the state of the other Checkbutton.

    pos_states: dict
        Keeps track of the states of the POS Checkbuttons.
        
    noun: Checkbutton
        A tkinter widget to choose whether nouns are displayed.
        
    verb: Checkbutton
        A tkinter widget to choose whether verbs are displayed.
        
    adj: Checkbutton
        A tkinter widget to choose whether adjectives are displayed.
        
    adv: Checkbutton
        A tkinter widget to choose whether adverbs are displayed.
        
    other: Checkbutton
        A tkinter widget to choose whether other parts of speech are 
        displayed.
    """
    
    def __init__(self):
        self.sentences = []
        self.word_dict = None
        self.doc_dict = None
        self.pos_dict = None
        self.word_freqs = None
        self.most_common = None


        self.n_frames = []
        self.word_frames = []
        self.doc_frames = []
        self.sent_frames = []
        
        self.ns = None
        self.words = None
        self.docs = None
        self.sents = None
        
        self.txt = None
        self.top_n = None
        self.nth = None
        
        # self.noun_state = None
        # self.verb_state = None
        # self.adj_state = None
        # self.adv_state = None
        # self.other_state = None

        self.pos_states = {}
        
        self.noun = None
        self.verb = None
        self.adj = None
        self.adv = None
        self.other = None
        
    def convert_pos_tag(self, tag):
        """Convert Treebank POS tags to wordbank POS tags."""
        if tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('RB'):
            return wordnet.ADV
        else:
            return 'other'

    def clear_words(self):
        """Destroy all the frames in the bottom pane."""
        for i in range(len(self.n_frames)):
            self.n_frames[i].destroy()
            self.word_frames[i].destroy()
            self.doc_frames[i].destroy()
            self.sent_frames[i].destroy()
        self.n_frames = []
        self.word_frames = []
        self.doc_frames = []
        self.sent_frames = []
    
    def show_chosen_word(self):
        """Get a user entered value word_choice to display."""
        try:
            word_choice = self.txt.get()
            word_choice = word_choice.lower()
            tag = nltk.pos_tag([word_choice])
            if self.convert_pos_tag(tag[0][1]) is not None:
                lemm = stem.WordNetLemmatizer()
                word_choice = lemm.lemmatize(tag[0][0], self.convert_pos_tag(tag[0][1]))
            if word_choice in self.word_freqs.keys():
                self.clear_words()
                self.display(word_choice=word_choice)
            else:
                tk.messagebox.showerror('Error', 
                                        'That word does not appear in this document.')
        except:
            pass
                
    def show_top_words(self):
        """Get a user entered value how_many to display that many words."""
        try:
            how_many = eval(self.top_n.get())
            if how_many < len(self.word_freqs):
                self.clear_words()
                valid_ns = []
                n = 0
                while len(valid_ns) < how_many:
                    word = self.most_common[n][0]
                    pos = self.pos_dict[word]
                    for p in pos:
                        if self.pos_states[p].get() == 1:
                            valid_ns.append(n)
                            break
                    n += 1
                for i in valid_ns:
                    self.display(n=i)
            else:
                tk.messagebox.showerror('Error', 
                                        f'Enter a value less than {len(self.word_freqs)}.')
        except NameError:
            tk.messagebox.showerror('Error', 'You must enter a number.')
        except:
            pass  
       
    def show_nth_word(self):
        """Get a user entered value n to display the nth word."""
        try:
            n = eval(self.nth.get()) - 1
            if n < len(self.word_freqs):
                self.clear_words()
                self.display(n=n)
            else:
                tk.messagebox.showerror('Error', 
                                    f'Enter a value less than {len(self.word_freqs)}')  
        except NameError:
            tk.messagebox.showerror('Error', 'You must enter a number.') 
        except:
            pass

    def display(self, n=0, word_choice=None):
        """Display a word, its frequency rank, and the sentences in which it appears."""
        self.n_frames.append(tk.LabelFrame(self.ns))
        self.n_frames[-1].pack(fill='both', expand=True)        
        self.word_frames.append(tk.LabelFrame(self.words))
        self.word_frames[-1].pack(fill='both', expand=True)
        self.doc_frames.append(tk.LabelFrame(self.docs))
        self.doc_frames[-1].pack(fill='both', expand=True)
        self.sent_frames.append(tk.LabelFrame(self.sents))
        self.sent_frames[-1].pack(fill='both', expand=True)
        if word_choice is not None:
            nth_word = word_choice
            n = self.most_common.index((nth_word, self.word_freqs[nth_word]))
        else:
            nth_word = self.most_common[n][0]
        nth = tk.Text(self.n_frames[-1], width=5, height=5)
        nth.insert('end', n+1)
        nth.config(state='disabled')
        nth.pack()
        word = tk.Text(self.word_frames[-1], width=15, height=5)
        word.insert('end', nth_word)
        word.config(state='disabled')
        word.pack()
        nth_docs = self.doc_dict[nth_word]
        doc_list = tk.Text(self.doc_frames[-1], width=10, height=5)
        for i in range(len(nth_docs)):
            doc_list.insert('end', f'{nth_docs[i]}\n')
        doc_list.config(state='disabled')
        doc_list.pack(fill='both', expand=True)
        nth_sents = [self.sentences[j] for j in self.word_dict[nth_word]]
        scrollbar = tk.Scrollbar(self.sent_frames[-1])
        scrollbar.pack(side='right', fill='y')
        sent_list = tk.Text(self.sent_frames[-1], wrap='word', height=5, 
                            yscrollcommand=scrollbar.set)
        for i in range(len(nth_sents)):
            sent_list.insert('end', f'{nth_sents[i]}\n\n')
        sent_list.config(state='disabled')
        sent_list.pack(fill='both', expand=True)
        scrollbar.config(command=sent_list.yview)
        
    def open_file(self):
        """Open a .txt document, process it, and display the most common word."""
        self.word_freqs = Counter()
        self.word_dict = defaultdict(list)
        self.doc_dict = defaultdict(list)
        self.pos_dict = defaultdict(list)
        lemmas = []
        file_names = tk.filedialog.askopenfilenames()
        for file in file_names:
            doc_name = os.path.split(file)[1]
            with open(file, 'r') as doc:
                text = doc.read()
                sentences = sent_tokenize(text)
            for i, sent in enumerate(sentences):
                sent = [word for word in re.findall('\w+', sent.lower()) 
                        if word not in stopwords.words('english')]
                tagged = nltk.pos_tag(sent)
                tagged = [(t[0], self.convert_pos_tag(t[1])) for t in tagged]
                lemm = stem.WordNetLemmatizer()
                sent = [lemm.lemmatize(t[0], t[1]) if t[1] is not 'other' else t[0] 
                        for t in tagged]
                tagged_lemmas = zip(sent, [t[1] for t in tagged])
                for word, pos in tagged_lemmas:
                    if doc_name not in self.doc_dict[word]:
                        self.doc_dict[word].append(doc_name)
                    if pos not in self.pos_dict[word]:
                        self.pos_dict[word].append(pos)
                lemmas.append(sent)
                self.word_freqs.update(sent)
            self.sentences = self.sentences + sentences
        for word in self.word_freqs.keys():
            self.word_dict[word] = [i for i in range(len(lemmas)) 
                                    if word in lemmas[i]]
        self.most_common = self.word_freqs.most_common()
        self.clear_words()
        self.display()
            
    def begin(self):
        """Create the UI to display the most common words."""
        root = tk.Tk()
        root.title('Create Hashtags')
        root.geometry('800x600')
        
        menubar = tk.Menu(root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label='Open...', command=self.open_file)
        filemenu.add_separator()
        filemenu.add_command(label='Exit', command=root.quit)
        menubar.add_cascade(label='File', menu=filemenu)
        root.config(menu=menubar)

        main = tk.PanedWindow(root, orient='vertical')
        main.pack(fill='both', expand=True)
        
        top = tk.PanedWindow(main)
        main.add(top)
        bottom = tk.PanedWindow(main)
        main.add(bottom)

        tk.Label(top, text='Show top n words: ', font=('Helvetica 12 bold')).grid(row=0,column=0)
        self.top_n = tk.Entry(top, width=10)
        self.top_n.grid(row=1,column=0)
        tk.Button(top, text='Show', command=self.show_top_words).grid(row=2,column=0)

        tk.Label(top, text='Include parts of speech:', font=('Helvetica 12 bold')).grid(row=0,column=1)
        self.pos_states[wordnet.NOUN] = tk.IntVar(value=1)
        self.noun = tk.Checkbutton(top, text='Noun', variable=self.pos_states[wordnet.NOUN], command=self.show_top_words)
        self.noun.grid(row=1, column=1)
        self.pos_states[wordnet.VERB] = tk.IntVar(value=1)
        self.verb = tk.Checkbutton(top, text='Verb', variable=self.pos_states[wordnet.VERB], command=self.show_top_words)
        self.verb.grid(row=1, column=2)
        self.pos_states[wordnet.ADJ] = tk.IntVar(value=1)
        self.adj = tk.Checkbutton(top, text='Adjective', variable=self.pos_states[wordnet.ADJ], command=self.show_top_words)
        self.adj.grid(row=1, column=3)
        self.pos_states[wordnet.ADV] = tk.IntVar(value=1)
        self.adv = tk.Checkbutton(top, text='Adverb', variable=self.pos_states[wordnet.ADV], command=self.show_top_words)
        self.adv.grid(row=2, column=1)
        self.pos_states['other'] = tk.IntVar(value=1)
        self.other = tk.Checkbutton(top, text='Other', variable=self.pos_states['other'], command=self.show_top_words)
        self.other.grid(row=2, column=2)
        
        tk.Label(top, text='Show nth word: ', font=('Helvetica 12 bold')).grid(row=0,column=4)
        self.nth = tk.Entry(top, width=10)
        self.nth.grid(row=1,column=4)
        tk.Button(top, text='Show', command=self.show_nth_word).grid(row=2,column=4)

        tk.Label(top, text='Show a word: ', font=('Helvetica 12 bold')).grid(row=0,column=5)
        self.txt = tk.Entry(top, width=10)
        self.txt.grid(row=1,column=5)
        tk.Button(top, text='Show', command=self.show_chosen_word).grid(row=2,column=5)
        

        self.ns = tk.PanedWindow(bottom, orient='vertical')
        bottom.add(self.ns)
        tk.Label(self.ns, text='n', width=5).pack()
        self.words = tk.PanedWindow(bottom, orient='vertical')
        bottom.add(self.words)
        tk.Label(self.words, text='Words', width=10).pack()
        self.docs = tk.PanedWindow(bottom, orient='vertical')
        bottom.add(self.docs)
        tk.Label(self.docs, text='Documents', width=10).pack()
        self.sents = tk.PanedWindow(bottom, orient='vertical')
        bottom.add(self.sents)
        tk.Label(self.sents, text='Sentences').pack()
        
        main.mainloop()

top_words = TopWords()
top_words.begin()