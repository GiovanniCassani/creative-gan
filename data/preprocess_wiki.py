from gensim.corpora.wikicorpus import WikiCorpus
import time


if __name__ == "__main__":

    print("Importing Wikipedia")
    wiki = WikiCorpus('en/enwiki-latest-pages-articles.xml.bz2',
                      lemmatize=False, dictionary={})

    print("Getting sentences")
    start = time.time()

    with open('sentences.txt', 'w') as f:
        for i, text in enumerate(wiki.get_texts()):
            f.write("%s\n" % text)
            i = i + 1
            if (i % 10000 == 0):
                print('Processed ' + str(i) + ' articles')

    end = time.time()
    print(end - start, ' seconds elapsed')
