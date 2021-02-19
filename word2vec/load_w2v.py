import os
import re
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


def load_embeddings(path, binary=False):

    """
    This function loads embedding spaces of a variety of formats.
    """

    try:
        # load files formatted according to word2vec style
        embeddings = KeyedVectors.load_word2vec_format(path, binary=binary)

    except ValueError:

        # load files formatted according to GloVe style
        folder, filename = os.path.split(path)
        outfile = os.path.join(folder, '.'.join([filename, 'word2vec']))

        try:
            glove2word2vec(path, outfile)
            embeddings = KeyedVectors.load_word2vec_format(outfile, binary=False)
            os.remove(outfile)

        except ValueError:

            # load files with unclear formatting: only consider lines containing an embedding (string+separator+number)
            outfile_glovelike = os.path.join(folder, '.'.join([filename, 'glovelike']))
            with open(outfile_glovelike, 'wb') as fout:
                with open(path, 'rb') as fin:
                    for line in fin:
                        # only print lines which start with a string followed by a separator (tab, space, comma) and a
                        # number, possibly negative
                        if re.match(r'^[A-z]+[\t ,]-?[0-9]', line):
                            fout.write(line)

            outfile = os.path.join(folder, '.'.join([filename, 'word2vec']))
            embeddings = KeyedVectors.load_word2vec_format(outfile_glovelike, binary=False)
            os.remove(outfile)
            os.remove(outfile_glovelike)

    return embeddings
