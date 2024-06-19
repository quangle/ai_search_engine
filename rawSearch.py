import json
import string
import math
import os

class TFIDF():
    def __init__(self):
        self.load_data()
        
    def load_data(self):
        # Load docs from file
        f = open('docs.json')
        data = json.load(f)
        self.docs = data['docs']
        # check if tf_idf_list.json or ds.json are empty
        if os.stat("tf_idf_list.json").st_size == 0 or os.stat("ds.json").st_size == 0:
            # handle constructing inverted index if not done
            self.construct_inverted_index()

        # load tf_idf_list from file
        f = open('tf_idf_list.json')    
        self.tf_idf_list = json.load(f)
        # load ds from file
        f = open('ds.json')    
        self.ds = json.load(f)
        
    def construct_inverted_index(self):
        from collections import defaultdict
        stats = {
            "words": {},
            "docs": {}
        }
        for i, doc in enumerate(self.docs):
            if i not in stats['docs']:
                stats['docs'][i] = defaultdict(int)

            for word in doc.split(' '):
                if word not in stats['words']:
                    stats['words'][word] = {i}
                else:
                    stats['words'][word].add(i)

                stats['docs'][i][word] += 1

        words = stats['words'].keys()

        # Cal IDF
        idf = defaultdict(float)

        N = len(self.docs)

        for word in words:
            df = len(stats['words'][word])
            idf[word] = math.log10(N / df)

        tf_idf_list = defaultdict(lambda: defaultdict(float))
        ds = defaultdict(float)

        for doc in stats['docs']:
            d = 0
            for word in words:
                # tf đã được tính sẵn
                tf = self.__get_tf(stats['docs'][doc][word])

                # Tính giá trị tf-idf
                tf_idf = tf * idf[word]

                d += tf_idf ** 2
                # Gán tf-idf vào tf_idf_list. Cứ một từ và một văn bản sẽ có một điểm
                tf_idf_list[word][doc] = tf_idf
            # Phép tính dưới mẫu của hình bên dưới
            d_ = d ** (1/2)
            # Lưu các giá trị
            ds[doc] = self.__rounding(d_)

        # save tf_idf_list to tf_idf_list.json
        with open('tf_idf_list.json', 'w') as outfile:
            json.dump(tf_idf_list, outfile)
        # save ds to ds.json
        with open('ds.json', 'w') as outfile:
            json.dump(ds, outfile)

    def search(self, q, k):
        # Search documents using TF-IDF
        results = []
        # lookp through docs
        for i in range(len(self.docs)):
            score = 0
            i = str(i)
            # loop through words in query
            for t in q.split():
                t = t.lower()
                # need to check if query word exists in tf_idf_list
                if t in self.tf_idf_list:
                    score += self.tf_idf_list[t][i] / self.ds[i]

            results.append((score, i))
            
        results = sorted(results, key= lambda x: -x[0])
        return results[:k]

    # private methods

    def __rounding(self, num):
        return math.floor(num * 1000) / 1000

    # calculate tf
    def __get_tf(self, num):
        return self.__rounding(math.log10(num + 1))
