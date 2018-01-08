import scipy.sparse as sp
import numpy as np
import os.path
from PIL import Image
import tensorflow as tf
from collections import defaultdict
import random
from scipy.sparse import dok_matrix, lil_matrix
from tqdm import tqdm
import csv
from gensim.models import Word2Vec
import string
class Dataset(object):

    def __init__(self, rating_matrix_path,num_negatives,data_set='citeulike-a',text_path=None,word2vec_model_path=None):
        pass
        self.epoch=0
        self.index_in_epoch=0
        self.num_negatives=num_negatives
        #choose citeulike or ml-1m
        #citeulike data
        self.trainMatrix,self.validationMatrix,self.testMatrix,self.user_num,self.item_num,self.new_item_index=self.read_citeulike_rating_matrix(rating_matrix_path)
        #ml-1m data
        #self.trainMatrix,self.validationMatrix,self.testMatrix,self.user_num,self.item_num=self.load_rating_file_as_matrix(path)
        self.test=self.get_BPR_test_instances()
        self.BPR_user_input,self.BPR_positive_item,self.BPR_negative_item,self.train_num=self.get_BPR_train_instances(self.trainMatrix)
        if data_set=='citeulike-a':
            self.doc_index,self.doc_array_reverse,self.doc_mask,self.doc_mask_bool,self.word_vec=self.read_citeulike_a_text(text_path,word2vec_model_path)
        elif data_set=='citeulike-t':
            self.doc_index,self.doc_array_reverse,self.doc_mask,self.doc_mask_bool,self.word_vec=self.read_citeulike_t_text(text_path,word2vec_model_path)

    def next_BPR_batch(self,batch_size):
        start=self.index_in_epoch
        self.index_in_epoch+=batch_size
        if self.index_in_epoch>self.train_num:
            self.epoch+=1
            self.BPR_user_input,self.BPR_positive_item,self.BPR_negative_item,_=self.get_BPR_train_instances(self.trainMatrix)
            start=0
            self.index_in_epoch=batch_size
        end=self.index_in_epoch

        return self.BPR_user_input[start:end],self.BPR_positive_item[start:end],self.BPR_negative_item[start:end]

    def get_doc(self):
        return self.doc_index,self.doc_array_reverse,self.doc_mask,self.doc_mask_bool,self.word_vec

    #get test matrix
    def get_BPR_test_instances(self):
        test=np.zeros([self.user_num,self.item_num],dtype=np.float32)
        for i in xrange(self.user_num):
            for j in xrange(self.item_num):
                if self.testMatrix.has_key((i,j)):
                    test[i][j]=1
        return test
    #get all test data for one user
    #BPR_test:[test_num]
    #labels:[test_num]
    def get_BPR_test(self,user_id):
        BPR_test=[]
        labels=[]
        for i in xrange(self.item_num):
            if self.trainMatrix.has_key((user_id, i)) or self.validationMatrix.has_key((user_id, i)):
                continue
            if self.testMatrix.has_key((user_id, i)):
                labels.append(1)
            else:
                labels.append(0)
            BPR_test.append(i)
        BPR_test=np.array(BPR_test)
        labels=np.array(labels)
        #BPR_test=np.arange(self.item_num)
        #labels=self.test[user_id]
        return BPR_test,labels

    #using CML method
    #BPR_user_input:[instances]
    #BPR_positive_item:[instances]
    #BPR_negative_item:[instances,num_neg]
    def get_BPR_train_instances(self,train):
        BPR_user_input,BPR_positive_item,BPR_negative_item=[],[],[]
        for (u,i) in train.keys():
            # positive instances
            BPR_user_input.append(u)
            BPR_positive_item.append(i)
            one_negative_item_group=[]
            for num in xrange(self.num_negatives):
                #negative instances
                j = np.random.randint(self.item_num)
                while self.trainMatrix.has_key((u, j)) or j in one_negative_item_group:
                    j = np.random.randint(self.item_num)
                one_negative_item_group.append(j)
            one_negative_item_group=np.array(one_negative_item_group)
            BPR_negative_item.append(one_negative_item_group)
        BPR_user_input=np.array(BPR_user_input)
        BPR_positive_item=np.array(BPR_positive_item)
        BPR_negative_item=np.array(BPR_negative_item)
        train_num=BPR_user_input.shape[0]
        #shuffle
        perm=np.arange(train_num)
        np.random.shuffle(perm)
        BPR_user_input=BPR_user_input[perm]
        BPR_positive_item=BPR_positive_item[perm]
        BPR_negative_item=BPR_negative_item[perm]
        return BPR_user_input,BPR_positive_item,BPR_negative_item,train_num

    def get_user_item_num(self):
        return self.user_num,self.item_num

    def get_train_num(self):
        return self.train_num

    def get_epoch(self):
        return self.epoch

    #split data into train,validation,test
    def split_data(self,user_item_matrix, split_ratio=(4, 0, 1), seed=1):
        np.random.seed(seed)
        train = dok_matrix(user_item_matrix.shape)
        validation = dok_matrix(user_item_matrix.shape)
        test = dok_matrix(user_item_matrix.shape)
        user_item_matrix = lil_matrix(user_item_matrix)
        for user in tqdm(range(user_item_matrix.shape[0]), desc="Split data into train/valid/test"):
            items = list(user_item_matrix[user, :].nonzero()[1])
            if len(items) >= 5:

                np.random.shuffle(items)

                train_count = int(len(items) * split_ratio[0] / sum(split_ratio))
                valid_count = int(len(items) * split_ratio[1] / sum(split_ratio))

                for i in items[0: train_count]:
                    train[user, i] = 1
                for i in items[train_count: train_count + valid_count]:
                    validation[user, i] = 1
                for i in items[train_count + valid_count:]:
                    test[user, i] = 1
        print("{}/{}/{} train/valid/test samples".format(
            len(train.nonzero()[0]),
            len(validation.nonzero()[0]),
            len(test.nonzero()[0])))
        return train, validation, test

    def read_citeulike_rating_matrix(self,path):
        user_dict = defaultdict(set)
        for u, item_list in enumerate(open(path).readlines()):
            items = item_list.strip().split(" ")
            for item in items:
                user_dict[u].add(int(item))

        n_users = len(user_dict)
        n_items = max([item for items in user_dict.values() for item in items]) + 1

        user_item_matrix = dok_matrix((n_users, n_items), dtype=np.int32)
        for u, item_list in enumerate(open(path).readlines()):
            items = item_list.strip().split(" ")
            for item in items:
                user_item_matrix[u,int(item)]=1

        #get tag
        # n_features = 0
        # for l in open(tag_path).readlines():
        #     items = l.strip().split(" ")
        #     if len(items) >= tag_occurence_thres:
        #         n_features += 1
        # print("{} features over tag_occurence_thres ({})".format(n_features, tag_occurence_thres))
        # features = dok_matrix((n_items, n_features), dtype=np.int32)
        # feature_index = 0
        # for l in open(tag_path).readlines():
        #     items = l.strip().split(" ")
        #     if len(items) >= tag_occurence_thres:
        #         features[[int(i) for i in items], feature_index] = 1
        #         feature_index += 1

        new_item_index,user_item_matrix=self.sort_item(user_item_matrix,n_users,n_items)
        train,validation,test=self.split_data(user_item_matrix)

        

        return train,validation,test,n_users,n_items,new_item_index

    #sort the items according to their frequency
    def sort_item(self,matrix,user_num,item_num):
        item_count=np.zeros([item_num],dtype=np.int32)
        for (u,i) in matrix.keys():
            item_count[i]+=1

        new_item_index=np.argsort(-item_count)
        new_item_index_transpose=np.zeros([item_num],dtype=np.int32)
        for i in xrange(item_num):
            new_item_index_transpose[new_item_index[i]]=i
        user_item_matrix = dok_matrix((user_num, item_num), dtype=np.int32)
        for (u,i) in matrix.keys():
            new_i=new_item_index_transpose[i]
            user_item_matrix[u,new_i]=1
        # new_features=dok_matrix((item_num,features_num),dtype=np.int32)
        # for (i,f) in features.keys():
        #     new_i=new_item_index_transpose[i]
        #     new_features[new_i,f]=1
        return new_item_index,user_item_matrix

    #sort the words according to their frequency
    def sort_word(self,doc):
        pass
        dic_count={}
        for paper in doc:
            for word in paper:
                if(dic_count.get(word,-1)==-1):
                    dic_count[word]=1
        word_num=len(dic_count)
        dic_index={}
        index=0
        frequency=np.zeros(word_num,dtype=np.int32)
        for paper in doc:
            for word in paper:
                if(dic_index.get(word,-1)==-1):    
                    dic_index[word]=index
                    frequency[index]+=1
                    index+=1
                else:
                    frequency[dic_index[word]]+=1
        new_word_index=np.argsort(-frequency)
        new_word_index_transpose=np.zeros([word_num],dtype=np.int32)
        for i in xrange(word_num):
            new_word_index_transpose[new_word_index[i]]=i
        dic_sorted={}
        for paper in doc:
            for word in paper:
                index=dic_index[word]
                new_index=new_word_index_transpose[index]
                dic_sorted[word]=new_index
        return dic_sorted

    #return word2vec of text
    #doc_array:[item_num,max_words_per_doc]
    #doc_array_reverse:[item_num,max_words_per_doc]
    #mask:[item_num]  save the number of words per doc
    #mask_bool:[item_num,max_words_per_doc]  1 for valid word,0 for invalid word
    #word_vec:[words] vector of words
    def read_citeulike_t_text(self,text_path,model_path):
        doc=[]
        punctuation='!"#$%&\'()*+,-/:;.<=>?@[\\]^_`{|}~'
        nextline='\n'
        with open(text_path, "r") as f:
            paper = f.readline()
            #one line text, one line number
            text=False
            while paper != None and paper != "":
                if text:
                    text=False
                    paper=paper.lower()
                    for i in punctuation:
                        paper=paper.replace(i,' ')
                    for i in nextline:
                        paper=paper.replace(i,' ')
                    #print line
                    array=line.split(' ')
                    while '' in array:
                        array.remove('')
                    doc.append(array)
                    paper=f.readline()
                else:
                    text=True
                    paper=f.readline()
        
        #remove common words and words that only appear once
        frequency = defaultdict(int)
        stoplist = set('for a of the and on at to in with by about under b c d e f g h j k l m n o p q r s t u v w x y z'.split())
        for line in doc:
            for word in line:
                frequency[word]+=1
        doc=[[word for word in line if frequency[word] >4 and word not in stoplist and  not word.isdigit()]  
                for line in doc]  


        #cut the line if it is too long
        max_words_per_doc=100
        for i in xrange(len(doc)):
            if len(doc[i])>max_words_per_doc:
                doc[i]=doc[i][0:max_words_per_doc]


        dic=self.sort_word(doc)#dictionary from word to index
        #add word dictionary. doc will only save indics
        #dic={} #dictionary from word to index
        doc_index=[]#same size as doc. but it save indics instead of word
        index=0
        for paper in doc:
            paper_index=[]
            for word in paper:
                paper_index.append(dic[word])
            doc_index.append(paper_index)
        #print doc_index

        #convert list into array
        #get mask for each sent
        doc_array=np.zeros([len(doc_index),max_words_per_doc],dtype=np.int32)
        doc_array_reverse=np.zeros([len(doc_index),max_words_per_doc],dtype=np.int32)
        mask=np.zeros([len(doc_index)],dtype=np.int32)
        mask_bool=np.zeros([len(doc_index),max_words_per_doc],dtype=np.float32)
        for i in xrange(len(doc_index)):
            mask[i]=len(doc_index[i])
            for j in xrange(len(doc_index[i])):
                doc_array[i][j]=doc_index[i][j]
                doc_array_reverse[i][mask[i]-1-j]=doc_index[i][j]
                mask_bool[i][j]=1
        #get word vectors
        model=Word2Vec.load(model_path)
        word_vec=np.zeros([len(dic)+1,model['computer'].shape[0]],dtype=np.float32) #size:word_num*vector_per_word
        for word in dic:
            vec=model[word]
            word_vec[dic[word],:]=vec

        doc_array=doc_array[self.new_item_index]
        doc_array_reverse=doc_array_reverse[self.new_item_index]
        mask=mask[self.new_item_index]
        # # Mask of valid places
        # mask = doc_array[:,:,:]>0
        # mask=mask.astype(np.float32)
        # mask=np.reshape(np.repeat(mask,word_vec.shape[1]),[mask.shape[0],mask.shape[1],word_vec.shape[1]])
        print 'read text data complete.'
        return doc_array,doc_array_reverse,mask,mask_bool,word_vec

    #return word2vec of text
    #doc_array:[item_num,max_words_per_doc]
    #doc_array_reverse:[item_num,max_words_per_doc]
    #mask:[item_num]  save the number of words per doc
    #mask_bool:[item_num,max_words_per_doc]  1 for valid word,0 for invalid word
    #word_vec:[words] vector of words
    def read_citeulike_a_text(self,text_path,model_path):
        doc=[]
        with open(text_path,'r') as f:
            reader=csv.reader(f)
            first=1
            for row in reader:
                if first>0:
                    first=0
                    continue
                line=row[1].lower()+' '+row[4].lower()
                #remove punctuation marks
                for i in '{}':
                    line=line.replace(i,'')
                for i in string.punctuation:
                    line=line.replace(i,' ')
                #print line
                array=line.split(' ')
                while '' in array:
                    array.remove('')
                doc.append(array)
        #remove common words and words that only appear once
        frequency = defaultdict(int)
        stoplist = set('for a of the and on at to in with by about under b c d e f g h j k l m n o p q r s t u v w x y z'.split())
        for line in doc:
            for word in line:
                frequency[word]+=1
        doc=[[word for word in line if frequency[word] >4 and word not in stoplist and  not word.isdigit()]  
                for line in doc]  


        #cut the line if it is too long
        max_words_per_doc=100
        for i in xrange(len(doc)):
            if len(doc[i])>max_words_per_doc:
                doc[i]=doc[i][0:max_words_per_doc]


        dic=self.sort_word(doc)#dictionary from word to index
        #add word dictionary. doc will only save indics
        #dic={} #dictionary from word to index
        doc_index=[]#same size as doc. but it save indics instead of word
        index=0
        for paper in doc:
            paper_index=[]
            for word in paper:
                paper_index.append(dic[word])
            doc_index.append(paper_index)
        #print doc_index

        #convert list into array
        doc_array=np.zeros([len(doc_index),max_words_per_doc],dtype=np.int32)
        doc_array_reverse=np.zeros([len(doc_index),max_words_per_doc],dtype=np.int32)
        mask=np.zeros([len(doc_index)],dtype=np.int32)
        mask_bool=np.zeros([len(doc_index),max_words_per_doc],dtype=np.float32)
        for i in xrange(len(doc_index)):
            mask[i]=len(doc_index[i])
            for j in xrange(len(doc_index[i])):
                doc_array[i][j]=doc_index[i][j]
                doc_array_reverse[i][mask[i]-1-j]=doc_index[i][j]
                mask_bool[i][j]=1
        #get word vectors
        model=Word2Vec.load(model_path)
        word_vec=np.zeros([len(dic)+1,model['computer'].shape[0]],dtype=np.float32) #size:word_num*vector_per_word
        for word in dic:
            vec=model[word]
            word_vec[dic[word],:]=vec

        doc_array=doc_array[self.new_item_index]
        doc_array_reverse=doc_array_reverse[self.new_item_index]
        mask=mask[self.new_item_index]
        # # Mask of valid places
        # mask = doc_array[:,:,:]>0
        # mask=mask.astype(np.float32)
        # mask=np.reshape(np.repeat(mask,word_vec.shape[1]),[mask.shape[0],mask.shape[1],word_vec.shape[1]])
        print 'read text data complete.'
        return doc_array,doc_array_reverse,mask,mask_bool,word_vec